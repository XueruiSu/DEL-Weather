"""A package for encoding and decoding weather dataset."""
# TODO(Cris): Enable type checking for the type-ignored lines in this file.
from __future__ import (
    annotations,
)  # Allows using classes defined in this file as a type annotation.

import copy
import io
import itertools
import os
import pickle
import random
import shutil
import subprocess
import threading
import time
import torch

from collections import deque
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import tqdm
import yaml
from data.seqrecord_utils.utils import (
    CONSUMER_SLEEP_INTERVAL,
    PRODUCER_SLEEP_INTERVAL,
    FileManager,
    LRUCache,
    WriterBuffer,
    stack_frame,
)
from torchdata.dataloader2 import communication


DEFAULT_MAX_RECORDFILE_SIZE = int(2e9)  # 1e9  # 1e8, 100 mb, maximum size of a single record file
DEFAULT_FILE_CACHE_SIZE = 10  # 10, maximum number of record files to keep in local disk


class _PrefetchData:
    def __init__(self, source_data_generator, buffer_size: int):
        self.run_prefetcher = True
        # python deque is thread safe for appends and pops from opposite sides.
        # ref: https://stackoverflow.com/questions/8554153/is-this-deque-thread-safe-in-python
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.source_data_generator = source_data_generator


class WSeqRecord:
    """A serialization protocal that stores a single continuous long sequence of weather data into record files.
    The protocol provides metadata of each frame to enable efficient random access of frames."""

    def __init__(self, recorddir: str, local_cache_dir: Optional[str] = None) -> None:
        """Initialize a WSeqRecord object.

        Args:
            recorddir: The directory where the data is stored.
            local_cache_dir: The directory for the local disk cache.
                When supplied, the data is first written to the local disk cache,
                and then flushed to the recorddir.
        """
        self.recorddir: str = recorddir
        os.makedirs(self.recorddir, exist_ok=True)

        self.local_cache_dir = local_cache_dir
        if local_cache_dir is not None:
            # If this is a realtive path '~', then expand it to absolute path.
            self.local_cache_dir = os.path.abspath(os.path.expanduser(local_cache_dir))
            if os.path.exists(self.local_cache_dir) and len(os.listdir(self.local_cache_dir)) > 0:
                print("Warning: local cache dir is not empty. Clearing it now.")
                shutil.rmtree(self.local_cache_dir, ignore_errors=True)
            os.makedirs(self.local_cache_dir, exist_ok=True)
        self.features_written: Optional[List[str]] = None

        self.meta_rank: Optional[Dict] = None
        self.num_ranks: Optional[int] = None

        # Variables used for metadata compression.
        self._meta_frame_compressed: bool = False
        self._meta_frame_load_keys: List[str] = list()
        self._meta_frame_outline: Dict[str, Any] = dict()

    @staticmethod
    def subfolder_name(rank: int, world_size: int) -> str:
        """Turn rank into the name of the corresponding subfolder."""
        return f"{rank}"

    @staticmethod
    def fileidx2name(file_idx: int) -> str:
        """Turn absolute file idx into the name of the corresponding record file."""
        return f"record_{file_idx}.bin"

    def fileidx2path(self, file_idx: int, local_cache_dir: Optional[str] = None) -> str:
        """Turn absolute file idx into relative path of the corresponding record file.

        Args:
            file_idx: The id of the record file.
            local_cache_dir: The directory to cache the record file. Defaults to None.
        Returns:
            str: Relative path to the record file.
        """
        dir = self.recorddir if local_cache_dir is None else local_cache_dir
        rank_id: int = self.meta_file[file_idx].get("rank_id", -1)  # type: ignore
        if rank_id == -1:
            # there is no rank hierarachy
            return os.path.join(dir, self.fileidx2name(file_idx))
        else:
            assert self.num_ranks is not None
            return os.path.join(
                dir,
                self.subfolder_name(rank_id, self.num_ranks),
                f"record_{self.meta_file[file_idx]['rel_file_idx']}.bin",
            )

    def recordfile_generator(
        self, frame_generator: Iterator, max_recordfile_size=DEFAULT_MAX_RECORDFILE_SIZE
    ) -> Generator[Tuple[str, bytes], None, None]:
        """Constructs a stream of record files from a stram of frames and stores the metadata
           required to randomly access any frame.

        Args:
            frame_generator: A generator of frames in dictionary format with keys
                denoting feature names and values denoting feature values.

        Yields:
            A record file path and the file contents.
        """
        try:
            write_buffer = WriterBuffer()
            num_bytes = 0
            self.meta_file = {}
            self.meta_frame = {}
            frame_idx = 0
            file_idx = 0
            for frame in frame_generator:
                if self.features_written is None:
                    self.features_written = [key for key in frame]
                if num_bytes == 0:
                    # new file
                    self.meta_file[file_idx] = {
                        "frame_idx_start": frame_idx,
                        "relative_path": self.fileidx2name(file_idx),
                    }
                    # relative path to the record file does not contain directory of the corresponding seqrecord
                self.meta_frame[frame_idx] = {
                    "file_idx": file_idx,
                    "bytes_offset": num_bytes,
                }
                num_bytes_in_frame = 0
                for feature, data in frame.items():
                    self.meta_frame[frame_idx][feature] = {  # type: ignore
                        "is_none": (
                            data.dtype == np.dtype("O") and data is None
                        ),  # this feature is essentially missing, and
                        "dtype": data.dtype,
                        "shape": data.shape,
                        "bytes_offset": num_bytes_in_frame,
                        "nbytes": data.nbytes,
                    }
                    write_buffer.write(data.tobytes())
                    num_bytes_in_frame += data.nbytes

                self.meta_frame[frame_idx]["nbytes"] = num_bytes_in_frame
                frame_idx += 1
                num_bytes += num_bytes_in_frame
                if num_bytes > max_recordfile_size:
                    # current file is big enough
                    num_bytes = 0
                    self.meta_file[file_idx]["frame_idx_end"] = frame_idx
                    write_buffer.clear()
                    yield (
                        self.fileidx2path(file_idx, local_cache_dir=self.local_cache_dir),
                        write_buffer.getvalue(),
                    )
                    file_idx += 1

            if (
                file_idx in self.meta_file
                and self.meta_file[file_idx].get("frame_idx_end", None) is None
            ):
                # there is content left in the write_buffer
                self.meta_file[file_idx]["frame_idx_end"] = frame_idx
                yield (
                    self.fileidx2path(file_idx, self.local_cache_dir),
                    write_buffer.getvalue(),
                )
                file_idx += 1
        finally:
            write_buffer.close()
            self.num_files = file_idx
            self.num_frames = frame_idx

    def put_frame(
        self,
        frame_generator: Iterator[Dict[str, np.ndarray]],
        max_recordfile_size: int = DEFAULT_MAX_RECORDFILE_SIZE,
        file_cache_size: int = DEFAULT_FILE_CACHE_SIZE,
        prefetch_buffer_size: int = 5,
    ):
        """Reads frames from frame_generator and writes them to record files.

        Args:
            frame_generator: A generator of frames in dictionary format with keys
                denoting feature names and values denoting feature values.
            max_recordfile_size: The maximum size of a record file in bytes.
            file_cache_size: The maximum number of record files to cache in memory.
            prefetch_buffer_size: The maximum number of record files to prefetch in memory.
        """
        try:
            prefetch_data = _PrefetchData(
                self.recordfile_generator(
                    frame_generator=frame_generator,
                    max_recordfile_size=max_recordfile_size,
                ),
                prefetch_buffer_size,
            )
            thread = threading.Thread(
                target=WSeqRecord.prefetch_thread_worker,
                args=(prefetch_data,),
                daemon=True,
            )
            thread.start()
            file_cache = []
            subprocesses: Deque[subprocess.Popen] = deque()
            while prefetch_data.run_prefetcher:
                if len(prefetch_data.prefetch_buffer) > 0:
                    (
                        file_path,
                        content,
                    ) = prefetch_data.prefetch_buffer.popleft()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    file_cache.append(file_path)
                    if self.local_cache_dir is not None and len(file_cache) > file_cache_size:
                        # Move record files from the disk cache to recorddir in the background.
                        subprocesses.append(
                            subprocess.Popen(
                                [
                                    "mv",
                                ]
                                + file_cache
                                + [f"{self.recorddir}/"]
                            )
                        )
                        file_cache = []
                else:
                    # TODO: Calculate sleep interval based on previous availability speed
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
                    if len(subprocesses) > 0 and subprocesses[0].poll() is not None:
                        subprocesses.popleft()

        finally:
            prefetch_data.run_prefetcher = False
            if thread is not None:
                thread.join()
                del thread
            if self.local_cache_dir is not None and len(file_cache) > 0:
                subprocesses.append(
                    subprocess.Popen(
                        [
                            "mv",
                        ]
                        + file_cache
                        + [f"{self.recorddir}/"]
                    )
                )
                file_cache = []
            for p in subprocesses:
                p.wait()

    @staticmethod
    def read_frame(
        file_desc: Union[io.BufferedReader, BinaryIO],
        metadata_frame: Dict[str, Union[int, dict]],
        features: List[str],
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single frame, return the
        decoded dictionary(feature->data(np.ndarray)) of the item.

        Args:
            file_desc (io.BufferedReader): python file object of the record file (required by numpy)
            metadata_frame (Dict[str, Any]): dict that contains meta info of a specific frame
            features (List[str]):  features requested for frame
        Returns:
            Dict[str, np.ndarray]: data
        """
        frame: Dict[str, torch.Tensor] = {}
        frame_offset = metadata_frame["bytes_offset"]
        for feature in features:
            frame[feature] = torch.from_numpy(
                np.memmap(
                    file_desc,
                    dtype=metadata_frame[feature]["dtype"],  # type: ignore
                    mode="r",
                    offset=frame_offset + metadata_frame[feature]["bytes_offset"],  # type: ignore
                    shape=metadata_frame[feature]["shape"],  # type: ignore
                )
            )
        return frame

    @staticmethod
    def next_timestep_framepairs_from_files(
        file_generator: Iterator[SeqRecordFileMetadata],
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Generates next-timestep prediction samples from a list of seqrecord files

        Args:
            fileidx_generator: Generator for seqrecord bin files
            frame_cache_cap: Size of LRU frame cache.
        Yields:
            sample: A dictionary of input and target frames and associated metadata.
        """
        file_manager = FileManager(cache_capacity=1)

        for file in file_generator:
            # For each file, generate all pairs of frames with index (i, i+1)
            filedesc = file_manager.open_file(file_idx=file.fileidx, file_path=file.path)

            input_frame = WSeqRecord.read_frame(
                filedesc,
                file.meta_frame[file.start],
                file.input_features,
            )
            input_frame = stack_frame(input_frame, file.input_features)

            for frameidx in range(file.start + 1, file.end):
                target_frame = WSeqRecord.read_frame(
                    filedesc,
                    file.meta_frame[frameidx],
                    file.target_features,
                )
                target_frame = stack_frame(target_frame, file.target_features)

                yield {
                    "input": input_frame,
                    "target": target_frame,
                    "lookahead_steps": torch.tensor(1),
                    "input_features": file.input_features,
                    "target_features": file.target_features,
                }

                input_frame = target_frame
        file_manager.close_all_files()

    @staticmethod
    def iterate_framepairs_from_files(
        record: WSeqRecord,
        file_generator: Iterator[SeqRecordFileMetadata],
        filedesc_cache_cap: int = 10,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Generates random lead time prediction samples from a list of seqrecord files
           by sammpling a random lead time per record.

        Args:
            fileidx_generator: Generator for seqrecord bin files
            filedesc_cache_cap: Size of LRU file descriptor cache.
            frame_cache_cap: Size of LRU frame cache.
        Yields:
            A dictionary of input and target frames and associated metadata.
        """
        file_manager = FileManager(
            cache_capacity=filedesc_cache_cap,
        )

        for file in file_generator:
            filedesc4input = file_manager.open_file(file_idx=file.fileidx, file_path=file.path)

            # no target frame to predict for the last frame
            for frameidx4input in range(  # type: ignore
                file.start,
                min(file.end, record.num_frames - 1),  # type: ignore
            ):
                input_frame = WSeqRecord.read_frame(
                    filedesc4input,
                    file.meta_frame[frameidx4input],
                    file.input_features,
                )
                # get the target frame for prediction, both start, stop inclusive
                lookahead_steps = min(
                    random.randint(1, record.framereader_args["max_pred_steps"]),
                    record.num_frames - 1 - frameidx4input,
                )
                frameidx4target = frameidx4input + lookahead_steps
                fileidx4target = record.meta_frame[frameidx4target]["file_idx"]
                filedesc4target = file_manager.open_file(
                    fileidx4target,
                    file_path=os.path.join(
                        record.recorddir,
                        record.meta_file[fileidx4target]["relative_path"],  # type: ignore
                    ),
                )
                target_frame = WSeqRecord.read_frame(
                    filedesc4target,
                    record.meta_frame[frameidx4target],  # type: ignore
                    record.framereader_args["target_features"],
                )
                # colllate input and target frames so that input and target frame are np.ndarray
                # each feature is a two-dimensional np.ndarray
                # output is channelxheightxwidth
                input_frame = stack_frame(input_frame, record.framereader_args["input_features"])
                target_frame = stack_frame(target_frame, record.framereader_args["target_features"])

                yield {
                    "input": input_frame,
                    "target": target_frame,
                    "lookahead_steps": torch.tensor(lookahead_steps),
                    "input_features": record.framereader_args["input_features"],
                    "target_features": record.framereader_args["target_features"],
                }
        file_manager.close_all_files()

    @staticmethod
    def fast_iterate_framepairs_from_files(
        record: WSeqRecord,
        file_generator: Iterator[SeqRecordFileMetadata],
        filedesc_cache_cap: int = 10,
        frame_cache_cap: int = 20,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Fast generation of random lead time samples from a list of seqrecord files,
           by sampling a single random lead time per file.

        Args:
            record: WSeqRecord object. Used to find the target frame.
            fileidx_generator: Generator for seqrecord bin files
            filedesc_cache_cap: Size of LRU file descriptor cache.
            frame_cache_cap: Size of LRU frame cache.
        Yields:
            A dictionary of input and target frames and associated metadata.
        """
        file_manager = FileManager(
            cache_capacity=filedesc_cache_cap,
        )
        # Input and target frames might have different features,
        # so we need two separate frame caches.
        # TODO(Cris): Decide the size of the cache based on number of frames per file.
        inp_frame_cache = LRUCache(frame_cache_cap)
        tar_frame_cache = LRUCache(frame_cache_cap)

        for file in file_generator:
            filedesc4input = file_manager.open_file(file_idx=file.fileidx, file_path=file.path)

            # Get the file where the target frames live.
            max_pred_steps: int = record.framereader_args["max_pred_steps"]
            lookahead_steps = min(
                random.randint(1, max_pred_steps),
                record.num_frames - 1 - file.start,
            )
            start_target_frame_id = file.start + lookahead_steps
            fileidx4target = record.meta_frame[start_target_frame_id]["file_idx"]
            target_file = SeqRecordFileMetadata(record, fileidx4target)
            filedesc4target = file_manager.open_file(fileidx4target, file_path=target_file.path)

            for frameidx4input, frameidx4target in itertools.product(
                range(file.start, file.end), range(target_file.start, target_file.end)
            ):
                if (frameidx4input >= frameidx4target) or (
                    frameidx4target - frameidx4input > max_pred_steps
                ):
                    continue

                input_frame = inp_frame_cache.get(frameidx4input)
                input_frame = (
                    WSeqRecord.read_frame(
                        filedesc4input,
                        file.meta_frame[frameidx4input],
                        file.input_features,
                    )
                    if input_frame is None
                    else input_frame
                )
                inp_frame_cache.put(frameidx4input, input_frame)

                target_frame = tar_frame_cache.get(frameidx4target)
                target_frame = (
                    WSeqRecord.read_frame(
                        filedesc4target,
                        target_file.meta_frame[frameidx4target],  # type: ignore
                        target_file.target_features,
                    )
                    if target_frame is None
                    else target_frame
                )
                tar_frame_cache.put(frameidx4target, target_frame)

                input_frame = stack_frame(input_frame, file.input_features)
                target_frame = stack_frame(target_frame, file.target_features)

                yield {
                    "input": input_frame,
                    "target": target_frame,
                    "frameidx4input": frameidx4input,
                    "frameidx4target": frameidx4target,
                    "lookahead_steps": torch.tensor(frameidx4target - frameidx4input),
                    "input_features": record.framereader_args["input_features"],
                    "target_features": record.framereader_args["target_features"],
                }
        file_manager.close_all_files()

    @staticmethod
    def prefetch_thread_worker(prefetch_data: _PrefetchData):
        """Thread worker for prefetching data from the source data generator.

        Args:
            prefetch_data: PrefetchData object containing the source data generator.
        """
        itr = iter(prefetch_data.source_data_generator)
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if (
                len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size
                and not stop_iteration
            ):
                try:
                    item = next(itr)
                    prefetch_data.prefetch_buffer.append(item)
                except StopIteration:
                    stop_iteration = True
                # shc: probably not necessary for now
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            elif stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                prefetch_data.run_prefetcher = False
            else:  # Buffer is full, waiting for main thread to consume items
                # TODO: Calculate sleep interval based on previous consumption speed
                time.sleep(PRODUCER_SLEEP_INTERVAL)

    def dump_record(self, rank: Optional[int] = None) -> None:
        """Save the dataset metadata (used for random frame access) into a pickle file.
           Additionally saves an equivalent yaml file for visual inspection.

        Note: We are saving attribute dict instead of pickled class.
            Pickling class and loading it is a mess because of path issues.

        Args:
            rank: Rank of the process generating the data.
                Defaults to None when using the main process.
        """

        # Determine file to write the state to.
        path = os.path.join(
            self.recorddir,
            f"record_{rank}.dict" if rank is not None else "record_all.dict",
        )

        # Save the state of this object.
        state = copy.deepcopy(self.__dict__)

        # Perform frame compression to greatly reduce the size of the index. Do this by assuming
        # that all frames store the same variables!
        state["_meta_frame_compressed"] = True  # Indicate that compression is used.
        state["_meta_frame_outline"] = state["meta_frame"][0]  # General form of frames
        # Compress by storing only these frame keys:
        save_keys = ["bytes_offset", "file_idx", "nbytes", "rel_frame_idx"]

        def compress_frame(x: dict) -> list:
            """Compress frame.

            Args:
                x (dict): Original frame.

            Returns:
                list: Compressed frame.
            """
            # In the tests, not all element of `save_keys` are in `x`. It's not exactly clear why.
            return [x[k] if k in x else None for k in save_keys]

        state["_meta_frame_load_keys"] = save_keys
        state["meta_frame"] = {k: compress_frame(v) for k, v in state["meta_frame"].items()}

        # Save index.
        with open(path, "wb") as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

        with open(Path(path).with_suffix(".yaml"), mode="w") as f:  # type: ignore
            f.write("# Configs for human inspection only!\n")  # type: ignore
            f.write(yaml.dump(state))  # type: ignore

    @classmethod
    def load_record(cls, recorddir: str, rank: Optional[int] = None) -> WSeqRecord:
        """Return an instance of seqrecord from a metadata pickle file.

        Args:
            reccorddir: path to the file that stores dict of attributes of seqrecord.
            rank: Rank of the process loading the data.

        Returns:
            WSR: an instance of record
        """
        # Determine file to load the state from.
        path = os.path.join(
            recorddir,
            "record_all.dict" if rank is None else f"record_{rank}.dict",
        )

        with open(path, mode="rb") as f:
            state = pickle.load(f)

        # Create a WSR with the restored state.
        obj = cls(recorddir=recorddir)
        state.pop("recorddir", None)
        for key, value in state.items():
            setattr(obj, key, value)

        # Inflate the frames if there were compressed.
        if obj._meta_frame_compressed:
            del obj._meta_frame_compressed

            # Extract meta-data required for inflation.
            frame_outline = obj._meta_frame_outline  # General form of frames
            load_keys = obj._meta_frame_load_keys  # Which keys to load into the general form
            del obj._meta_frame_outline
            del obj._meta_frame_load_keys

            def inflate_frame(x: list) -> dict:
                """Inflate frame.

                Args:
                    x (list): Compressed frame.

                Returns:
                    dict: Original frame.
                """
                return dict(frame_outline, **{k: v for k, v in zip(load_keys, x) if v is not None})

            # Perform inflation.
            obj.meta_frame = {k: inflate_frame(v) for k, v in obj.meta_frame.items()}  # type: ignore

        return obj

    @classmethod
    def gather_subseqrecords(
        cls,
        recorddir: str,
        world_size: int,
        rank2folder: Optional[Dict[int, str]] = None,
    ) -> WSeqRecord:
        """Gathers subseqrecord metadata from different ranks into one seqrecord metadata file.

        Args:
            recorddir: path to the directory that stores subseqrecord metadata.
            world_size: number of ranks.
            rank2folder: mapping from rank to subseqrecord folder name.
        Returns:
            An instance of the record.
        """

        # Make everything hierarchical to make it consistent
        if rank2folder is None:
            rank2folder = {i: cls.subfolder_name(i, world_size) for i in range(world_size)}
        sub_records: List[WSeqRecord] = []
        for i in tqdm.tqdm(range(world_size)):
            sub_records.append(cls.load_record(os.path.join(recorddir, rank2folder[i]), rank=i))

        # combine meta data
        features_written = sub_records[0].features_written

        # meta data on each rank collected data
        meta_rank = {}
        meta_file = {}
        meta_frame = {}
        abs_file_idx = 0
        abs_frame_idx = 0
        for i in tqdm.tqdm(range(world_size)):
            meta_rank[i] = {
                "file_idx_start": abs_file_idx,
                "file_idx_end": abs_file_idx + sub_records[i].num_files,
                "frame_idx_start": abs_frame_idx,
            }
            for j in range(sub_records[i].num_files):
                meta_file[abs_file_idx] = {
                    "relative_path": os.path.join(
                        rank2folder[i],
                        sub_records[i].meta_file[j]["relative_path"],
                    ),
                    "frame_idx_start": abs_frame_idx,
                }
                for k in range(
                    sub_records[i].meta_file[j]["frame_idx_start"],
                    sub_records[i].meta_file[j]["frame_idx_end"],
                ):
                    meta_frame[abs_frame_idx] = sub_records[i].meta_frame[k]
                    meta_frame[abs_frame_idx]["rel_frame_idx"] = k
                    meta_frame[abs_frame_idx]["file_idx"] = abs_file_idx
                    abs_frame_idx += 1
                meta_file[abs_file_idx]["frame_idx_end"] = abs_frame_idx
                abs_file_idx += 1
            meta_rank[i]["frame_idx_end"] = abs_frame_idx

        record = cls(recorddir)
        record.meta_file = meta_file
        record.meta_frame = meta_frame
        record.meta_rank = meta_rank
        record.features_written = features_written

        record.num_ranks = world_size
        record.num_files = abs_file_idx
        record.num_frames = abs_frame_idx
        return record

    def set_framereader_args(self, args: Dict[str, Any]) -> None:
        """Sets metadata required to read the frames."""
        self.framereader_args = args


class SeqRecordFileMetadata:
    """Metadata for a single SeqRecord binary file."""

    def __init__(self, record: WSeqRecord, fileidx: int) -> None:
        self.fileidx: int = fileidx
        self.meta_file: Dict = record.meta_file[fileidx]
        self.path: str = os.path.join(record.recorddir, self.meta_file["relative_path"])  # type: ignore
        self.input_features: List[str] = record.framereader_args["input_features"]
        self.target_features: List[str] = record.framereader_args["target_features"]

        assert record.features_written is not None
        self.written_features: List[str] = record.features_written

        self.start: int = self.meta_file["frame_idx_start"]  # type: ignore
        self.end: int = self.meta_file["frame_idx_end"]  # type: ignore
        self.meta_frame: Dict[int, Dict] = {
            i: record.meta_frame[i] for i in range(self.start, self.end)
        }
        self.num_records: int = self.end - self.start
