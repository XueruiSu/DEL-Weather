import logging
import math
import multiprocessing as mp
import shutil
import subprocess as sp
import time
from queue import Queue
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = ["monitor"]


def _command_exists(name: str) -> bool:
    """Check whether the command line tool with the name `name` exists.

    Args:
        name (str): Name of program.

    Returns:
        bool: Whether `name` exists or not.
    """
    return shutil.which(name) is not None


def _run_command(command: str) -> List[str]:
    """Run the command `command`.

    Args:
        command (str): Command to run.

    Returns:
        list[str]: Lines of the response with any trailing newlines removes.
    """
    return sp.check_output(command, shell=True).strip().decode().splitlines()


def _monitor_cpu(queue_cmd: mp.Queue, queue_out: mp.Queue, interval: float = 5) -> None:
    """Monitor CPU utilisation.

    Args:
        queue_cmd (:class:`multiprocessing.queues.Queue`): The queue to use for commands.
        queue_out (:class:`multiprocessing.queues.Queue`): The queue to use for logs.
        interval (float, optional): Interval at which to report in seconds. Defaults to five.
    """
    while True:
        if not queue_cmd.empty() and queue_cmd.get() == "kill":
            break
        out = _run_command(f"sar {math.ceil(interval):d} 1")
        cpu_utilisation = float(out[-1].split()[2])
        queue_out.put(("cpu/utilisation", cpu_utilisation))
        # We don't need to sleep, because `sar` takes a second to run.


def _monitor_ram(queue_cmd: mp.Queue, queue_out: mp.Queue, interval: float = 5) -> None:
    """Monitor RAM usage.

    Args:
        queue_cmd (:class:`multiprocessing.queues.Queue`): The queue to use for commands.
        queue_out (:class:`multiprocessing.queues.Queue`): The queue to use for logs.
        interval (float, optional): Interval at which to report in seconds. Defaults to five.
    """
    while True:
        if not queue_cmd.empty() and queue_cmd.get() == "kill":
            break
        out = _run_command("free -t -m")
        ram_total = float(out[1].split()[1])
        ram_used = float(out[1].split()[2])
        queue_out.put(("cpu/memory", 100 * ram_used / ram_total))
        # RAM usage shouldn't change very abruptly, so there is no need to continuously sample
        # and compute an average. We can just sleep and run `free` again.
        time.sleep(interval)


def _monitor_gpu(
    queue_cmd: mp.Queue,
    queue_out: mp.Queue,
    interval: float = 5,
    sampling_rate: float = 5,
) -> None:
    """Monitor GPU utilisation and memory usage.

    Args:
        queue_cmd (:class:`multiprocessing.queues.Queue`): The queue to use for commands.
        queue_out (:class:`multiprocessing.queues.Queue`): The queue to use for logs.
        interval (float, optional): Interval at which to report in seconds. Defaults to five.
        sampling_rate (float, optional): Rate at which to query `nvidia-smi` in Hertz. Defaults to
            five.
    """
    iters = math.ceil(interval * sampling_rate)
    while True:
        if not queue_cmd.empty() and queue_cmd.get() == "kill":
            break
        # GPU utilisation and memory usage can vary very quickly, so we sample at a higher rate
        # than we report these statistics.
        ut = 0
        mem = 0
        for _ in range(iters):
            out = _run_command(
                "nvidia-smi"
                " --query-gpu=utilization.gpu,memory.used,memory.total"
                " --format=csv,noheader,nounits"
            )
            uts, mems, totals = zip(*[tuple(map(float, line.split(","))) for line in out])
            ut += np.mean(uts) / iters
            mem += np.mean([100 * x / y for x, y in zip(mems, totals)]) / iters
            time.sleep(1 / sampling_rate)
        queue_out.put(("gpu/utilisation", ut))
        queue_out.put(("gpu/memory", mem))


def _get_current_bytes() -> Tuple[int, int]:
    """Get the total number of bytes recieved and transmitted over `eth0`.

    Returns:
        int: Number of bytes received.
        int: Number of bytes transmitted.
    """
    out = _run_command("ifconfig eth0")
    rx_bytes = int(out[4].strip().split()[4])
    tx_bytes = int(out[6].strip().split()[4])
    return rx_bytes, tx_bytes


def _get_delta_bytes(prev_rx_bytes: int, prev_tx_bytes: int) -> Tuple[int, int, int, int]:
    """Get the number of bytes recieved and transmitted over `eth0` with respect to a previous
    snapshot.

    Args:
        prev_rx_bytes (int): Previous total number of bytes received.
        prev_tx_bytes (int): Previous total number of bytes transmitted.

    Returns:
        int: Current total number of bytes received.
        int: Current total number of bytes transmitted.
        int: Number of bytes received with respect to previous snapshot.
        int: Number of bytes transmitted with respect to previous snapshot.
    """
    rx_bytes, tx_bytes = _get_current_bytes()
    rx_delta = rx_bytes - prev_rx_bytes
    tx_delta = tx_bytes - prev_tx_bytes
    return rx_bytes, tx_bytes, rx_delta, tx_delta


def _monitor_traffic(queue_cmd: mp.Queue, queue_out: mp.Queue, interval: float = 5) -> None:
    """Monitor traffic over `eth0`.

    Args:
        queue_cmd (:class:`multiprocessing.queues.Queue`): The queue to use for commands.
        queue_out (:class:`multiprocessing.queues.Queue`): The queue to use for logs.
        interval (float, optional): Interval at which to report in seconds. Defaults to five.
    """
    rx_bytes, tx_bytes = _get_current_bytes()
    t_prev = time.time()
    while True:
        if not queue_cmd.empty() and queue_cmd.get() == "kill":
            break
        rx_bytes, tx_bytes, rx_delta, tx_delta = _get_delta_bytes(rx_bytes, tx_bytes)
        t_cur = time.time()
        t_delta = t_cur - t_prev
        t_prev = t_cur
        queue_out.put(("traffic/eth0/incoming", rx_delta * 8 / t_delta / 1e9))
        queue_out.put(("traffic/eth0/outgoing", tx_delta * 8 / t_delta / 1e9))
        time.sleep(interval)


def monitor() -> Tuple[Callable[[], List[Tuple[str, float]]], Callable[[], None]]:
    """Monitor the machine in a separate process. All utilisation and memory numbers are percentages
    with respect to the maximum capacity. All network numbers are in Gbps.

    Returns:
        Callable[[], list[tuple[str, float]]]: Function which returns all recent logging calls.
            Repeatedly call this function and feed the output to a logger.
        Callable[[], None]: Function which gracefully ends the monitoring processes and joins them
            to the main process.
    """
    manager = mp.Manager()
    joiners: List[Tuple[Callable[[Optional[float]], None], Queue]] = []

    # The queue which will receive all logging calls.
    q_out = manager.Queue()

    # Setup logging of CPU utilisation.
    if _command_exists("sar"):
        q = manager.Queue()
        p = mp.Process(target=_monitor_cpu, args=(q, q_out))
        p.start()
        joiners.append((p.join, q))
    else:
        logging.warn(
            "Command line tool `sar` not found. "
            "Not logging CPU utilisation. "
            "To install `sar` with `apt`, please run `sudo apt install sysstat`."
        )

    # Setup logging of RAM usage.
    if _command_exists("free"):
        q = manager.Queue()
        p = mp.Process(target=_monitor_ram, args=(q, q_out))
        p.start()
        joiners.append((p.join, q))
    else:
        logging.warn("Command line tool `free` not found. Not logging RAM usage. ")

    # Setup logging of GPU.
    if _command_exists("nvidia-smi"):
        q = manager.Queue()
        p = mp.Process(target=_monitor_gpu, args=(q, q_out))
        p.start()
        joiners.append((p.join, q))
    else:
        logging.warn("Command line tool `nvidia-smi` not found. Not logging GPU usage. ")

    # Setup logging of network traffic.
    if _command_exists("ifconfig"):
        q = manager.Queue()
        p = mp.Process(target=_monitor_traffic, args=(q, q_out))
        p.start()
        joiners.append((p.join, q))
    else:
        logging.warn("Command line tool `ifconfig` not found. Not logging network traffic. ")

    def receive() -> List[Tuple[str, float]]:
        """Receive all logging calls.

        Returns:
            list[tuple[str, float]]: List of recent logging calls.
        """
        out = []
        while not q_out.empty():
            out.append(q_out.get())
        return out

    def join() -> None:
        """Gracefully end and join the monitoring processes."""
        for joiner, queue in joiners:
            queue.put("kill")
            joiner(None)

    return receive, join
