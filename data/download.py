import logging
import os

from multiurl import Downloader


logger = logging.getLogger(__name__)

TIMEOUT = None


def cached_download(url, target, *, chunk_size=1024 * 1024, force=False, **kwargs):
    range_method = kwargs.pop("range_method", "auto")
    parts = kwargs.pop("parts", None)
    logger.debug(f"URL {url}")
    downloader = Downloader(
        url,
        chunk_size=chunk_size,
        timeout=TIMEOUT,
        verify=True,
        parts=parts,
        range_method=range_method,
        http_headers=None,
        fake_headers=None,
        resume_transfers=True,
        override_target_file=False,
        download_file_extension=".download",
    )

    if os.path.exists(target):
        if force:
            logger.info(f"Removing {target}")
            os.remove(target)
        else:
            logger.info(f"Skipping {target}")
            return target

    if not os.path.exists(target):
        downloader.download(target)

    return target
