"""Single read stream from disk; bounded shared-RAM queue; process workers.

Read every RAW once into a shared-memory slot in sorted filename/folder order.
Workers only touch RAM, not the source HDD. The caller runs this in a background
server thread, so the main process can keep feeding the queue while workers
analyse in separate Python processes (spawn on macOS).
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import get_context, shared_memory
from pathlib import Path
import os

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}


def enumerate_raw_files(root: str | Path) -> list[Path]:
    """Directory entries only, no image I/O during enumeration."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("Choose an existing folder containing RAW photographs.")
    paths = []
    for folder, dirs, files in os.walk(base):
        dirs.sort(key=str.casefold)
        for name in sorted(files, key=str.casefold):
            if Path(name).suffix.casefold() in RAW_EXTENSIONS:
                paths.append(Path(folder) / name)
    return paths


def _worker(shared_name: str, length: int):
    from .features import extract_raw_features
    shm = shared_memory.SharedMemory(name=shared_name, create=False)
    view = shm.buf[:length]
    try:
        data = bytes(view)
    finally:
        view.release()
        shm.close()
    # Decode after releasing shared memory, so a decode exception cannot retain
    # a memoryview in its traceback and mask the actual failure with BufferError.
    return extract_raw_features(data).tolist()


def _load_into_ram(path: Path):
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("File is empty")
    shm = shared_memory.SharedMemory(create=True, size=size)
    view = None
    try:
        view = shm.buf
        with path.open("rb", buffering=1024 * 1024) as source:
            offset = 0
            while offset < size:
                # readinto() avoids an extra Python bytes object for each RAW.
                count = source.readinto(view[offset:offset + min(1024 * 1024, size - offset)])
                if not count:
                    raise OSError(f"Unexpected end of file: {path}")
                offset += count
        del view
        return shm, size
    except BaseException:
        if view is not None:
            del view
        shm.close()
        shm.unlink()
        raise


def analyse_raws(paths: list[Path], progress=None, workers=3, ram_mb=384, readers=1):
    """Read sorted RAWs with bounded reader threads; decode in a process pool.

    Shared-memory allocations and pending reads count against the byte budget.
    The returned list is aligned with paths, including a message for failures.
    """
    workers = max(1, min(int(workers), 12))
    readers = max(1, min(int(readers), 4))
    budget = max(64, int(ram_mb)) * 1024 * 1024
    results = [None] * len(paths)
    read_jobs, decode_jobs = {}, {}
    reserved = 0
    finished = 0
    index = 0

    def report(pos, result):
        nonlocal finished
        results[pos] = result
        finished += 1
        if progress:
            progress(finished, len(paths), paths[pos].name)

    with ThreadPoolExecutor(max_workers=readers) as reader_pool, ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as decode_pool:
        try:
            while index < len(paths) or read_jobs or decode_jobs:
                while index < len(paths) and len(read_jobs) < readers and len(read_jobs)+len(decode_jobs) < workers * 2:
                    pos = index
                    try:
                        size = paths[pos].stat().st_size
                        if size <= 0: raise ValueError("File is empty")
                    except Exception as exc:
                        index += 1
                        report(pos, f"{type(exc).__name__}: {exc}")
                        continue
                    if (read_jobs or decode_jobs) and reserved + size > budget:
                        break
                    index += 1
                    reserved += size
                    read_jobs[reader_pool.submit(_load_into_ram, paths[pos])] = (pos, size)
                if not read_jobs and not decode_jobs:
                    continue
                ready, _ = wait(set(read_jobs) | set(decode_jobs), return_when=FIRST_COMPLETED)
                for future in ready:
                    if future in read_jobs:
                        pos, size = read_jobs.pop(future)
                        try:
                            shm, actual_size = future.result()
                            if actual_size != size:
                                shm.close(); shm.unlink()
                                raise ValueError("File changed size during reading")
                            try:
                                job = decode_pool.submit(_worker, shm.name, actual_size)
                            except BaseException:
                                shm.close(); shm.unlink(); raise
                            decode_jobs[job] = (pos, shm, size)
                        except Exception as exc:
                            reserved -= size
                            report(pos, f"{type(exc).__name__}: {exc}")
                    else:
                        pos, shm, size = decode_jobs.pop(future)
                        try:
                            try: result = future.result()
                            except Exception as exc: result = f"{type(exc).__name__}: {exc}"
                        finally:
                            reserved -= size
                            shm.close(); shm.unlink()
                        report(pos, result)
        finally:
            # A running read may still create shared memory; collect it before
            # cleanup rather than discarding its future and leaking its segment.
            for future, (pos, size) in list(read_jobs.items()):
                if not future.cancel():
                    try:
                        shm, _ = future.result()
                        shm.close(); shm.unlink()
                    except Exception: pass
            for future, (pos, shm, size) in list(decode_jobs.items()):
                future.cancel()
                shm.close(); shm.unlink()
    return results
