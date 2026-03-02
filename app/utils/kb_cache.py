import hashlib
import os
from datetime import datetime


def get_latest_kb_update_timestamp(log_dir: str) -> datetime | None:
    latest_ts = None
    for file_name in os.listdir(log_dir):
        if file_name.startswith("run_") and file_name.endswith(".log"):
            ts_str = file_name[4:-4].replace("/", ":")  # Remove pre/suffix, / -> :
            ts = datetime.fromisoformat(ts_str)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
    return latest_ts


def get_latest_cache(cache_dir: str) -> tuple[datetime | None, str | None]:
    os.makedirs(cache_dir, exist_ok=True)
    latest_ts = None
    latest_file = None
    for file_name in os.listdir(cache_dir):
        if file_name.startswith("app_") and file_name.endswith(".pkl"):
            ts_str = file_name[4:-4].replace("/", ":")  # Remove pre/suffix, / -> :
            ts = datetime.fromisoformat(ts_str)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_file = file_name
    latest_file_path = os.path.join(cache_dir, latest_file) if latest_file else None
    return latest_ts, latest_file_path


def get_cache_dir(kb_full_path: str, cache_dir: str):
    if cache_dir == ".":
        return kb_full_path
    else:
        hashed = hashlib.md5(kb_full_path.encode()).hexdigest()[:8]
        print("hash check: ", kb_full_path, hashed)
        return os.path.join(cache_dir, hashed)


def check_cache_status(log_dir: str, cache_dir: str = ".") -> tuple[str, str | None]:
    cache_dir = get_cache_dir(log_dir, cache_dir)
    cache_ts, cache_file_path = get_latest_cache(cache_dir)
    log_ts = get_latest_kb_update_timestamp(log_dir)
    if not cache_ts:
        cache_status = "none"
    elif log_ts and cache_ts > log_ts:
        cache_status = "up-to-date"
    else:  # no log or log outdated
        cache_status = "outdated"
    return cache_status, cache_file_path
