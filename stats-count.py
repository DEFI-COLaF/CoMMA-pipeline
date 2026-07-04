import os
import json
from glob import glob
import tqdm
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

CACHE_PATH = "wordcount.json"
NUM_PROCESSES = 15

def load_cache(path: str) -> Dict[str, Dict[str, int]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(path: str, cache: Dict[str, Dict[str, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def count_words_if_modified(filepath: str, cached_data: Dict[str, Dict[str, int]]) -> Tuple[str, Dict[str, int] | None]:
    abs_path = os.path.basename(filepath)
    try:
        mod_time = os.path.getmtime(filepath)
    except FileNotFoundError:
        return abs_path, None  # File might have been deleted between scan and read

    if abs_path in cached_data and cached_data[abs_path].get("mtime") == mod_time:
        return abs_path, None  # No need to update

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        word_count = len(text.split())
        return abs_path, {"count": word_count, "mtime": mod_time}
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return abs_path, None

def main():
    cache = load_cache(CACHE_PATH)
    txt_files = glob("txt/**/*.txt", recursive=True)
    updated = False
    pbar = tqdm.tqdm(total=len(txt_files))
    total = 0
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(count_words_if_modified, path, cache) for path in txt_files]

        for future in as_completed(futures):
            path, result = future.result()
            if result is not None:
                cache[path] = result
                #print(f"Updated: {path}")
                updated = True
            #else:
                #print(f"Cached: {path}")
            pbar.update(1)
            total += cache[path]["count"]
            pbar.set_description(f"Tokens: {total:,}")

    if updated:
        save_cache(CACHE_PATH, cache)
        print("Cache updated.")
    else:
        print("No changes detected. Cache is up-to-date.")

    # Optional summary
    real_total = 0
    for path, data in cache.items():
        real_total += data["count"]
    print(f"Real total: {real_total:,} words")

if __name__ == "__main__":
    main()
