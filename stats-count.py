import os
import tqdm

def count_tokens_in_file(filepath: str) -> int:
    """Count whitespace-separated tokens in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return len([x for x in text.split() if x and x.strip() if len(x) >= 2])

def count_tokens_in_directory(directory: str):
    """Count total and per-file tokens in a directory."""
    total_tokens = 0
    tokens_per_file = {}

    for filename in tqdm.tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            token_count = count_tokens_in_file(filepath)
            tokens_per_file[filename] = token_count
            total_tokens += token_count


    return total_tokens, tokens_per_file

# Example usage
directory = "txt"
total, per_file = count_tokens_in_directory(directory)

print(f"Total tokens: {total}")
print(f"{total/len(per_file)} tokens / file")