"""
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
sudo apt install p7zip-full -y
7z x knesset_phonemes_v1.txt.7z

uv run src/prepare_data.py --input_file knesset_phonemes_v1.txt --output_file data3.csv --limit 10000
"""

import re
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="knesset_phonemes_v1.txt")
parser.add_argument('--output_file', type=str, default="data.csv")
parser.add_argument('--limit', type=int, default=-1)
args = parser.parse_args()

TASK = (
    "Given the following Hebrew sentence, convert it to IPA phonemes.\n\n"
    "Input Format: A Hebrew sentence.\n"
    "Output Format: A string of IPA phonemes."
)

# Regex to strip Hebrew diacritics + prefix char
PREFIX_CHAR = "|"
HEBREW_PATTERN = re.compile(rf"[\u0590-\u05C7{PREFIX_CHAR}]")

def remove_diacritics(text: str) -> str:
    return HEBREW_PATTERN.sub("", text)

def main():
    total_lines = sum(1 for _ in open(args.input_file, encoding="utf-8"))
    rows = []
    with Path(args.input_file).open("r", encoding="utf-8") as file:
        for count, line in tqdm(enumerate(file, start=1), total=total_lines):
            if args.limit != -1 and count > args.limit:
                break
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue  # skip malformed lines

            graphemes, phonemes = parts
            clean_graphemes = remove_diacritics(graphemes)

            rows.append({
                "task": TASK,
                "input": clean_graphemes,
                "expected_output": phonemes,
            })

    df = pd.DataFrame(rows, columns=["task", "input", "expected_output"])
    df.to_csv(Path(args.output_file), index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {Path(args.output_file)}")

if __name__ == "__main__":
    main()
