"""
Prepare raw phoneme data:
- Remove diacritics from Hebrew text
- Filter malformed rows
- Validate format

Input: Raw TSV with diacritics (graphemes\tphonemes)
Output: Clean TSV (input\toutput) - NO system prompt column

Usage:
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
sudo apt install p7zip-full -y
7z x knesset_phonemes_v1.txt.7z

uv run src/prepare_dataset.py --input_file knesset_phonemes_v1.txt --output_file data.tsv --limit 10000
"""

import re
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="knesset_phonemes_v1.txt")
parser.add_argument('--output_file', type=str, default="data.tsv")
parser.add_argument('--limit', type=int, default=-1)
args = parser.parse_args()

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

            # Just write input\toutput - NO system prompt
            rows.append({
                "input": clean_graphemes,
                "output": phonemes,
            })

    # Save as TSV (no header, just data)
    df = pd.DataFrame(rows, columns=["input", "output"])
    df.to_csv(
        Path(args.output_file),
        sep="\t",
        index=False,
        header=False,  # No header for TSV
        encoding="utf-8"
    )
    print(f"Saved {len(df)} rows to {Path(args.output_file)}")

if __name__ == "__main__":
    main()
