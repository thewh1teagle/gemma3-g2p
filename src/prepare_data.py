"""
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
sudo apt install p7zip-full -y
7z x knesset_phonemes_v1.txt.7z

uv run src/prepare_data.py
"""

import re
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ---------------- #
INPUT_FILE = Path("knesset_phonemes_v1.txt")
OUTPUT_FILE = Path("knesset_phonemes_v1.csv")
LIMIT = 100_000
# ---------------------------------------- #

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
    rows = []
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for count, line in enumerate(file, start=1):
            if count > LIMIT:
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
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
