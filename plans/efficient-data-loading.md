# Efficient Data Loading Plan

## Problem

Current data pipeline is inefficient:

### Current Flow:
1. **Source**: 1GB TSV file with 2 columns: `input\toutput` (graphemes → phonemes)
2. **Preprocessing**: `prepare_data.py` reads TSV, adds system prompt to EVERY row, writes CSV
3. **Result**: CSV with 3 columns: `task, input, expected_output`
4. **Training**: Loads CSV and uses the `task` column

### Issues:
1. **Massive duplication**: System prompt (300+ chars) duplicated for every single row
   - Original TSV: ~1GB
   - CSV with system prompt: Much larger (1GB + 300 bytes × num_rows)
   - Example: 1M rows = 300MB just for duplicate prompts!

2. **Unnecessary preprocessing**: Extra step to convert TSV → CSV

3. **Double storage**: System prompt exists in:
   - `src/config.py` (source of truth)
   - CSV file (duplicated millions of times)

4. **Slower loading**: Larger file = slower I/O

5. **Wasted disk space**: Storing redundant data

### Example of Waste:

Original TSV (2 columns):
```
שלום עולם	ʃalˈom ʔolˈam
מה קורה	mˈa koʁˈe
```

Current CSV (3 columns with duplicated system prompt):
```
task,input,expected_output
"You will receive Hebrew text. Convert it into IPA-like phonemes using ONLY the following symbols:

Vowels: a, e, i, o, u
Consonants: b, v, d, h, z, χ, t, j, k, l, m, n, s, f, p, ts, tʃ, w, ʔ, ɡ, ʁ, ʃ, ʒ, dʒ

Rules:
1. Every Hebrew word must include a stress mark.
2. Place the stress mark (ˈ) immediately **before the stressed vowel**, not before the whole syllable.
   Example: shalom → ʃalˈom
3. Keep punctuation exactly as in the input.
4. Output ONLY the phonemes (no explanations, no slashes).
5. Use ʔ for א / ע.
6. Don't add vowels or consonants that aren't written.

Examples:
שלום עולם → ʃalˈom ʔolˈam
מה קורה? → mˈa koʁˈe?
אתה יודע → ʔatˈa jodˈeʔa

Now wait for the text.",שלום עולם,ʃalˈom ʔolˈam
"You will receive Hebrew text. Convert it into IPA-like phonemes using ONLY the following symbols:

Vowels: a, e, i, o, u
Consonants: b, v, d, h, z, χ, t, j, k, l, m, n, s, f, p, ts, tʃ, w, ʔ, ɡ, ʁ, ʃ, ʒ, dʒ

Rules:
1. Every Hebrew word must include a stress mark.
2. Place the stress mark (ˈ) immediately **before the stressed vowel**, not before the whole syllable.
   Example: shalom → ʃalˈom
3. Keep punctuation exactly as in the input.
4. Output ONLY the phonemes (no explanations, no slashes).
5. Use ʔ for א / ע.
6. Don't add vowels or consonants that aren't written.

Examples:
שלום עולם → ʃalˈom ʔolˈam
מה קורה? → mˈa koʁˈe?
אתה יודע → ʔatˈa jodˈeʔa

Now wait for the text.",מה קורה,mˈa koʁˈe
```

**This is ABSURD!** The system prompt is 300+ characters, repeated for EVERY row.

---

## Solution: Load TSV Directly with On-the-Fly Prompt Injection

### New Flow:

1. **Source**: 1GB TSV file (unchanged)
2. **Training**: Load TSV directly, add system prompt on-the-fly during mapping
3. **Result**: No intermediate CSV needed!

### Implementation

#### Update `src/data.py`:

**Current code:**
```python
from config import SYSTEM_PROMPT

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]}
        ]
    }

def prepare_dataset_from_csv(tokenizer, file_path, split='train'):
    dataset = load_dataset("csv", data_files=file_path, split=split)
    dataset = dataset.map(convert_to_chatml)
    dataset = dataset.map(formatting_prompts_func, batched=True, fn_kwargs={"tokenizer": tokenizer})
    return dataset
```

**New code:**
```python
from config import SYSTEM_PROMPT

def convert_to_chatml(example):
    """Add system prompt on-the-fly - no need to store in dataset"""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
    }

def prepare_dataset_from_tsv(tokenizer, file_path, split='train'):
    """Load TSV directly without intermediate CSV preprocessing"""
    dataset = load_dataset(
        "csv",
        data_files=file_path,
        split=split,
        delimiter="\t",  # TSV uses tabs
        column_names=["input", "output"],  # Explicit column names
        header=None  # No header in raw TSV
    )
    dataset = dataset.map(convert_to_chatml)
    dataset = dataset.map(formatting_prompts_func, batched=True, fn_kwargs={"tokenizer": tokenizer})
    return dataset
```

#### Update `src/train.py`:

**Current:**
```python
parser.add_argument("--csv_file", type=str, default="data.csv")
...
dataset = prepare_dataset_from_csv(tokenizer, file_path=args.csv_file, split='train')
```

**New:**
```python
parser.add_argument("--data_file", type=str, default="data.tsv", help="Path to TSV data file")
...
dataset = prepare_dataset_from_tsv(tokenizer, file_path=args.data_file, split='train')
```

#### Update `src/eval.py`:

**Current:**
```python
df = pd.read_csv(args.input_file, sep='\t', header=None, names=['input', 'expected'])
```

Already using TSV format - just verify it works correctly.

---

## Benefits

### Before:
- 1GB TSV → preprocessing → 1.3GB+ CSV (with duplicated prompts) → load into training
- 2 steps, larger file

### After:
- 1GB TSV → load directly into training (prompt added on-the-fly)
- 1 step, smaller file

### Concrete Benefits:

1. **Disk space saved**: ~300MB+ (depending on dataset size)
   - No more storing system prompt millions of times

2. **Faster loading**: Smaller file = faster I/O

3. **Simpler pipeline**: No preprocessing step needed

4. **Single source of truth**: System prompt only in `config.py`

5. **Easier updates**: Change prompt in config.py, no need to regenerate CSV

6. **Memory efficient**: Prompt added during dataset mapping (lazy evaluation)

---

## What About `prepare_data.py`?

### Current Purpose:
Converts TSV → CSV by adding system prompt column

### New Purpose - Option 1: Data Cleaning Only

Rename to `preprocess_tsv.py` and change purpose to data cleaning only:

```python
"""
Preprocess raw phoneme data:
- Remove diacritics from Hebrew text
- Filter malformed rows
- Validate format

Input: Raw TSV with diacritics
Output: Clean TSV (input\toutput) - NO system prompt column
"""

import re
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="knesset_phonemes_v1.txt")
parser.add_argument('--output_file', type=str, default="data.tsv")  # TSV output
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
```

### New Purpose - Option 2: Remove Entirely

If the raw data is already clean (no diacritics, proper format), we don't need `prepare_data.py` at all. Just use the TSV directly.

**Recommended**: Keep it for data cleaning (Option 1), but remove system prompt logic.

---

## Files to Modify

### 1. `src/data.py`
- Remove `prepare_dataset_from_csv()`
- Rename/update to `prepare_dataset_from_tsv()`
- Change column reference: `example["expected_output"]` → `example["output"]`

### 2. `src/train.py`
- Change argument: `--csv_file` → `--data_file`
- Update default: `data.csv` → `data.tsv`
- Update function call: `prepare_dataset_from_csv()` → `prepare_dataset_from_tsv()`
- Update docstring examples

### 3. `src/prepare_data.py`
- **Option A**: Rename to `preprocess_tsv.py`
- Remove `SYSTEM_PROMPT` import and usage
- Change output to TSV format (no header, 2 columns only)
- Update to only clean data, not add system prompt

### 4. Documentation
- Update `train.py` docstring to mention TSV format
- Update examples in comments

---

## Testing Plan

1. **Create test TSV**:
   ```bash
   echo -e "שלום עולם\tʃalˈom ʔolˈam\nמה קורה\tmˈa koʁˈe" > test.tsv
   ```

2. **Test loading**:
   ```python
   from src.data import prepare_dataset_from_tsv
   from transformers import AutoTokenizer
   from unsloth.chat_templates import get_chat_template

   tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')
   tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
   dataset = prepare_dataset_from_tsv(tokenizer, 'test.tsv')

   print(dataset[0])
   # Should have conversations with SYSTEM_PROMPT from config
   ```

3. **Test training**:
   ```bash
   uv run src/train.py --data_file test.tsv --max_steps 1
   ```

4. **Verify system prompt**: Check that conversations have correct system prompt from `config.py`

5. **Test preprocessing** (if keeping):
   ```bash
   uv run src/preprocess_tsv.py --input_file raw_data.txt --output_file clean_data.tsv --limit 100
   ```

---

## Impact Analysis

### Disk Space Savings:

For 1M rows:
- System prompt size: ~300 bytes
- Duplication: 300 bytes × 1M rows = 300MB wasted
- **Savings: 300MB+**

For 10M rows:
- **Savings: 3GB+**

### Performance Impact:

- **I/O**: Faster (smaller file)
- **Parsing**: Same (HuggingFace datasets handles both efficiently)
- **Memory**: Same (lazy loading with map)

### Breaking Changes:

- CLI argument: `--csv_file` → `--data_file`
- Function name: `prepare_dataset_from_csv()` → `prepare_dataset_from_tsv()`
- Data format: Must use TSV instead of CSV

**Mitigation**: Simple find-replace in any scripts using the old arguments.

---

## Estimated Effort

- Update `data.py`: 15 minutes
- Update `train.py`: 10 minutes
- Update `prepare_data.py`: 15 minutes
- Testing: 20 minutes

**Total: ~1 hour**

---

## Implementation Steps

1. **Update `src/data.py`**:
   - Remove `prepare_dataset_from_csv` function
   - Add `prepare_dataset_from_tsv` function with TSV loading logic
   - Update `convert_to_chatml` to use `example["output"]` instead of `example["expected_output"]`

2. **Update `src/train.py`**:
   - Change `--csv_file` to `--data_file`
   - Change default from `data.csv` to `data.tsv`
   - Update function call to `prepare_dataset_from_tsv()`
   - Update docstring

3. **Update `src/prepare_data.py`**:
   - Rename to `preprocess_tsv.py`
   - Remove `SYSTEM_PROMPT` import
   - Remove `task` column from output
   - Change output to TSV format (tab-separated, no header)
   - Update docstring

4. **Test thoroughly**:
   - Test TSV loading
   - Test training with new format
   - Verify system prompt is correctly applied

5. **Clean up**:
   - Delete any old CSV files with duplicated prompts
   - Update any documentation

---

## Conclusion

Loading TSV directly is **significantly more efficient** than the current CSV preprocessing approach:

- ✅ Saves 300MB-3GB+ disk space (depending on dataset size)
- ✅ Faster I/O (smaller files)
- ✅ Simpler pipeline (no preprocessing step)
- ✅ Single source of truth (system prompt only in config.py)
- ✅ Easy to update (change config.py, no need to regenerate data)

**Recommendation**: Replace CSV approach entirely with direct TSV loading. No backward compatibility needed.
