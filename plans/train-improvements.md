# Train.py Improvements Plan

## Overview
This plan outlines changes to align `train.py` with the original `notebook.py` while keeping practical improvements like CSV dataset loading and CLI arguments.

## Critical Issues to Fix

### 1. Missing Chat Template Setup [CRITICAL - BREAKS TRAINING]
**Location**: `src/train.py:89-91`

**Problem**: The `get_chat_template()` function is defined but never called. This causes `data.py:18` to fail when calling `tokenizer.apply_chat_template()`.

**Current code**:
```python
model, tokenizer = enable_fast_training()
model = add_lora_adapters(model, tokenizer)
dataset = prepare_dataset_from_csv(tokenizer, file_path=args.csv_file, split='train')
```

**Fix**: Add chat template setup between lines 89-90:
```python
model, tokenizer = enable_fast_training()
tokenizer = get_chat_template(tokenizer)  # ADD THIS LINE
model = add_lora_adapters(model, tokenizer)
dataset = prepare_dataset_from_csv(tokenizer, file_path=args.csv_file, split='train')
```

**Impact**: Without this, the tokenizer won't have the gemma3 chat template configured, causing either:
- Runtime errors when applying chat template
- Incorrect token formatting
- Training on wrong data format

**Priority**: CRITICAL - Must fix before any training runs

---

## Configuration Differences to Review

### 2. Weight Decay Discrepancy
**Location**: `src/train.py:109`

**Notebook value**: `0.001`
**Train.py value**: `0.01` (10x higher)

**Analysis**:
- Higher weight_decay = stronger L2 regularization
- May prevent overfitting on small datasets
- Could also reduce model capacity to learn

**Recommendation**:
- Add this as a CLI argument with default `0.001` (notebook value)
- Allow users to experiment with different values
- Document that 0.01 was tested in production

**Implementation**:
```python
parser.add_argument("--weight_decay", type=float, default=0.001)
```

Then use `weight_decay = args.weight_decay` in SFTConfig

**Priority**: MEDIUM - Affects training dynamics but not broken

---

## Optional Debugging Features

### 3. GPU Memory Statistics
**Location**: Currently removed from train.py

**Notebook implementation**: Lines 203-226

**Purpose**:
- Track memory usage before/after training
- Identify OOM issues early
- Optimize batch size and gradient accumulation

**Recommendation**: Add as optional feature with `--verbose` flag

**Implementation**:
```python
parser.add_argument("--verbose", action="store_true", help="Show GPU memory stats")

if args.verbose:
    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
```

**Priority**: LOW - Nice to have for debugging

---

### 4. Dataset Inspection
**Location**: Currently removed from train.py

**Notebook implementation**: Lines 139, 154, 196, 200

**Purpose**:
- Verify data preprocessing is correct
- Check chat template application
- Debug masking for response-only training

**Recommendation**: Add as optional feature with `--show_examples` flag

**Implementation**:
```python
parser.add_argument("--show_examples", action="store_true", help="Show dataset examples before training")

if args.show_examples:
    print("\n=== Dataset Example ===")
    print(dataset[100])
    print("\n=== Formatted Text ===")
    print(dataset[100]['text'])
    print("\n=== Tokenized Input ===")
    print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
    print("\n=== Masked Labels (only assistant responses) ===")
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x
                           for x in trainer.train_dataset[100]["labels"]])
          .replace(tokenizer.pad_token, " "))
```

**Priority**: LOW - Helpful for debugging but not essential

---

## Things to Keep (Don't Change)

### Keep CSV Dataset Loading
- `prepare_dataset_from_csv()` is more flexible than hardcoded ChessInstruct
- Allows custom datasets without code changes
- User specifically wants this

### Keep CLI Arguments
- More professional than hardcoded values
- Easier to run experiments
- Better for production use

### Keep Checkpoint Saving
- `save_steps=500` and `save_total_limit=5` not in notebook
- Critical for long training runs
- Allows resuming from failures

### Keep Modular Structure
- Functions like `enable_fast_training()`, `add_lora_adapters()` make code cleaner
- Easier to test and modify
- Better software engineering practice

---

## Implementation Priority

1. **CRITICAL (Must Do)**: Fix missing chat template setup
2. **MEDIUM (Should Do)**: Make weight_decay configurable via CLI
3. **LOW (Nice to Have)**: Add `--verbose` flag for GPU stats
4. **LOW (Nice to Have)**: Add `--show_examples` flag for dataset inspection

---

## Verification Steps

After implementing fixes:

1. **Verify chat template is applied**:
   ```python
   # Add assertion in main()
   assert hasattr(tokenizer, 'chat_template'), "Chat template not configured!"
   ```

2. **Test with small dataset**:
   ```bash
   uv run src/train.py --csv_file data.csv --max_steps 10 --show_examples --verbose
   ```

3. **Check dataset formatting**:
   - Verify conversations have proper format
   - Confirm masking works (only assistant text in labels)
   - Validate chat template produces expected tokens

4. **Compare training metrics**:
   - Run with `weight_decay=0.001` (original)
   - Run with `weight_decay=0.01` (current)
   - Compare validation loss if available

---

## Files to Modify

- `src/train.py` - Add chat template call, make weight_decay configurable, add optional debug features
- `README.md` (optional) - Document new CLI flags

## Estimated Changes

- Critical fix: 1 line addition
- Weight decay: ~3 lines (arg + usage)
- Verbose mode: ~15 lines (optional)
- Show examples: ~15 lines (optional)

**Total**: 1-35 lines depending on which features are implemented
