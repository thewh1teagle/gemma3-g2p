# Training-time Evaluation

## Problem

Currently, evaluation is manual and happens **after** training completes:
- Run training to completion or specific checkpoint
- Manually run `eval.py` with checkpoint path
- Review WER/CER metrics from JSON report

**Issues**:
- No insight into model quality during training
- Can't detect overfitting or plateaus early
- Miss opportunity to stop training when metrics stop improving
- No automatic logging to WandB

**Goal**: Integrate automatic evaluation during training every N steps, with WER/CER metrics logged to console and WandB.

---

## Current Evaluation Approach

From `src/eval.py`:

```python
# Loads evaluation TSV (sentence\tphonemes)
df = pd.read_csv(eval_file, sep='\t', header=0)

# For each sample:
#   1. Format as chat message
#   2. Run model.generate()
#   3. Calculate WER and CER using jiwer
#   4. Save to JSON report

# Metrics:
- mean_wer (Word Error Rate)
- mean_cer (Character Error Rate)
```

---

## Solution Overview

Use HuggingFace Trainer's built-in evaluation capabilities:

1. **Load eval dataset** from TSV file (separate from training data)
2. **Create custom `compute_metrics` function** that:
   - Runs model generation on eval samples
   - Calculates WER/CER using jiwer
   - Returns metrics dict for logging
3. **Configure SFTTrainer** with:
   - `eval_dataset` parameter
   - `eval_steps` in SFTConfig (evaluate every N steps)
   - `evaluation_strategy="steps"`
4. **Automatic logging** to console and WandB

---

## Implementation Plan

### Step 1: Create Eval Dataset Loader

**File**: `src/data.py` (add new function)

```python
def prepare_eval_dataset_from_tsv(file_path, max_samples=None):
    """
    Load evaluation dataset from TSV file.

    Format: sentence\tphonemes (with header row)

    Returns dataset with columns:
    - input: Hebrew text
    - expected_output: IPA phonemes
    """
    from datasets import Dataset
    import pandas as pd

    # Read TSV with header
    df = pd.read_csv(file_path, sep='\t', header=0, usecols=[0, 1])
    df.columns = ['input', 'expected_output']

    # Limit samples if specified (for faster eval during training)
    if max_samples:
        df = df.head(max_samples)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    return dataset
```

**Notes**:
- Don't apply chat template here (will do in compute_metrics)
- `max_samples` allows using subset for faster eval (e.g., 100 samples instead of full dataset)
- Returns raw dataset with input/expected_output

### Step 2: Create Custom Metric Computation

**File**: `src/metrics.py` (new file)

```python
"""Custom metrics for G2P evaluation during training."""
import jiwer
from typing import Dict, List
import numpy as np
from config import SYSTEM_PROMPT


def create_compute_metrics_fn(tokenizer, model):
    """
    Create a compute_metrics function for Trainer evaluation.

    Args:
        tokenizer: Model tokenizer with chat template
        model: The model being trained

    Returns:
        compute_metrics function that calculates WER/CER
    """

    def compute_metrics(eval_pred):
        """
        Compute WER and CER metrics on evaluation set.

        Args:
            eval_pred: EvalPrediction object with:
                - predictions: model predictions (logits or token IDs)
                - label_ids: ground truth labels
                - inputs: input token IDs

        Returns:
            Dict with mean_wer and mean_cer
        """
        # Get predictions and labels
        # Note: eval_pred structure depends on Trainer setup
        # For generation tasks, we need to run model.generate() manually

        # This is a callback-based approach since Trainer doesn't
        # naturally support generation during eval

        # We'll use a different approach - see Step 3
        pass

    return compute_metrics
```

**Issue**: HuggingFace Trainer's default `compute_metrics` works with logits, not generated text. For WER/CER, we need actual generated sequences.

**Solution**: Use a **custom callback** instead (see Step 3).

### Step 3: Create Custom Evaluation Callback

**File**: `src/metrics.py`

```python
"""Custom metrics for G2P evaluation during training."""
import jiwer
from transformers import TrainerCallback
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from config import SYSTEM_PROMPT


class G2PEvalCallback(TrainerCallback):
    """
    Custom callback to evaluate G2P model with WER/CER during training.

    Runs model.generate() on eval dataset and calculates metrics.
    """

    def __init__(self, eval_dataset: Dataset, tokenizer, eval_steps: int = 500):
        """
        Args:
            eval_dataset: Dataset with 'input' and 'expected_output' columns
            tokenizer: Tokenizer with chat template applied
            eval_steps: Evaluate every N steps
        """
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.last_eval_step = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""

        # Check if it's time to evaluate
        if state.global_step % self.eval_steps == 0 and state.global_step != self.last_eval_step:
            self.last_eval_step = state.global_step

            print(f"\n{'='*60}")
            print(f"Running G2P evaluation at step {state.global_step}...")
            print(f"{'='*60}")

            # Run evaluation
            metrics = self._evaluate(model)

            # Log to console
            print(f"Step {state.global_step} - WER: {metrics['eval_wer']:.4f}, CER: {metrics['eval_cer']:.4f}")

            # Log to WandB (if enabled)
            if state.is_world_process_zero and args.report_to == ['wandb']:
                import wandb
                wandb.log({
                    'eval_wer': metrics['eval_wer'],
                    'eval_cer': metrics['eval_cer'],
                    'step': state.global_step
                })

        return control

    def _evaluate(self, model) -> dict:
        """Run evaluation on the full eval dataset."""
        model.eval()  # Set to eval mode

        total_wer = 0.0
        total_cer = 0.0
        num_samples = len(self.eval_dataset)

        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = self.eval_dataset[idx]
            user_input = sample['input']
            expected_output = sample['expected_output']

            # Format as chat message
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ).removeprefix("<bos>")

            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                )

            # Decode prediction
            prediction = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Calculate metrics
            wer = jiwer.wer(expected_output, prediction)
            cer = jiwer.cer(expected_output, prediction)

            total_wer += wer
            total_cer += cer

        model.train()  # Set back to train mode

        return {
            'eval_wer': total_wer / num_samples,
            'eval_cer': total_cer / num_samples,
        }
```

**Key features**:
- Runs every `eval_steps` steps
- Uses `model.generate()` for actual inference
- Calculates WER/CER with jiwer
- Logs to console and WandB
- Sets model to eval mode during evaluation

### Step 4: Update train.py

**File**: `src/train.py`

**Changes**:

1. Add CLI arguments:
```python
parser.add_argument("--eval_file", type=str, default=None, help="Path to eval TSV file")
parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
parser.add_argument("--eval_samples", type=int, default=None, help="Max samples for eval (None = all)")
```

2. Load eval dataset:
```python
# After loading training dataset
eval_dataset = None
eval_callback = None

if args.eval_file:
    from data import prepare_eval_dataset_from_tsv
    from metrics import G2PEvalCallback

    print(f"Loading evaluation dataset from {args.eval_file}...")
    eval_dataset = prepare_eval_dataset_from_tsv(
        file_path=args.eval_file,
        max_samples=args.eval_samples
    )
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Create callback
    eval_callback = G2PEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        eval_steps=args.eval_steps
    )
```

3. Update SFTTrainer:
```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,  # Keep None - we use callback instead
    callbacks = [eval_callback] if eval_callback else None,  # Add callback
    args = SFTConfig(
        # ... existing config ...
    ),
)
```

**Note**: We don't use `eval_dataset` parameter because Trainer's default eval doesn't support generation. We use callback instead.

### Step 5: Add Dependencies

**File**: `pyproject.toml`

Ensure `jiwer` is listed:
```toml
dependencies = [
    # ... existing deps ...
    "jiwer>=3.0.0",  # For WER/CER calculation
]
```

Then run:
```bash
uv sync
```

---

## Usage

### Basic Training with Evaluation

```bash
# Download eval dataset
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv -O eval_data.tsv

# Train with evaluation every 500 steps
uv run src/train.py \
    --data_file data.tsv \
    --eval_file eval_data.tsv \
    --eval_steps 500 \
    --report_to wandb
```

### Fast Evaluation (Subset)

For faster training iterations, evaluate on subset:

```bash
# Evaluate on only 100 samples (instead of full dataset)
uv run src/train.py \
    --data_file data.tsv \
    --eval_file eval_data.tsv \
    --eval_steps 500 \
    --eval_samples 100 \
    --report_to wandb
```

### Training Without Evaluation

Evaluation is optional - just omit `--eval_file`:

```bash
# No evaluation (same as current behavior)
uv run src/train.py \
    --data_file data.tsv \
    --report_to wandb
```

---

## Output Examples

### Console Output

```
Step 500/10000 - Loss: 0.234
==============================================================
Running G2P evaluation at step 500...
==============================================================
Evaluating: 100%|██████████| 200/200 [00:45<00:00,  4.42it/s]
Step 500 - WER: 0.1234, CER: 0.0567
==============================================================

Step 1000/10000 - Loss: 0.198
==============================================================
Running G2P evaluation at step 1000...
==============================================================
Evaluating: 100%|██████████| 200/200 [00:43<00:00,  4.65it/s]
Step 1000 - WER: 0.0987, CER: 0.0432
==============================================================
```

### WandB Dashboard

Charts will show:
- `eval_wer` over training steps
- `eval_cer` over training steps
- Compare against `train/loss` to detect overfitting

---

## Alternative Approach: Use Trainer's Built-in Eval

**Pros of Callback Approach (Recommended)**:
- ✅ Full control over generation parameters
- ✅ Easy to log to WandB
- ✅ Can evaluate on subset for speed
- ✅ Clear console output

**Alternative**: Use Trainer's `evaluation_strategy` + `compute_metrics`
- Requires precompute_ref_wrapper or custom Trainer subclass
- More complex to set up for generation tasks
- Less flexible for G2P use case

**Recommendation**: Use callback approach (simpler, more flexible).

---

## Performance Considerations

### Evaluation Time

Evaluation adds overhead to training:

| Eval Samples | Time per Eval | Recommended eval_steps |
|--------------|---------------|------------------------|
| 100 samples  | ~20-30 sec    | 250-500 steps          |
| 500 samples  | ~2-3 min      | 500-1000 steps         |
| 1000 samples | ~5-6 min      | 1000-2000 steps        |

**Recommendation**: Use 100-200 samples with `eval_steps=500` for good balance.

### GPU Memory

Evaluation runs `model.generate()` which uses more VRAM than training:
- Uses same model (no extra VRAM)
- Generation is done with `torch.no_grad()` (saves memory)
- Processes one sample at a time (minimal overhead)

**Should be fine** even on smaller GPUs.

---

## Testing the Implementation

### 1. Test Eval Dataset Loading

```python
# Test loading
from src.data import prepare_eval_dataset_from_tsv

dataset = prepare_eval_dataset_from_tsv("eval_data.tsv", max_samples=10)
print(f"Loaded {len(dataset)} samples")
print(dataset[0])
# Output: {'input': 'שלום', 'expected_output': 'ʃalˈom'}
```

### 2. Test Metric Calculation

```python
import jiwer

reference = "ʃalˈom"
prediction = "ʃalom"  # Missing stress mark

wer = jiwer.wer(reference, prediction)
cer = jiwer.cer(reference, prediction)

print(f"WER: {wer:.4f}, CER: {cer:.4f}")
```

### 3. Test Training with Eval

```bash
# Create mini datasets for testing
echo -e "שלום\tʃalˈom\nמה\tmˈa" > test_train.tsv
echo -e "sentence\tphonemes\nשלום\tʃalˈom\nמה\tmˈa" > test_eval.tsv

# Run short training with eval
uv run src/train.py \
    --data_file test_train.tsv \
    --eval_file test_eval.tsv \
    --eval_steps 5 \
    --max_steps 20 \
    --save_steps 10
```

Expected: Evaluation runs at steps 5, 10, 15, 20 with WER/CER output.

---

## Benefits

✅ **Early stopping**: See when model stops improving
✅ **Overfitting detection**: Compare eval metrics vs train loss
✅ **Hyperparameter tuning**: Compare WER/CER across different runs
✅ **WandB tracking**: Automatic logging of metrics over time
✅ **Checkpoint selection**: Know which checkpoint has best WER/CER
✅ **Faster iteration**: No manual eval.py runs needed

---

## Future Enhancements

### 1. Save Best Checkpoint Based on WER

Add to callback:
```python
class G2PEvalCallback(TrainerCallback):
    def __init__(self, ...):
        self.best_wer = float('inf')
        self.best_checkpoint = None

    def _evaluate(self, model, state):
        metrics = # ... calculate WER/CER

        # Save best checkpoint
        if metrics['eval_wer'] < self.best_wer:
            self.best_wer = metrics['eval_wer']
            # Save checkpoint
            model.save_pretrained(f"outputs/best_checkpoint_step_{state.global_step}")
```

### 2. Early Stopping

Stop training if WER doesn't improve for N evaluations:
```python
class G2PEvalCallback(TrainerCallback):
    def __init__(self, ..., patience=5):
        self.patience = patience
        self.patience_counter = 0
        self.best_wer = float('inf')

    def on_step_end(self, args, state, control, **kwargs):
        metrics = self._evaluate(...)

        if metrics['eval_wer'] < self.best_wer:
            self.best_wer = metrics['eval_wer']
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            control.should_training_stop = True
```

### 3. Per-Sample Error Analysis

Save worst predictions for debugging:
```python
# In _evaluate()
errors = []
for sample in eval_dataset:
    # ... generate prediction ...
    if wer > 0.5:  # High error
        errors.append({
            'input': user_input,
            'expected': expected_output,
            'predicted': prediction,
            'wer': wer
        })

# Save to file
with open(f'outputs/errors_step_{state.global_step}.json', 'w') as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)
```

---

## Summary

**Implementation Steps**:
1. Create `src/metrics.py` with `G2PEvalCallback`
2. Add `prepare_eval_dataset_from_tsv()` to `src/data.py`
3. Update `src/train.py` with CLI args and callback integration
4. Add `jiwer` dependency to `pyproject.toml`

**Usage**:
```bash
uv run src/train.py \
    --data_file data.tsv \
    --eval_file eval_data.tsv \
    --eval_steps 500 \
    --eval_samples 100 \
    --report_to wandb
```

**Benefits**:
- Automatic WER/CER tracking during training
- WandB integration for metric visualization
- Flexible eval frequency and dataset size
- No manual eval.py runs needed
