# Deduplicate System Prompt Plan

## Problem

The Hebrew-to-IPA system prompt is currently duplicated across **9 different files**:

1. `src/data.py` - `SYSTEM_PROMPT` constant
2. `src/prepare_data.py` - `TASK` constant
3. `src/eval.py` - `TASK` constant
4. `src/infer.py` - `TASK` constant
5. `src/export_gguf.py` - `TASK` constant
6. `onnx-infer/export.py` - `system_message` variable
7. `onnx-infer/eval.py` - `TASK` constant
8. `ollama-infer/main.py` - `system_message` variable
9. `gguf-infer/main.py` - `system_message` variable

**Issues with current approach:**
- Maintenance nightmare: updating prompt requires changing 9 files
- Risk of inconsistency: easy to miss a file during updates
- Violates DRY (Don't Repeat Yourself) principle
- Hard to track which files are in sync

---

## Solution: Single Source of Truth

Create a central configuration module that all other modules import from.

### Approach 1: Central Config Module (Recommended)

**Create**: `src/config.py`

```python
"""Central configuration for gemma3-g2p project."""

SYSTEM_PROMPT = """You will receive Hebrew text. Convert it into IPA-like phonemes using ONLY the following symbols:

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

Now wait for the text."""

# Model configuration
DEFAULT_MODEL = "unsloth/gemma-3-270m-it"
MAX_SEQ_LENGTH = 2048

# Generation parameters (for inference)
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
```

**Benefits:**
- Single location to update
- Can also centralize other configs (model name, generation params)
- Easy to version control
- Clear documentation

---

## Implementation Steps

### Step 1: Create Central Config Module

Create `src/config.py` with:
- `SYSTEM_PROMPT` - The Hebrew-to-IPA prompt
- Optional: other shared constants (model params, generation settings)

### Step 2: Update src/ Files

Update these files to import from config:

**`src/data.py`:**
```python
from config import SYSTEM_PROMPT

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},  # Use imported constant
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]}
        ]
    }
```

**`src/prepare_data.py`:**
```python
from config import SYSTEM_PROMPT

# Remove local TASK definition, use SYSTEM_PROMPT instead
rows.append({
    "task": SYSTEM_PROMPT,
    "input": clean_graphemes,
    "expected_output": phonemes,
})
```

**`src/eval.py`:**
```python
from config import SYSTEM_PROMPT

# Replace TASK with SYSTEM_PROMPT
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_message},
]
```

**`src/infer.py`:**
```python
from config import SYSTEM_PROMPT

# Replace TASK with SYSTEM_PROMPT
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_message},
]
```

**`src/export_gguf.py`:**
```python
from config import SYSTEM_PROMPT

convo = [
    {'role': 'system','content': SYSTEM_PROMPT},
    {'role': 'user','content': 'שלום עולם! מה קורה?'},
]
```

### Step 3: Handle Sub-packages (onnx-infer, ollama-infer, gguf-infer)

These are workspace members with separate dependencies. Two options:

#### Option A: Copy config.py to each workspace (Simple)
- Copy `src/config.py` to each workspace directory
- Each workspace has its own copy
- Still better than embedding in each file
- Trade-off: 3 copies instead of 1, but isolated

#### Option B: Share config via parent package (Better)
Make the workspaces import from parent:

**`onnx-infer/export.py`:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import SYSTEM_PROMPT

system_message = SYSTEM_PROMPT
```

**`ollama-infer/main.py`:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import SYSTEM_PROMPT

system_message = SYSTEM_PROMPT
```

**`gguf-infer/main.py`:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import SYSTEM_PROMPT

system_message = SYSTEM_PROMPT
```

---

## Recommended Approach

**Use Option A (Simple) for now:**

1. Create `src/config.py` with `SYSTEM_PROMPT`
2. Update all `src/*.py` files to import from `config`
3. Create `onnx-infer/config.py`, `ollama-infer/config.py`, `gguf-infer/config.py`
4. Copy the same content to each config.py
5. Update inference scripts to import from their local config

**Future improvement (Option B):**
- Later, refactor to share single config across workspaces
- Requires understanding workspace import structure better

---

## Files to Modify

### New Files:
- `src/config.py` - Create
- `onnx-infer/config.py` - Create (copy of src/config.py)
- `ollama-infer/config.py` - Create (copy of src/config.py)
- `gguf-infer/config.py` - Create (copy of src/config.py)

### Files to Update (import from config):
1. `src/data.py` - Replace `SYSTEM_PROMPT` with import
2. `src/prepare_data.py` - Replace `TASK` with `SYSTEM_PROMPT` import
3. `src/eval.py` - Replace `TASK` with `SYSTEM_PROMPT` import
4. `src/infer.py` - Replace `TASK` with `SYSTEM_PROMPT` import
5. `src/export_gguf.py` - Replace `TASK` with `SYSTEM_PROMPT` import
6. `onnx-infer/export.py` - Replace `system_message` with import
7. `onnx-infer/eval.py` - Replace `TASK` with import
8. `ollama-infer/main.py` - Replace `system_message` with import
9. `gguf-infer/main.py` - Replace `system_message` with import

---

## Verification Steps

After implementation:

1. **Grep for duplicates**: Ensure no hardcoded prompts remain
   ```bash
   grep -r "You will receive Hebrew text" --include="*.py" .
   ```
   Should only find matches in the 4 config.py files

2. **Test imports**: Run each script to ensure imports work
   ```bash
   python -c "from src.config import SYSTEM_PROMPT; print(SYSTEM_PROMPT)"
   ```

3. **Run training**: Verify training still works with new import
   ```bash
   uv run src/train.py --max_steps 1
   ```

4. **Run inference**: Test each inference script
   ```bash
   uv run src/infer.py --model_path ./outputs/checkpoint-10000
   ```

---

## Benefits After Implementation

✅ **Single source of truth** - Update prompt in 4 places (one per workspace) instead of 9
✅ **Consistency guaranteed** - Impossible to have mismatched prompts within a workspace
✅ **Easy maintenance** - Change config.py instead of hunting through files
✅ **Better organization** - Configs separate from logic
✅ **Extensible** - Can add more shared configs (model params, generation settings)

---

## Estimated Effort

- Create config files: 10 minutes
- Update imports in 9 files: 20 minutes
- Testing: 10 minutes

**Total**: ~40 minutes

---

## Future Enhancements

Once this is done, can extend config.py to include:

```python
# Model settings
MODEL_CONFIG = {
    "name": "unsloth/gemma-3-270m-it",
    "max_seq_length": 2048,
    "load_in_4bit": False,
    "load_in_8bit": False,
}

# LoRA settings
LORA_CONFIG = {
    "r": 128,
    "lora_alpha": 128,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
}

# Training settings
DEFAULT_TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 0.001,
    "max_steps": 10_000,
}
```

This would centralize all configuration, making the codebase even more maintainable.
