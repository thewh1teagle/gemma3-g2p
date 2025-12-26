# DGX Spark Native Training (Without Docker)

## Problem

Training on DGX Spark requires special Blackwell-compatible packages, but we want to avoid Docker overhead and use native Python with uv.

**Challenges**:
1. Need specific package versions for Blackwell GPU support (built from source)
2. Can't modify `pyproject.toml` (would affect other environments)
3. `uv run` syncs from `pyproject.toml` by default (would override our Blackwell packages)
4. Need Triton and xformers built from source with Blackwell support (CUDA 13.0, compute 12.1)

**Solution**:
- Use `uv sync` to install all dependencies from `pyproject.toml`
- Override Blackwell-specific packages with `uv pip install`
- Set `UV_NO_SYNC=1` environment variable (uv 0.4.18+) to prevent future syncs

---

## Solution Overview

Use `uv sync` to install all project dependencies, then override Blackwell-specific packages with `uv pip install`, and set `UV_NO_SYNC=1` to prevent future syncs.

### Strategy

```
1. Install all dependencies with uv sync
2. Override Blackwell packages with uv pip install (Triton, xformers, unsloth, etc.)
3. Set UV_NO_SYNC=1 environment variable (uv 0.4.18+) to prevent re-syncing
4. Run training with uv run (no --no-sync needed thanks to UV_NO_SYNC)
```

---

## Detailed Implementation Plan

### Step 1: Install Prerequisites

DGX Spark already has CUDA 13.0 installed. We need Python 3.12+ and uv:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify versions
python3 --version  # Should be 3.12+
uv --version
nvcc --version     # Should show CUDA 13.0
```

### Step 2: Install All Project Dependencies

First, sync all dependencies from `pyproject.toml`:

```bash
cd /home/yakov/Documents/audio/gemma3-g2p

# Install all project dependencies (creates .venv and syncs from pyproject.toml)
uv sync
```

**Note**: This installs the standard PyPI versions of all packages, including PyTorch, transformers, etc.

### Step 3: Override with Blackwell-Compatible Packages

Now override the packages that need Blackwell GPU support by building from source:

```bash
# 1. Install build dependencies
uv pip install wheel setuptools

# 2. Build and install Triton from source (Blackwell support)
git clone https://github.com/triton-lang/triton.git /tmp/triton
cd /tmp/triton
git checkout c5d671f91d90f40900027382f98b17a3e04045f6
uv pip install -r python/requirements.txt
uv pip install .
cd -

# 3. Build and install xformers from source (Blackwell support)
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive /tmp/xformers
cd /tmp/xformers
export TORCH_CUDA_ARCH_LIST="12.1"  # Blackwell compute capability
uv run python setup.py install
cd -

# 4. Override unsloth and core dependencies with specific versions
uv pip install --upgrade \
    unsloth \
    unsloth_zoo \
    bitsandbytes==0.48.0 \
    transformers==4.56.2 \
    trl==0.22.2
```

**Estimated time**: 30-60 minutes (building from source)

**Key advantage**: We get all project dependencies from `pyproject.toml` automatically, then just override the Blackwell-specific ones.

### Step 4: Set UV_NO_SYNC Environment Variable

Prevent uv from re-syncing with `pyproject.toml` in future runs:

```bash
# Set UV_NO_SYNC=1 to prevent future syncs (uv 0.4.18+)
export UV_NO_SYNC=1

# Make it permanent by adding to ~/.bashrc
echo 'export UV_NO_SYNC=1' >> ~/.bashrc
source ~/.bashrc
```

**Important**: This tells uv to never sync the environment with `pyproject.toml` again, preserving our Blackwell-compatible packages.

### Step 5: Create Setup Script

**File**: `scripts/setup-dgx-venv.sh`

```bash
#!/bin/bash
# One-time setup for DGX Spark native training environment

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Setting up DGX Spark native training environment..."

# Install all project dependencies
echo "Installing project dependencies from pyproject.toml..."
uv sync

# Install build dependencies
echo "Installing build dependencies..."
uv pip install wheel setuptools

# Build Triton
echo "Building Triton from source (this will take ~15 minutes)..."
rm -rf /tmp/triton
git clone https://github.com/triton-lang/triton.git /tmp/triton
cd /tmp/triton
git checkout c5d671f91d90f40900027382f98b17a3e04045f6
uv pip install -r python/requirements.txt
uv pip install .
cd "$PROJECT_DIR"

# Build xformers
echo "Building xformers from source (this will take ~20 minutes)..."
rm -rf /tmp/xformers
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive /tmp/xformers
cd /tmp/xformers
export TORCH_CUDA_ARCH_LIST="12.1"
uv run python setup.py install
cd "$PROJECT_DIR"

# Override unsloth and core dependencies
echo "Installing Blackwell-compatible unsloth and dependencies..."
uv pip install --upgrade \
    unsloth \
    unsloth_zoo \
    bitsandbytes==0.48.0 \
    transformers==4.56.2 \
    trl==0.22.2

# Set UV_NO_SYNC environment variable to prevent future syncs
echo "Setting UV_NO_SYNC=1 to prevent future syncs..."
if ! grep -q "export UV_NO_SYNC=1" ~/.bashrc; then
    echo 'export UV_NO_SYNC=1' >> ~/.bashrc
    echo "Added UV_NO_SYNC=1 to ~/.bashrc"
fi
export UV_NO_SYNC=1

echo ""
echo "✅ Setup complete!"
echo ""
echo "IMPORTANT: UV_NO_SYNC=1 has been set in your environment."
echo "This prevents uv from re-syncing with pyproject.toml and overwriting Blackwell packages."
echo ""
echo "To run training:"
echo "  uv run src/train.py --data_file data.tsv --report_to wandb"
echo ""
echo "Or use direct Python:"
echo "  .venv/bin/python src/train.py --data_file data.tsv --report_to wandb"
```

**Usage**:
```bash
chmod +x scripts/setup-dgx-venv.sh
./scripts/setup-dgx-venv.sh
```

---

## Running Training

### Option 1: Using `uv run` (Recommended)

With `UV_NO_SYNC=1` set, you can use `uv run` directly without the `--no-sync` flag:

```bash
export WANDB_API_KEY=your_key

uv run src/train.py \
    --data_file data.tsv \
    --report_to wandb \
    --max_steps 10000
```

**Pros**:
- Clean command (no `--no-sync` needed)
- `UV_NO_SYNC=1` prevents overwriting our Blackwell packages
- Still uses uv tooling and project isolation

**Cons**:
- Relies on environment variable being set

### Option 2: Direct Python (Simpler)

Use the venv's Python directly:

```bash
export WANDB_API_KEY=your_key

.venv/bin/python src/train.py \
    --data_file data.tsv \
    --report_to wandb \
    --max_steps 10000
```

**Pros**:
- No risk of uv syncing
- Clearer what's happening
- Slightly faster (no uv overhead)

**Cons**:
- Bypasses uv entirely
- Need to use full path to Python

### Option 3: Activate venv (Traditional)

```bash
source .venv/bin/activate

export WANDB_API_KEY=your_key

python src/train.py \
    --data_file data.tsv \
    --report_to wandb \
    --max_steps 10000

deactivate
```

**Pros**:
- Traditional workflow
- Can run multiple commands without prefixing

**Cons**:
- Need to remember to activate/deactivate
- Shell state changes

---

## Verification

### Test GPU Detection

```bash
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

**Expected output**:
```
CUDA available: True
CUDA version: 13.0
GPU: NVIDIA GB202 (or similar Blackwell GPU)
GPU memory: 128.00 GB
```

### Test Unsloth Detection

```bash
uv run python -c "
from unsloth import FastModel
print('✅ Unsloth successfully detected GPU!')
"
```

### Quick Training Test

```bash
# Create test dataset
echo -e "שלום\tʃalˈom\nמה קורה\tmˈa koʁˈe" > test_mini.tsv

# Run 10 steps
uv run src/train.py \
    --data_file test_mini.tsv \
    --max_steps 10 \
    --save_steps 5 \
    --batch_size 2
```

**Expected**: Creates `outputs/checkpoint-5` and `outputs/checkpoint-10`

---

## Avoiding pyproject.toml Sync Issues

### The Problem

If you run `uv run` without protection, uv might:
1. Read `pyproject.toml`
2. Sync dependencies to match it
3. **Overwrite** your Blackwell-compatible packages with standard PyPI versions
4. Result: GPU detection fails again!

### Solution: UV_NO_SYNC Environment Variable (Recommended)

**Our setup script already sets this for you!**

The `UV_NO_SYNC=1` environment variable (added in uv 0.4.18) tells uv to **never** sync the environment with `pyproject.toml`:

```bash
# Already set by setup script
export UV_NO_SYNC=1

# Now uv run won't sync - safe to use!
uv run src/train.py --data_file data.tsv
```

**Pros**:
- ✅ Set once, works forever
- ✅ Clean commands (no `--no-sync` flag needed)
- ✅ Automatic protection against accidental syncs
- ✅ Works for all uv commands in this shell

**Cons**:
- ⚠️ Affects all uv projects in this shell
- ⚠️ If you work on other projects, you might want to use per-project settings

### Alternative: Per-Command Protection

If you don't want to set `UV_NO_SYNC` globally, you can use it per-command:

```bash
# Option 1: Environment variable per-command
UV_NO_SYNC=1 uv run src/train.py [args...]

# Option 2: --no-sync flag
uv run --no-sync src/train.py [args...]

# Option 3: Direct Python (bypasses uv entirely)
.venv/bin/python src/train.py [args...]
```

### Create Convenient Aliases

Add to `~/.bashrc` for easy access:

```bash
# Use uv with no-sync protection
alias uv-dgx='UV_NO_SYNC=1 uv run'

# Or use direct Python
alias python-dgx='/home/yakov/Documents/audio/gemma3-g2p/.venv/bin/python'

# Then use:
uv-dgx src/train.py --data_file data.tsv
python-dgx src/train.py --data_file data.tsv
```

---

## Maintenance

### Updating Packages

If you need to update project dependencies (non-Blackwell packages):

```bash
# Install/update a single package
uv pip install package-name

# Update all dependencies from pyproject.toml
# NOTE: This will sync all packages, so you'll need to re-override Blackwell ones!
uv sync
uv pip install --upgrade unsloth unsloth_zoo bitsandbytes==0.48.0 transformers==4.56.2 trl==0.22.2
```

**Warning**: Avoid running `uv sync` unless necessary, as it will reset all packages to PyPI versions.

### Rebuilding Environment

If something breaks:

```bash
# Delete venv
rm -rf .venv

# Re-run setup script (handles everything)
./scripts/setup-dgx-venv.sh
```

---

## Comparison: Docker vs Native venv

| Aspect | Docker | Native venv |
|--------|--------|-------------|
| **Setup Time** | 30-60 min (one-time build) | 30-60 min (one-time setup) |
| **Startup Time** | ~5 sec (container start) | Instant (venv already there) |
| **Isolation** | Perfect (container) | Good (venv) |
| **Performance** | Slight overhead | Native speed |
| **Ease of Use** | Launch script | `uv run` or direct Python |
| **Risk of Breakage** | Low (immutable image) | Low (UV_NO_SYNC=1 prevents accidental sync) |
| **Portability** | High (same image anywhere) | Medium (need to rebuild on new machine) |
| **Debugging** | Harder (inside container) | Easier (native Python) |
| **IDE Integration** | Requires Remote Dev | Direct (just point to .venv) |

### When to Use Each

**Use Docker when**:
- You need guaranteed reproducibility
- Training on multiple machines
- Want to avoid "it works on my machine" issues
- Don't mind container overhead

**Use Native venv when**:
- Training on single DGX Spark machine
- Want maximum performance
- Prefer simpler debugging
- Comfortable with manual dependency management

---

## Troubleshooting

### Issue: `uv run` still syncs packages

**Cause**: `UV_NO_SYNC` not set in current shell

**Solution**:
```bash
# Check if it's set
echo $UV_NO_SYNC

# Set it temporarily
export UV_NO_SYNC=1

# Set it permanently
echo 'export UV_NO_SYNC=1' >> ~/.bashrc
source ~/.bashrc

# Or use direct Python as fallback
.venv/bin/python src/train.py [args...]
```

### Issue: "CUDA not available" after setup

**Cause**: PyTorch built without CUDA support

**Solution**:
```bash
# Reinstall PyTorch with CUDA support
uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Issue: Triton build fails

**Cause**: Missing build dependencies or CUDA mismatch

**Solution**:
```bash
# Install build dependencies
sudo apt install build-essential cmake ninja-build

# Verify CUDA 13.0
nvcc --version
export CUDA_HOME=/usr/local/cuda-13.0

# Retry build
cd /tmp/triton
uv pip install .
```

### Issue: xformers build fails

**Cause**: Missing submodules or wrong compute capability

**Solution**:
```bash
# Re-clone with submodules
rm -rf /tmp/xformers
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive /tmp/xformers

# Set correct compute capability for Blackwell
export TORCH_CUDA_ARCH_LIST="12.1"

# Retry
cd /tmp/xformers
uv run python setup.py install
```

### Issue: Out of memory during training

**Solution**: Same as Docker approach - reduce batch size
```bash
uv run src/train.py --data_file data.tsv --batch_size 4
```

---

## Best Practices

### 1. Ensure UV_NO_SYNC is Set

```bash
# Good ✅ (UV_NO_SYNC=1 is set by setup script)
uv run src/train.py [args...]

# Also good ✅ (direct Python, bypasses uv)
.venv/bin/python src/train.py [args...]

# Bad ❌ (if UV_NO_SYNC not set, will overwrite Blackwell packages!)
# uv run src/train.py [args...]  # without UV_NO_SYNC
```

**Verify it's set**:
```bash
echo $UV_NO_SYNC  # Should output: 1
```

### 2. Document Your Setup

Create a README in the project:
```markdown
# DGX Spark Training

## Setup (one-time)
./scripts/setup-dgx-venv.sh

## Training
# UV_NO_SYNC=1 is set by setup script
uv run src/train.py --data_file data.tsv --report_to wandb

## OR
.venv/bin/python src/train.py --data_file data.tsv --report_to wandb
```

### 3. Checkpoint Frequently

```bash
uv run src/train.py \
    --data_file data.tsv \
    --save_steps 500 \
    --save_total_limit 5 \
    --resume_from_checkpoint  # if resuming
```

### 4. Monitor with WandB

```bash
export WANDB_API_KEY=your_key
uv run src/train.py --data_file data.tsv --report_to wandb
```

---

## Summary

### Recommended Workflow

1. **One-time setup** (~45 min):
   ```bash
   ./scripts/setup-dgx-venv.sh
   ```
   This script:
   - Runs `uv sync` to install all project dependencies
   - Builds Triton and xformers from source with Blackwell support
   - Overrides unsloth and core packages with specific versions
   - **Sets `UV_NO_SYNC=1` in ~/.bashrc** to prevent future syncs

2. **Run training**:
   ```bash
   # With UV_NO_SYNC=1 set, uv run is safe to use directly
   uv run src/train.py --data_file data.tsv --report_to wandb
   ```
   Or:
   ```bash
   # Direct Python (bypasses uv entirely)
   .venv/bin/python src/train.py --data_file data.tsv --report_to wandb
   ```

3. **Resume training**:
   ```bash
   uv run src/train.py \
       --data_file data.tsv \
       --report_to wandb \
       --resume_from_checkpoint
   ```

### Key Advantages Over Docker

✅ **Native performance** - No container overhead
✅ **Easier debugging** - Direct Python access
✅ **Better IDE integration** - Just point to `.venv/`
✅ **Instant startup** - No container launch delay
✅ **Clean commands** - `UV_NO_SYNC=1` handles protection automatically

### Key Disadvantages

❌ **Initial setup time** - ~45 min to build Triton and xformers from source
❌ **Less isolated** - Not as portable as Docker image
❌ **Environment variable dependency** - Relies on `UV_NO_SYNC=1` being set

### Recommendation

- **For production/reproducibility**: Use Docker (guaranteed identical environment)
- **For development/iteration**: Use native venv with `uv run` (faster, cleaner)
- **For debugging**: Use native venv with `.venv/bin/python` (most direct)
