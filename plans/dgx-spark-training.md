# DGX Spark Training Setup Plan

## Problem

Training fails on NVIDIA DGX Spark (ARM + Blackwell GPU) with:
```
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

**Root Cause**:
- DGX Spark has ARM CPU + Blackwell (GB202) GPU
- Requires specific CUDA 13.0, Triton, and xformers builds for Blackwell support
- Standard pip installations don't include ARM64 + Blackwell support
- Unsloth needs custom builds to detect the GPU

**Official Solution**: Docker container with pre-built dependencies

**Your Requirements**:
1. ✅ Good developer experience (not just Jupyter notebooks)
2. ✅ Data persistence (don't lose training data/checkpoints)
3. ✅ Ability to pause/resume training
4. ✅ Keep current project structure (Python scripts, not notebooks)

---

## Solution: Docker + Volume Mounts + Current Workflow

### Important: Why NOT Use uv Inside Container

**Problem with uv**:
- `uv run` creates/uses `.venv/` based on `pyproject.toml`
- Installs packages from PyPI (standard builds)
- **Does NOT use** the special Blackwell-compatible packages installed in Docker image
- Result: Same GPU detection error!

**Solution**:
- Use Docker's system Python with pre-installed packages
- Run scripts with `python` directly (not `uv run`)
- Keep uv on host for local development/testing (non-GPU tasks)

### Architecture

```
Host (DGX Spark)
├── /home/yakov/Documents/audio/gemma3-g2p/  (project code - mounted)
│   └── .venv/                               (NOT mounted - excluded)
├── ~/.cache/huggingface/                    (HF cache - mounted)
└── Docker Container (unsloth-dgx-spark)
    ├── Project mounted at /workspace/gemma3-g2p
    ├── HF cache mounted at /root/.cache/huggingface
    ├── System Python with Blackwell packages (pre-installed)
    └── Training runs with persistence
```

**Key Benefits**:
- Code changes on host instantly reflected in container
- Training checkpoints persist on host
- HuggingFace models cached on host (not re-downloaded)
- Can stop/start container without losing data
- IDE (VS Code) can connect to container

---

## Implementation Plan

### Step 1: Create Custom Dockerfile

**File**: `Dockerfile.dgx-spark`

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set CUDA environment variables for Blackwell
ENV CUDA_HOME=/usr/local/cuda-13.0/
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# Install triton from source for Blackwell support
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
    pip install -r python/requirements.txt && \
    pip install . && \
    cd ..

# Install xformers from source for Blackwell support
RUN git clone --depth=1 https://github.com/facebookresearch/xformers --recursive && \
    cd xformers && \
    export TORCH_CUDA_ARCH_LIST="12.1" && \
    python setup.py install && \
    cd ..

# Install unsloth and core dependencies with specific versions for Blackwell
RUN pip install unsloth unsloth_zoo bitsandbytes==0.48.0 transformers==4.56.2 trl==0.22.2

# Copy pyproject.toml to extract additional dependencies
COPY pyproject.toml .

# Install remaining project dependencies from pyproject.toml
# (excluding ones we already installed with specific versions)
RUN python -c "\
import tomllib; \
with open('pyproject.toml', 'rb') as f: \
    deps = tomllib.load(f)['project']['dependencies']; \
    exclude = ['unsloth', 'transformers', 'trl', 'bitsandbytes']; \
    deps = [d for d in deps if not any(pkg in d.lower() for pkg in exclude)]; \
    print(' '.join(deps)) \
" | xargs pip install

# Set working directory
WORKDIR /workspace/gemma3-g2p

# Keep container running
CMD ["/bin/bash"]
```

**Why this works**:
- Based on official DGX Spark Dockerfile
- Installs Blackwell-compatible packages with specific versions
- Auto-installs remaining dependencies from `pyproject.toml`
- Maintains single source of truth for dependencies
- All dependencies pre-installed in system Python

### Step 2: Build Docker Image

```bash
cd /home/yakov/Documents/audio/gemma3-g2p

# Download or create Dockerfile
cat > Dockerfile.dgx-spark << 'EOF'
[content from above]
EOF

# Build image (this will take ~30-60 minutes first time)
docker build -f Dockerfile.dgx-spark -t unsloth-dgx-spark .
```

**Note**: Build only once, reuse forever (unless dependencies change)

### Step 3: Create Container Launch Script

**File**: `scripts/run-dgx-container.sh`

```bash
#!/bin/bash
# Launch DGX Spark training container with proper volume mounts

# Get absolute paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

# Create cache directory if it doesn't exist
mkdir -p "${HF_CACHE_DIR}"

# Launch container
docker run -it --rm \
    --gpus=all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PROJECT_DIR}:/workspace/gemma3-g2p" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${HOME}/.wandb:/root/.wandb" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="gemma3" \
    -w /workspace/gemma3-g2p \
    unsloth-dgx-spark \
    "$@"
```

**Volume Mounts Explained**:
- `-v PROJECT_DIR:/workspace/gemma3-g2p` - Your code (editable from host)
- `-v HF_CACHE:/root/.cache/huggingface` - Model cache (persist downloads)
- `-v ~/.wandb:/root/.wandb` - WandB config/cache
- `-e WANDB_API_KEY` - Pass WandB API key from host

**Flags Explained**:
- `--gpus=all` - Access all GPUs
- `--rm` - Remove container on exit (data persists via volumes)
- `--net=host` - Use host networking (WandB, SSH, etc.)
- `--ipc=host` - Shared memory for PyTorch DataLoader
- `--ulimit memlock=-1` - Unlimited locked memory for CUDA

### Step 4: Usage Workflow

#### A. Interactive Development (Recommended)

```bash
# Make script executable
chmod +x scripts/run-dgx-container.sh

# Launch container interactively
./scripts/run-dgx-container.sh

# Inside container:
# Run training (packages already installed in Docker image)
python src/train.py --data_file data.tsv --report_to wandb --max_steps 10000

# Training saves to ./outputs/ which persists on host
# Can Ctrl+C to stop, resume later with --resume_from_checkpoint
```

#### B. One-Shot Training (Background)

```bash
# Run training command directly (no uv sync needed!)
./scripts/run-dgx-container.sh bash -c "
    python src/train.py \
        --data_file data.tsv \
        --report_to wandb \
        --max_steps 10000 \
        --save_steps 500
"
```

#### C. Resume Training After Pause

```bash
# Same command with resume flag
./scripts/run-dgx-container.sh bash -c "
    python src/train.py \
        --data_file data.tsv \
        --report_to wandb \
        --max_steps 20000 \
        --save_steps 500 \
        --resume_from_checkpoint
"
```

---

## Data Persistence Strategy

### What Gets Persisted (On Host)

1. **Training Checkpoints**: `./outputs/checkpoint-*`
   - Automatically saved every 500 steps
   - Mounted from host, survives container restarts

2. **Model Cache**: `~/.cache/huggingface/`
   - Downloaded models (gemma-3-270m-it)
   - Never re-downloaded

3. **WandB Logs**: `~/.wandb/`
   - Training metrics, runs history
   - Synced to cloud + local cache

4. **Your Code**: `./src/`, `./pyproject.toml`, etc.
   - Edited on host with your IDE
   - Changes instantly available in container

5. **Data**: `./data.tsv`, `./data_eval.csv`
   - On host, accessible in container

### What Doesn't Persist (Container-Only)

1. **Nothing significant!**
   - Python packages are baked into Docker image
   - No need to reinstall on each run
   - Just launch and train

2. **Temporary files**: `/tmp/` inside container
   - Gets recreated, doesn't matter

---

## Development Experience Improvements

### Option 1: VS Code Remote Container (Best DevEx)

**File**: `.devcontainer/devcontainer.json`

```json
{
  "name": "DGX Spark Training",
  "dockerFile": "../Dockerfile.dgx-spark",
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace/gemma3-g2p,type=bind",
    "source=${localEnv:HOME}/.cache/huggingface,target=/root/.cache/huggingface,type=bind",
    "source=${localEnv:HOME}/.wandb,target=/root/.wandb,type=bind"
  ],
  "runArgs": [
    "--gpus=all",
    "--net=host",
    "--ipc=host",
    "--ulimit", "memlock=-1",
    "--ulimit", "stack=67108864"
  ],
  "containerEnv": {
    "WANDB_API_KEY": "${localEnv:WANDB_API_KEY}",
    "WANDB_PROJECT": "gemma3"
  },
  "workspaceFolder": "/workspace/gemma3-g2p",
  "extensions": [
    "ms-python.python",
    "ms-python.vscode-pylance"
  ]
}
```

**Usage**:
1. Open project in VS Code
2. Install "Remote - Containers" extension
3. Click "Reopen in Container"
4. Full IDE experience inside container!

### Option 2: Attach to Running Container

```bash
# Terminal 1: Start container with training
./scripts/run-dgx-container.sh

# Terminal 2: Attach to same container for debugging
docker exec -it $(docker ps -q -f ancestor=unsloth-dgx-spark) bash
```

---

## Training Best Practices for DGX Spark

### 1. Checkpoint Strategy

```python
# In train.py, already configured:
parser.add_argument("--save_steps", type=int, default=500)  # Save every 500 steps
parser.add_argument("--resume_from_checkpoint", action="store_true")

# SFTConfig:
save_total_limit=5  # Keep only last 5 checkpoints (save disk space)
```

**Recommended**:
- Save every 500 steps for 10K+ step runs
- Save every 100 steps for shorter runs
- Use `save_total_limit=5` to avoid filling disk

### 2. Monitor Training

```bash
# Option A: WandB (recommended)
export WANDB_API_KEY=your_key
python src/train.py --report_to wandb --data_file data.tsv

# View on wandb.ai in real-time from any device

# Option B: TensorBoard
python src/train.py --report_to tensorboard --data_file data.tsv

# In another terminal:
docker exec -it $(docker ps -q -f ancestor=unsloth-dgx-spark) \
    tensorboard --logdir outputs --host 0.0.0.0
```

### 3. Graceful Shutdown

```bash
# In training terminal, press Ctrl+C
# Training will save checkpoint before exiting

# Resume later:
python src/train.py --data_file data.tsv --resume_from_checkpoint
```

### 4. Long Training Runs (Multi-Day)

Use `tmux` or `screen` inside container:

```bash
# Launch container
./scripts/run-dgx-container.sh

# Inside container, start tmux
tmux new -s training

# Run training
python src/train.py --data_file data.tsv --max_steps 50000 --report_to wandb

# Detach: Ctrl+B, then D
# Exit container safely

# Later, reattach:
./scripts/run-dgx-container.sh
tmux attach -t training
```

---

## File Structure After Setup

```
gemma3-g2p/
├── .devcontainer/
│   └── devcontainer.json          # VS Code container config
├── scripts/
│   └── run-dgx-container.sh       # Container launcher
├── Dockerfile.dgx-spark           # Custom Dockerfile
├── outputs/                       # Training checkpoints (persisted)
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── data.tsv                       # Training data (persisted)
├── src/                           # Your code (persisted)
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   └── ...
└── pyproject.toml                 # Dependencies (persisted)
```

---

## Testing the Setup

### Step 1: Verify GPU Detection

```bash
./scripts/run-dgx-container.sh bash -c "python -c '
import torch
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"CUDA version: {torch.version.cuda}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
    print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")
'"
```

**Expected Output**:
```
CUDA available: True
CUDA version: 13.0
GPU count: 1
GPU name: NVIDIA GB202 (or similar Blackwell GPU)
GPU memory: 128.00 GB
```

### Step 2: Verify Unsloth Detection

```bash
./scripts/run-dgx-container.sh bash -c "
python -c '
from unsloth import FastModel
print(\"✅ Unsloth successfully detected GPU!\")
'
"
```

### Step 3: Quick Training Test

```bash
# Create small test dataset
echo -e "שלום\tʃalˈom\nמה קורה\tmˈa koʁˈe" > test_mini.tsv

# Run 10 steps of training
./scripts/run-dgx-container.sh bash -c "
python src/train.py \
    --data_file test_mini.tsv \
    --max_steps 10 \
    --save_steps 5 \
    --batch_size 2
"
```

**Expected**: Training completes, creates `outputs/checkpoint-5` and `outputs/checkpoint-10`

---

## Troubleshooting

### Issue: "Docker: command not found"

**Solution**: Install Docker on DGX Spark
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in
```

### Issue: "permission denied while trying to connect to Docker daemon"

**Solution**: Add user to docker group
```bash
sudo usermod -aG docker $USER
newgrp docker  # or logout/login
```

### Issue: "Out of memory" during training

**Solution**: Reduce batch size
```bash
python src/train.py --data_file data.tsv --batch_size 4  # instead of 8
```

Or use gradient accumulation:
```python
# In train.py SFTConfig:
per_device_train_batch_size = 2,
gradient_accumulation_steps = 4,  # Effective batch size = 2 * 4 = 8
```

### Issue: "No space left on device"

**Solution**: Clean up old checkpoints
```bash
# Keep only last 2 checkpoints
ls -dt outputs/checkpoint-* | tail -n +3 | xargs rm -rf
```

Or reduce `save_total_limit` in train.py

### Issue: Container exits immediately

**Solution**: Run with bash to debug
```bash
./scripts/run-dgx-container.sh bash
# Now you're inside, can debug
```

---

## Comparison: Official vs Our Approach

| Aspect | Official Tutorial | Our Approach |
|--------|------------------|--------------|
| **Interface** | Jupyter Notebook | Python scripts + CLI |
| **Dependency Mgmt** | pip in Docker | pip in Docker (same) |
| **Code Editing** | Inside container (Jupyter) | Host editor (VS Code) |
| **Data Persistence** | Manual volume mounts | Automatic (script) |
| **Resuming Training** | Manual checkpoint management | Built-in `--resume_from_checkpoint` |
| **Development** | Download/edit notebooks | Git workflow, proper code |
| **Monitoring** | Jupyter output | WandB/TensorBoard |
| **Multi-file Projects** | Hard (notebook cells) | Easy (Python modules) |

---

## Next Steps

1. **Build Docker image** (30-60 min, one-time)
   ```bash
   docker build -f Dockerfile.dgx-spark -t unsloth-dgx-spark .
   ```

2. **Create launch script**
   ```bash
   mkdir -p scripts
   # Copy script from above
   chmod +x scripts/run-dgx-container.sh
   ```

3. **Test GPU detection**
   ```bash
   ./scripts/run-dgx-container.sh bash -c "python -c 'import torch; print(torch.cuda.is_available())'"
   ```

4. **Run actual training**
   ```bash
   export WANDB_API_KEY=your_key
   ./scripts/run-dgx-container.sh bash -c "
       python src/train.py \
           --data_file data.tsv \
           --report_to wandb \
           --max_steps 10000
   "
   ```

5. **(Optional) Set up VS Code Remote**
   - Copy `.devcontainer/devcontainer.json`
   - Install Remote-Containers extension
   - Reopen in container

---

## Benefits Summary

✅ **GPU Support**: Blackwell GPU properly detected with custom builds
✅ **Data Persistence**: All checkpoints, code, cache persisted on host
✅ **Resume Training**: Seamless checkpoint resume with `--resume_from_checkpoint`
✅ **Good DevEx**: Edit code on host, use your favorite IDE
✅ **Monitoring**: WandB/TensorBoard integration
✅ **Reproducible**: Docker ensures consistent environment
✅ **No Data Loss**: Volumes persist across container restarts
✅ **Your Workflow**: Keep using Python scripts, Git, proper code structure

This combines the reliability of Docker with the developer experience you're used to!
