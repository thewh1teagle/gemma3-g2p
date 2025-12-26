# Gemma3-G2P

Based on https://unsloth.ai/blog/gemma3

## Train

See `src/prepare_dataset.py` and `src/train.py`

## Infer

See `src/infer.py`

## Export to GGUF

See `src/export_gguf.py`

## Inference with Ollama

See `ollama-infer/`

## Inference with GGUF

See `gguf-infer/`

## Continue fine-tuning

```console
mkdir outputs
uv run hf download thewh1teagle/gemma3-270m-heb-g2p --local-dir ./outputs/checkpoint-10000
uv run src/prepare_data.py --input_file raw_data.csv --output_file data.csv
uv run src/train.py --report_to tensorboard --csv_file data.csv --resume_from_checkpoint --batch_size 16 --max_steps 90000000
```

## DGX Spark Setup

To fine-tune on DGX Spark, run these commands first:

```console
uv sync
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
export UV_SYNC=0
```