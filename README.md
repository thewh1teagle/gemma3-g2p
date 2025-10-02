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
uv run hf download thewh1teagle/gemma3-heb-g2p --local-dir ./outputs/checkpoint-10000
uv run src/prepare_data.py knesset_phonemes_v1.txt data.csv 5000
uv run src/train.py --report_to wandb --csv_file data.csv --resume_from_checkpoint --batch_size 16
```