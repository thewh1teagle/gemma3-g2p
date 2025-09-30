"""
uv run src/export_gguf.py --model_path ./outputs/checkpoint-7500
"""
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./outputs/checkpoint-9500")
parser.add_argument("--output_path", type=str, default="./model_gguf")
args = parser.parse_args()

Path(args.output_path).mkdir(parents=True, exist_ok=True)


# Load model & tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_path,  # local saved folder
    load_in_4bit = False,
    load_in_8bit = False,
)


TASK = (
    "Given the following Hebrew sentence, convert it to IPA phonemes.\n\n"
    "Input Format: A Hebrew sentence.\n"
    "Output Format: A string of IPA phonemes."
)

convo = [
    {'role': 'system','content': TASK},
    {'role': 'user','content': 'שלום עולם! מה קורה?'},
]

model.save_pretrained_merged('gemma-3')

tokenizer.save_pretrained('gemma-3')



tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)
# mkdir gemma-3-gguf
model.save_pretrained_gguf('gemma-3', tokenizer)

# save Modelfile


# ollama serve
# ollama create gemma-3-gguf -f Modelfile

