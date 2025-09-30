from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch

# Load model & tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = "./outputs/checkpoint-1300",  # local saved folder
    load_in_4bit = False,
    load_in_8bit = False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# Example inference
TASK = (
    "Given the following Hebrew sentence, convert it to IPA phonemes.\n\n"
    "Input Format: A Hebrew sentence.\n"
    "Output Format: A string of IPA phonemes."
)
user_message = """
שלום עולם! מה קורה?
"""
# expected_output = """
# שלום עולם! מה קורה?
# """
messages = [
    {"role": "system", "content": TASK},
    {"role": "user", "content": user_message},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
).removeprefix("<bos>")

inputs = tokenizer(text, return_tensors="pt").to("cuda")

_ = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
