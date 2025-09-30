from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]}
        ]
    }


def formatting_prompts_func(examples, tokenizer):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }



def prepare_dataset(tokenizer):
    # https://huggingface.co/datasets/Thytu/ChessInstruct
    dataset = load_dataset("Thytu/ChessInstruct", split = "train[:10000]")
    dataset = dataset.map(
        convert_to_chatml,
    )
    dataset = dataset.map(formatting_prompts_func, batched = True, fn_kwargs = {"tokenizer": tokenizer})
    return dataset


def prepare_dataset_from_csv(tokenizer):
    dataset = load_dataset("csv", data_files="knesset_phonemes_v1.csv", split="train[:10000]")
    dataset = dataset.map(
        convert_to_chatml,
    )
    dataset = dataset.map(formatting_prompts_func, batched=True, fn_kwargs={"tokenizer": tokenizer})
    return dataset