from datasets import load_dataset
from config import SYSTEM_PROMPT

def convert_to_chatml(example):
    """Add system prompt on-the-fly - no need to store in dataset"""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
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


def prepare_dataset_from_tsv(tokenizer, file_path, split='train'):
    """Load TSV directly without intermediate CSV preprocessing.

    TSV format: input\\toutput (tab-separated, no header)
    System prompt is added automatically from config.SYSTEM_PROMPT
    """
    dataset = load_dataset(
        "csv",
        data_files=file_path,
        split=split,
        delimiter="\t",  # TSV uses tabs
        column_names=["input", "output"],  # Explicit column names
        header=None  # No header in raw TSV
    )
    dataset = dataset.map(convert_to_chatml)
    dataset = dataset.map(formatting_prompts_func, batched=True, fn_kwargs={"tokenizer": tokenizer})
    return dataset