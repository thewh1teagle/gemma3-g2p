"""
uv run src/eval.py ./outputs/checkpoint-10000 ./data_eval.csv ./report.json
"""
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
import argparse
import jiwer
import json
from tqdm import tqdm
from config import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()


# Load model & tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_path,  # local saved folder
    load_in_4bit = False,
    load_in_8bit = False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# Load evaluation data
df = pd.read_csv(args.input_file, sep='\t', header=None, names=['input', 'expected'])

# Evaluate
results = []
total_wer = 0.0
total_cer = 0.0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    user_message = row['input']
    expected_output = row['expected']
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix("<bos>")
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    
    prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    wer = jiwer.wer(expected_output, prediction)
    cer = jiwer.cer(expected_output, prediction)
    
    total_wer += wer
    total_cer += cer
    
    results.append({
        "input": user_message,
        "expected": expected_output,
        "predicted": prediction,
        "wer": wer,
        "cer": cer
    })

# Generate report
report = {
    "summary": {
        "mean_wer": total_wer / len(df),
        "mean_cer": total_cer / len(df),
        "total_samples": len(df)
    },
    "individual": results
}

with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"Mean WER: {report['summary']['mean_wer']:.4f}")
print(f"Mean CER: {report['summary']['mean_cer']:.4f}")
