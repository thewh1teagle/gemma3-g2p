"""
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv -O eval_data.tsv
uv run src/manual_eval.py ./outputs/checkpoint-10000 ./eval_data.tsv ./report.json
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


def save_report(results, total_wer, total_cer, num_samples, output_file):
    """Save evaluation report to JSON file."""
    report = {
        "summary": {
            "mean_wer": total_wer / num_samples if num_samples > 0 else 0.0,
            "mean_cer": total_cer / num_samples if num_samples > 0 else 0.0,
            "total_samples": num_samples
        },
        "individual": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# Load model & tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_path,  # local saved folder
    load_in_4bit = False,
    load_in_8bit = False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# Load evaluation data
# Read TSV file - handle header row and rename columns for consistency
df = pd.read_csv(args.input_file, sep='\t', header=0, usecols=[0, 1])
# Rename columns to standard names (handles various header formats)
df.columns = ['sentence', 'phonemes']

# Evaluate
results = []
total_wer = 0.0
total_cer = 0.0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    user_message = row['sentence']
    expected_output = row['phonemes']
    
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
    
    # Save report after each iteration
    num_processed = len(results)
    save_report(results, total_wer, total_cer, num_processed, args.output_file)

# Generate final report
report = {
    "summary": {
        "mean_wer": total_wer / len(df),
        "mean_cer": total_cer / len(df),
        "total_samples": len(df)
    },
    "individual": results
}

print(f"Mean WER: {report['summary']['mean_wer']:.4f}")
print(f"Mean CER: {report['summary']['mean_cer']:.4f}")
