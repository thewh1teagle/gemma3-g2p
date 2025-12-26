"""
uv run onnx-infer/eval.py ./gemma3_onnx ./data_eval.csv ./report.json
"""
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
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


# Load ONNX model & tokenizer
ort_model = ORTModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = ort_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(["<end_of_turn>", "</s>"])
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

