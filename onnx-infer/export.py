"""Simple example: Export Gemma3 270M with LoRA adapter to ONNX and generate text.

Usage:
    uv pip install onnxruntime peft
    uv run examples/gemma3.py
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.exporters.onnx import onnx_export_from_model
from optimum.onnxruntime import ORTModelForCausalLM
import time


# Load base model and merge with LoRA adapter
base_model_id = "google/gemma-3-270m-it"  # The base model for your LoRA
adapter_id = "thewh1teagle/gemma3-heb-g2p"

base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
model = model.merge_and_unload()  # Merge LoRA weights into base model

tokenizer = AutoTokenizer.from_pretrained(adapter_id)

# Export merged model to ONNX
print("Exporting to ONNX...")
output_dir = "gemma3_onnx"
onnx_export_from_model(
    model=model,
    output=output_dir,
    task="text-generation-with-past"
)

# Save tokenizer to the same directory
tokenizer.save_pretrained(output_dir)

# Load the exported ONNX model
ort_model = ORTModelForCausalLM.from_pretrained(output_dir)

# Chat with instruction-tuned model
system_message = """Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""

user_prompt = "אז מה דעתך, האם אתה יודע לדבר עברית גם כמו שאני יודע לדבר או שאתה לא?"

conversation = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with parameters similar to the working Ollama script
start_time = time.time()
outputs = ort_model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids(["<end_of_turn>", "</s>"])
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only the model's response (after the last "model" turn)
if "<start_of_turn>model" in response:
    response = response.split("<start_of_turn>model")[-1].strip()
    # Remove any end tokens
    for end_token in ["<end_of_turn>", "</s>"]:
        response = response.replace(end_token, "")

print(response.strip())

print(f"Time taken: {time.time() - start_time:.2f} seconds")