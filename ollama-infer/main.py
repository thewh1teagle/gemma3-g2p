#!/usr/bin/env python3
import ollama

system_message = """Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""

user_prompt = "אז מה דעתך, האם אתה יודע לדבר עברית גם כמו שאני יודע לדבר או שאתה לא?"

# build ollama-style template
prompt = f"""<start_of_turn>system
{system_message}<end_of_turn>
<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""

# run inference with ollama
response = ollama.chat(
    model="gemma3-g2p",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ],
    options={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "num_predict": 150,
        "stop": ["<end_of_turn>", "</s>"]
    }
)

print(response["message"]["content"].strip())
