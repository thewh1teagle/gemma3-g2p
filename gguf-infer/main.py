"""
wget wget https://huggingface.co/thewh1teagle/gemma3-heb-g2p-gguf/resolve/main/model.gguf
uv run main.py
"""
from llama_cpp import Llama, llama_log_set
import ctypes

def my_log_callback(level, message, user_data):
    pass
log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(my_log_callback)
llama_log_set(log_callback, ctypes.c_void_p())


# load gguf model
llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=-1,  # set -1 to put as many layers as possible on GPU
)

system_message = """Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""

user_prompt = "אז מה דעתך, האם אתה יודע לדבר עברית גם כמו שאני יודע לדבר או שאתה לא?"

# build the ollama-style template
prompt = f"""<start_of_turn>system
{system_message}<end_of_turn>
<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""

# run inference
output = llm(
    prompt,
    max_tokens=150,
    temperature=0.9,
    top_p=0.95,
    top_k=64,
    stop=["<end_of_turn>", "</s>"],
    echo=False,
)

print(output["choices"][0]["text"].strip())
