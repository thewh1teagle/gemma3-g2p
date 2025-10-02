# Gemma3-G2P

WIP: code not working yet

Based on https://unsloth.ai/blog/gemma3

## Train

See `src/prepare_dataset.py` and `src/train.py`

## Infer

See `src/infer.py`

## Export to ollama

1. create Modelfile

```console
FROM ./model.gguf

SYSTEM """
Given the following Hebrew sentence, convert it to IPA phonemes.
Input Format: A Hebrew sentence.
Output Format: A string of IPA phonemes.
"""

TEMPLATE "<start_of_turn>system
{{ .System }}<end_of_turn>
<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"
PARAMETER temperature 0.9
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_predict 150
PARAMETER stop <end_of_turn>
PARAMETER stop </s>
```

See `src/export_gguf.py`

And finally use

```consoole
ollama create gemma3-g2p -f ./Modelfile
```

## Infer with python

See `gguf-infer/`