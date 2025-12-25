"""
# https://wandb.ai/authorize?ref=models
export WANDB_PROJECT=gemma3
export WANDB_API_KEY=...

uv run src/train.py --report_to wandb --data_file data.tsv

To resume training:
uv run src/train.py --report_to wandb --data_file data.tsv --resume_from_checkpoint --batch_size 16

With custom weight decay:
uv run src/train.py --report_to wandb --data_file data.tsv --weight_decay 0.01

To upload:
uv run hf upload --repo-type model thewh1teagle/gemma3-heb-g2p ./outputs/checkpoint-10000
"""
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from data import prepare_dataset_from_tsv
import argparse

def enable_fast_training(full_finetuning=False):
    # Enable fast training

    max_seq_length = 2048
    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

        # Other popular models!
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.3-70B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-270m-it",
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = full_finetuning, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )
    return model, tokenizer


def add_lora_adapters(model, tokenizer):

    # Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model


def get_chat_template(tokenizer):
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma3",
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--data_file", type=str, default="data.tsv", help="Path to TSV data file (input\\toutput)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--full_finetuning", action="store_true")
    args = parser.parse_args()

    model, tokenizer = enable_fast_training(full_finetuning=args.full_finetuning)
    tokenizer = get_chat_template(tokenizer)
    model = add_lora_adapters(model, tokenizer)
    dataset = prepare_dataset_from_tsv(tokenizer, file_path=args.data_file, split='train') # change to train[:10000] for smaller dataset

    # Train the model
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = args.max_steps,
            learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir="outputs",
            report_to = args.report_to, # Use this for WandB etc
            save_steps=args.save_steps,
            save_total_limit=5
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


    # Inference

    messages = [
        {'role': 'system','content':dataset['conversations'][10][0]['content']},
        {"role" : 'user', 'content' : dataset['conversations'][10][1]['content']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    ).removeprefix('<bos>')

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 125,
        temperature = 1, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    # Save the model
    model.save_pretrained("gemma-3")  # Local saving
    tokenizer.save_pretrained("gemma-3")

main()