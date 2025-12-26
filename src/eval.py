"""Evaluation utilities for gemma3-g2p training."""
import pandas as pd
import jiwer
from transformers import TrainerCallback
import wandb
from config import SYSTEM_PROMPT


def run_evaluation(model, tokenizer, eval_file, max_samples=100):
    """Run evaluation and return WER and CER metrics.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        eval_file: Path to TSV evaluation file
        max_samples: Maximum number of samples to evaluate (default: 100)

    Returns:
        Dictionary with eval_wer, eval_cer, and eval_samples
    """
    if eval_file is None:
        return None

    # Load evaluation data
    try:
        df = pd.read_csv(eval_file, sep='\t', header=0, usecols=[0, 1])
        df.columns = ['sentence', 'phonemes']
    except Exception as e:
        print(f"Warning: Could not load eval file {eval_file}: {e}")
        return None

    # Limit number of samples for faster evaluation
    if len(df) > max_samples:
        df = df.head(max_samples)

    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    for idx, row in df.iterrows():
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
        num_samples += 1

    mean_wer = total_wer / num_samples if num_samples > 0 else 0.0
    mean_cer = total_cer / num_samples if num_samples > 0 else 0.0

    return {
        "eval_wer": mean_wer,
        "eval_cer": mean_cer,
        "eval_samples": num_samples
    }


class EvaluationCallback(TrainerCallback):
    """Custom callback to run evaluation during training."""

    def __init__(self, model, tokenizer, eval_file, eval_steps):
        """Initialize the evaluation callback.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            eval_file: Path to TSV evaluation file
            eval_steps: Run evaluation every N steps
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_file = eval_file
        self.eval_steps = eval_steps
        self.last_eval_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Run evaluation at specified intervals."""
        if self.eval_file is None or self.eval_steps <= 0:
            return

        current_step = state.global_step

        # Check if it's time to evaluate
        if current_step - self.last_eval_step >= self.eval_steps:
            print(f"\nRunning evaluation at step {current_step}...")

            # Run evaluation
            eval_results = run_evaluation(self.model, self.tokenizer, self.eval_file)

            if eval_results:
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        "eval/wer": eval_results["eval_wer"],
                        "eval/cer": eval_results["eval_cer"],
                        "eval/samples": eval_results["eval_samples"],
                        "train/global_step": current_step
                    })

                print(f"Evaluation results at step {current_step}:")
                print(f"  WER: {eval_results['eval_wer']:.4f}")
                print(f"  CER: {eval_results['eval_cer']:.4f}")

            self.last_eval_step = current_step
