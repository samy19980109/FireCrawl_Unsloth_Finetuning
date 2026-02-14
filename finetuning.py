"""
Unsloth LoRA SFT Fine-Tuning Script

Takes a ShareGPT JSONL dataset (output of main.py) and fine-tunes an LLM
using Unsloth's LoRA pipeline with 4-bit quantization.

Usage:
    python finetuning.py --dataset ./output/dataset_sharegpt.jsonl
    python finetuning.py --dataset ./output/dataset_sharegpt.jsonl --max-steps 10
    python finetuning.py --help
"""

import argparse

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_sharegpt_dataset(path: str):
    """Load a local ShareGPT JSONL file into an HF Dataset."""
    dataset = load_dataset("json", data_files=path, split="train")
    print(f"Loaded {len(dataset)} examples from {path}")
    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, max_seq_length: int):
    """Load a base model with 4-bit quantization via Unsloth."""
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer


def apply_lora(model, max_seq_length: int, lora_rank: int, lora_alpha: int):
    """Apply LoRA adapters to the model."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    print(f"LoRA applied (r={lora_rank}, alpha={lora_alpha})")
    return model


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_dataset(dataset, tokenizer, chat_template: str):
    """Apply chat template and standardize ShareGPT format."""
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping={
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt",
        },
        map_eos_token=True,
    )

    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Dataset formatted with '{chat_template}' chat template")
    return dataset, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokenizer, dataset, args: argparse.Namespace):
    """Run SFT training."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    print("Starting training...")
    stats = trainer.train()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

    return stats


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, args: argparse.Namespace):
    """Save LoRA adapters locally, and optionally push to HF Hub or export GGUF."""
    print(f"Saving LoRA adapters to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    if args.gguf:
        gguf_dir = f"{args.output_dir}_gguf"
        print(f"Exporting GGUF ({args.gguf}) to {gguf_dir}")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.gguf)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM with LoRA using Unsloth (consumes ShareGPT JSONL from main.py)."
    )

    # Data
    parser.add_argument("--dataset", default="./output/dataset_sharegpt.jsonl",
                        help="Path to ShareGPT JSONL dataset (default: ./output/dataset_sharegpt.jsonl)")

    # Model
    parser.add_argument("--model", default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                        help="Base model ID from HF or Unsloth (default: unsloth/gpt-oss-20b-unsloth-bnb-4bit)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--chat-template", default="chatml",
                        help="Chat template: chatml, llama, mistral, zephyr, etc. (default: chatml)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")

    # Training
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size (default: 2)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps, -1 = use epochs (default: -1)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--logging-steps", type=int, default=1,
                        help="Log every N steps (default: 1)")

    # Output
    parser.add_argument("--output-dir", default="./finetuned_model",
                        help="Directory to save LoRA adapters (default: ./finetuned_model)")
    parser.add_argument("--push-to-hub", default=None,
                        help="Optional HuggingFace Hub repo to push model (e.g. username/model-name)")
    parser.add_argument("--gguf", default=None,
                        help="Optional GGUF quantization method: q4_k_m, q8_0, f16, etc.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load dataset
    dataset = load_sharegpt_dataset(args.dataset)

    # 2. Load model + tokenizer
    model, tokenizer = load_model(args.model, args.max_seq_length)

    # 3. Apply LoRA
    model = apply_lora(model, args.max_seq_length, args.lora_rank, args.lora_alpha)

    # 4. Format dataset with chat template
    dataset, tokenizer = format_dataset(dataset, tokenizer, args.chat_template)

    # 5. Train
    train(model, tokenizer, dataset, args)

    # 6. Save
    save_model(model, tokenizer, args)

    print("\nDone! Fine-tuned model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
