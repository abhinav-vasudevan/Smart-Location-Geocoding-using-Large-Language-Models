"""
train_t5.py
===========
Fine-tunes a fresh T5-small model on the noisy-to-correct Indian location
dataset following the methodology described in:

  "Smart Location Geocoding using Large Language Models"
  Maheshwari, Vasudevan, Ramudu, Panigrahi (2025)
  VIT Chennai & DRDO CAIR Bangalore

Paper specs implemented:
  - T5-small (60M parameters), fresh from HuggingFace
  - SentencePiece tokenizer, vocab size 32k
  - 2 training epochs
  - Batch size 8, gradient accumulation steps 2 (effective batch = 16)
  - AdamW optimizer with weight decay
  - Linear warmup + linear decay LR schedule
  - FP16 mixed-precision (GPU only; falls back gracefully on CPU)
  - Early stopping based on validation loss (patience = 2)
  - Stratified train / val / test split by Indian state
  - Fuzzy accuracy metric (SequenceMatcher ≥ 0.85 threshold, as in accuracy.py)

Run on Kaggle / Colab (GPU recommended):
  python train_t5.py

Run locally (CPU):
  python train_t5.py --no_fp16 --max_train_samples 50000

Output: t5_corrector_final/ — drop-in replacement for the existing model folder.
"""

import os
import argparse
import logging
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants — mirror the paper's exact configuration
# ---------------------------------------------------------------------------

TASK_PREFIX       = "rectify location: "   # T5 seq2seq task prefix
MAX_INPUT_LENGTH  = 64                      # Location names are short; 64 is more than enough
MAX_TARGET_LENGTH = 64
MODEL_NAME        = "t5-small"              # Fresh base model (60M params)
OUTPUT_DIR        = "t5_corrector_final"    # Drop-in replacement for existing folder
DATA_PATH         = "final_training_geocoded.csv"
FUZZY_THRESHOLD   = 0.85                    # As used in accuracy.py
RANDOM_SEED       = 42

# Paper training config
NUM_EPOCHS             = 2
PER_DEVICE_BATCH_SIZE  = 8
GRAD_ACCUMULATION      = 2                  # Effective batch = 16
LEARNING_RATE          = 5e-4
WEIGHT_DECAY           = 0.01
WARMUP_RATIO           = 0.05              # 5% of total steps as warmup
EARLY_STOPPING_PATIENCE = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for location rectification")
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH,
        help="Path to final_training_geocoded.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Limit training rows (useful for CPU testing, e.g. --max_train_samples 50000)"
    )
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="Disable FP16 (required on CPU or older GPUs)"
    )
    parser.add_argument(
        "--use_augmented", action="store_true",
        help="Also include the noisy_address_augmented column as additional training pairs"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Data Loading & Stratified Split
# ---------------------------------------------------------------------------

def load_and_split(data_path: str, max_train_samples: int | None, use_augmented: bool) -> DatasetDict:
    """
    Loads the CSV, optionally includes augmented pairs, and creates a
    stratified train / val / test split by Indian state.

    Returns a HuggingFace DatasetDict with keys: train, validation, test.
    """
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Drop rows with missing inputs or targets
    df = df.dropna(subset=["noisy_address", "correct_address", "state"])
    df = df[df["noisy_address"].str.strip() != ""]
    df = df[df["correct_address"].str.strip() != ""]
    logger.info(f"Rows after cleaning: {len(df):,}")

    # Build base pairs
    pairs = df[["noisy_address", "correct_address", "state"]].rename(
        columns={"noisy_address": "input", "correct_address": "target"}
    )

    # Optionally include augmented noisy variants as extra training pairs
    if use_augmented and "noisy_address_augmented" in df.columns:
        aug = df[["noisy_address_augmented", "correct_address", "state"]].dropna(
            subset=["noisy_address_augmented"]
        )
        aug = aug.rename(columns={"noisy_address_augmented": "input", "correct_address": "target"})
        pairs = pd.concat([pairs, aug], ignore_index=True)
        logger.info(f"Rows after including augmented pairs: {len(pairs):,}")

    # Stratified split by state: 80% train / 10% val / 10% test
    # Uses multi-class stratification; rare states fall into train if < 2 samples
    train_df, temp_df = train_test_split(
        pairs, test_size=0.20, random_state=RANDOM_SEED,
        stratify=pairs["state"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_df["state"]
    )

    logger.info(f"Split sizes → train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    # Optional: cap training size for quick CPU tests
    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(max_train_samples, random_state=RANDOM_SEED)
        logger.info(f"Training capped at {max_train_samples:,} samples for quick run")

    def to_hf_dataset(dataframe: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "input":  dataframe["input"].tolist(),
            "target": dataframe["target"].tolist(),
        })

    return DatasetDict({
        "train":      to_hf_dataset(train_df),
        "validation": to_hf_dataset(val_df),
        "test":       to_hf_dataset(test_df),
    })


# ---------------------------------------------------------------------------
# 3. Tokenisation
# ---------------------------------------------------------------------------

def build_tokenize_fn(tokenizer: T5Tokenizer):
    """
    Returns a preprocessing function compatible with Dataset.map().

    Prepends TASK_PREFIX to each input (standard T5 practice).
    Labels have padding token IDs replaced with -100 so the Trainer
    ignores them when computing cross-entropy loss.
    """
    def tokenize(batch):
        # Prepend task prefix exactly as described in the paper
        inputs = [TASK_PREFIX + text for text in batch["input"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            padding=False,          # DataCollatorForSeq2Seq handles dynamic padding
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=MAX_TARGET_LENGTH,
                padding=False,
                truncation=True,
            )

        # Replace padding id with -100 → ignored by cross-entropy loss
        label_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in ids]
            for ids in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize


# ---------------------------------------------------------------------------
# 4. Fuzzy Accuracy Metric (matching accuracy.py logic from the project)
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer: T5Tokenizer):
    """
    Returns a compute_metrics function for the Seq2SeqTrainer that reports:
      - fuzzy_accuracy: % of predictions with SequenceMatcher ratio ≥ 0.85
      - exact_accuracy: % of predictions that exactly match the target (case-insensitive)
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode predicted token IDs
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        fuzzy_correct = 0
        exact_correct = 0

        for pred, label in zip(decoded_preds, decoded_labels):
            ratio = SequenceMatcher(None, pred.lower(), label.lower()).ratio()
            if ratio >= FUZZY_THRESHOLD:
                fuzzy_correct += 1
            if pred.lower() == label.lower():
                exact_correct += 1

        total = len(decoded_preds)
        return {
            "fuzzy_accuracy": round(fuzzy_correct / total * 100, 2),
            "exact_accuracy": round(exact_correct / total * 100, 2),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# 5. Main Training Entry Point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    use_fp16 = not args.no_fp16 and torch.cuda.is_available()
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device.upper()}  |  FP16: {use_fp16}")

    # ------------------------------------------------------------------
    # Load & split data
    # ------------------------------------------------------------------
    dataset = load_and_split(args.data_path, args.max_train_samples, args.use_augmented)

    # ------------------------------------------------------------------
    # Load fresh T5-small tokenizer and model from HuggingFace
    # We use the standard t5-small, NOT the already fine-tuned one,
    # so the model learns from scratch on our dataset.
    # ------------------------------------------------------------------
    logger.info(f"Loading fresh {MODEL_NAME} tokenizer and model from HuggingFace...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # Tokenise datasets
    # ------------------------------------------------------------------
    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["input", "target"],
        desc="Tokenising",
    )

    # ------------------------------------------------------------------
    # Data collator — handles dynamic padding per batch
    # ------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,   # Matches our -100 label masking above
        pad_to_multiple_of=8 if use_fp16 else None,  # Required for FP16 tensor cores
    )

    # ------------------------------------------------------------------
    # Compute total training steps for warmup scheduling
    # ------------------------------------------------------------------
    steps_per_epoch = len(tokenized["train"]) // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)
    logger.info(f"Total training steps: {total_steps:,}  |  Warmup steps: {warmup_steps:,}")

    # ------------------------------------------------------------------
    # Seq2SeqTrainingArguments — exact paper config
    # ------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # --- Epochs & batch ---
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE * 2,  # Larger eval batch (no gradients)
        gradient_accumulation_steps=GRAD_ACCUMULATION,

        # --- Optimiser (paper: Adam with weight decay) ---
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",          # Linear decay after warmup
        warmup_steps=warmup_steps,

        # --- FP16 mixed precision (paper: Tesla P100 on Kaggle) ---
        fp16=use_fp16,

        # --- Evaluation strategy ---
        eval_strategy="epoch",               # Evaluate at end of each epoch
        save_strategy="epoch",
        save_total_limit=2,                  # Keep only the 2 best checkpoints

        # --- Generation settings for seq2seq eval ---
        predict_with_generate=True,          # Required for seq2seq metric computation
        generation_max_length=MAX_TARGET_LENGTH,

        # --- Best model tracking (for early stopping) ---
        load_best_model_at_end=True,
        metric_for_best_model="fuzzy_accuracy",
        greater_is_better=True,

        # --- Logging ---
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
        report_to="none",                    # Disable wandb/tensorboard unless user sets up

        # --- Reproducibility ---
        seed=RANDOM_SEED,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting fine-tuning...")
    logger.info("=" * 60)
    trainer.train()

    # ------------------------------------------------------------------
    # Save best model — same folder structure as existing t5_corrector_final/
    # so it is a direct drop-in replacement without changing any other code.
    # ------------------------------------------------------------------
    logger.info(f"Saving best model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved successfully.")

    # ------------------------------------------------------------------
    # Final evaluation on held-out test set
    # ------------------------------------------------------------------
    logger.info("Running final evaluation on test set...")
    test_results = trainer.predict(
        tokenized["test"],
        metric_key_prefix="test",
    )
    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info(f"  Fuzzy Accuracy : {test_results.metrics.get('test_fuzzy_accuracy', 'N/A')}%")
    logger.info(f"  Exact Accuracy : {test_results.metrics.get('test_exact_accuracy', 'N/A')}%")
    logger.info(f"  Test Loss      : {test_results.metrics.get('test_loss', 'N/A'):.4f}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
