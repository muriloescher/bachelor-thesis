#!/usr/bin/env python3
"""
ByT5 Context-Aware Verb Morphology Prediction

Fine-tunes ByT5 to predict verb morphological tags using sentence context.
This script trains on data with format: lemma\ttags\tform\tcontext

The model learns to predict tags given: lemma + form + context
This helps disambiguate cases where the same verb form can have different tags
depending on context (e.g., tense/person ambiguities).
"""

import argparse
import os
import random
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_context_data(filepath: str) -> List[Tuple[str, str, str, str]]:
    """
    Load data from file with format: lemma\ttags\tform\tcontext
    
    Returns:
        List of tuples: (lemma, tags, form, context)
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 4:
                lemma, tags, form, context = parts
                data.append((lemma, tags, form, context))
    return data


def build_training_examples(data: List[Tuple[str, str, str, str]], 
                            use_context: bool = True) -> List[Dict[str, str]]:
    """
    Build training examples from data.
    
    Inverse task: predict lemma + morphological tags from inflected form (+ context)
    
    Input format options:
    - With context: "form: FORM | context: CONTEXT"
    - Without context: "form: FORM"
    
    Target: "lemma tags" (combined)
    
    Args:
        data: List of (lemma, tags, form, context) tuples
        use_context: Whether to include context in input
        
    Returns:
        List of dicts with 'input' and 'target' keys
    """
    examples = []
    for lemma, tags, form, context in data:
        if use_context:
            input_text = f"form: {form} | context: {context}"
        else:
            input_text = f"form: {form}"
        
        # Target is lemma + tags (space-separated)
        target_text = f"{lemma} {tags}".strip()
        
        examples.append({
            'input': input_text,
            'target': target_text
        })
    
    return examples


def normalize(s):
    """Unicode normalize, lowercase, strip spaces."""
    return unicodedata.normalize('NFC', s).strip().lower()


def _split_lemma_msd(s: str):
    """Split 'lemma FEATURES' into (lemma, FEATURES). If no space, returns (s, '')."""
    s = s.strip()
    if not s:
        return "", ""
    parts = s.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _msd_tokens(msd: str):
    """Lowercased ';'-separated feature tags as a set."""
    if not msd:
        return set()
    return {t.strip().lower() for t in msd.split(';') if t.strip()}


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (iterative DP)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def tokenize_function(examples, tokenizer, max_source_length=128, max_target_length=32):
    """Tokenize inputs and targets for seq2seq training."""
    model_inputs = tokenizer(
        examples['input'],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    
    # Use text_target API for proper seq2seq tokenization
    try:
        labels = tokenizer(text_target=examples['target'], max_length=max_target_length, truncation=True, padding=False)
    except TypeError:
        # Backward compatibility for older Transformers
        try:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples['target'], max_length=max_target_length, truncation=True, padding=False)
        except AttributeError:
            labels = tokenizer(examples['target'], max_length=max_target_length, truncation=True, padding=False)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def evaluate_inverse(predictions, gold_forms, output_path, debug_mismatches=5):
    """Inverse task metrics: lemma accuracy + mean Levenshtein; MSD exact-set accuracy + micro-F1."""
    total = len(predictions)
    if total == 0:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\nInverse evaluation (lemma/MSD): no samples.\n")
        print("  Inverse metrics: no samples")
        return {}

    lemma_correct = 0
    msd_exact_correct = 0
    lemma_edit_sum = 0
    tp = fp = fn = 0
    mismatches = []

    for pred, gold in zip(predictions, gold_forms):
        pred_lemma_raw, pred_msd_raw = _split_lemma_msd(pred)
        gold_lemma_raw, gold_msd_raw = _split_lemma_msd(gold)
        pred_lemma = normalize(pred_lemma_raw)
        gold_lemma = normalize(gold_lemma_raw)
        pred_tags = _msd_tokens(pred_msd_raw)
        gold_tags = _msd_tokens(gold_msd_raw)

        if gold_lemma and pred_lemma == gold_lemma:
            lemma_correct += 1
        lemma_edit_sum += _levenshtein(pred_lemma, gold_lemma)

        if pred_tags == gold_tags:
            msd_exact_correct += 1
        inter = pred_tags & gold_tags
        tp += len(inter)
        fp += len(pred_tags - gold_tags)
        fn += len(gold_tags - pred_tags)

        if len(mismatches) < debug_mismatches and (pred_lemma != gold_lemma or pred_tags != gold_tags):
            mismatches.append((pred, gold))

    lemma_acc = lemma_correct / total
    msd_acc = msd_exact_correct / total
    mean_lev = lemma_edit_sum / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\nInverse evaluation (lemma/MSD):\n")
        f.write(f"  Lemma accuracy: {lemma_acc:.4f} ({lemma_correct}/{total})\n")
        f.write(f"  Lemma mean Levenshtein distance: {mean_lev:.4f}\n")
        f.write(f"  MSD accuracy (exact set match): {msd_acc:.4f} ({msd_exact_correct}/{total})\n")
        f.write(f"  MSD micro-precision: {prec:.4f}  micro-recall: {rec:.4f}  micro-F1: {f1:.4f}\n")
        if mismatches:
            f.write("  Mismatches (prediction | gold):\n")
            for pred, gold in mismatches:
                f.write(f"  {pred} | {gold}\n")
    
    print("  Inverse metrics written:")
    print(f"    Lemma accuracy: {lemma_acc:.4f} ({lemma_correct}/{total})")
    print(f"    Lemma mean Levenshtein: {mean_lev:.4f}")
    print(f"    MSD accuracy: {msd_acc:.4f} ({msd_exact_correct}/{total})")
    print(f"    MSD micro-F1: {f1:.4f} (P={prec:.4f}, R={rec:.4f})")
    
    return {
        'lemma_accuracy': lemma_acc,
        'msd_accuracy': msd_acc,
        'msd_f1': f1,
        'mean_levenshtein': mean_lev
    }


def evaluate_model(model, tokenizer, eval_data, device, max_source_length=128, batch_size=16):
    """
    Evaluate model on evaluation data with batch prediction.
    Uses inverse task metrics: lemma accuracy + mean Levenshtein; MSD accuracy + F1.
    
    Returns:
        Dict with inverse metrics and predictions
    """
    model.eval()
    predictions = []
    references = []
    inputs = [ex['input'] for ex in eval_data]
    references = [ex['target'] for ex in eval_data]
    
    pred_start = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Evaluating", unit="batch"):
            batch_inputs = inputs[i:i+batch_size]
            tokenized = tokenizer(
                batch_inputs,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = model.generate(
                **tokenized,
                max_length=32,
                num_beams=1,
            )
            
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)
    
    pred_elapsed = time.time() - pred_start
    print(f"  Prediction time: {pred_elapsed:.2f} seconds")
    
    # Compute inverse task metrics: lemma accuracy, MSD accuracy, F1
    total = len(predictions)
    if total == 0:
        return {'predictions': [], 'references': []}
    
    lemma_correct = 0
    msd_exact_correct = 0
    lemma_edit_sum = 0
    tp = fp = fn = 0
    mismatches = []
    
    for pred, gold in zip(predictions, references):
        pred_lemma_raw, pred_msd_raw = _split_lemma_msd(pred)
        gold_lemma_raw, gold_msd_raw = _split_lemma_msd(gold)
        pred_lemma = normalize(pred_lemma_raw)
        gold_lemma = normalize(gold_lemma_raw)
        pred_tags = _msd_tokens(pred_msd_raw)
        gold_tags = _msd_tokens(gold_msd_raw)
        
        if gold_lemma and pred_lemma == gold_lemma:
            lemma_correct += 1
        lemma_edit_sum += _levenshtein(pred_lemma, gold_lemma)
        
        if pred_tags == gold_tags:
            msd_exact_correct += 1
        inter = pred_tags & gold_tags
        tp += len(inter)
        fp += len(pred_tags - gold_tags)
        fn += len(gold_tags - pred_tags)
        
        if len(mismatches) < 5 and (pred_lemma != gold_lemma or pred_tags != gold_tags):
            mismatches.append((pred, gold))
    
    lemma_acc = lemma_correct / total
    msd_acc = msd_exact_correct / total
    mean_lev = lemma_edit_sum / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    print(f"  Lemma accuracy: {lemma_acc:.4f} ({lemma_correct}/{total})")
    print(f"  Lemma mean Levenshtein: {mean_lev:.4f}")
    print(f"  MSD accuracy: {msd_acc:.4f} ({msd_exact_correct}/{total})")
    print(f"  MSD micro-F1: {f1:.4f} (P={prec:.4f}, R={rec:.4f})")
    
    if mismatches:
        print("  Example mismatches (prediction | gold):")
        for pred, gold in mismatches[:3]:
            print(f"    {pred} | {gold}")
    
    return {
        'lemma_accuracy': lemma_acc,
        'msd_accuracy': msd_acc,
        'msd_f1': f1,
        'mean_levenshtein': mean_lev,
        'predictions': predictions,
        'references': references
    }


def train_and_evaluate(
    train_file: str,
    dev_file: str,
    test_file: str,
    output_dir: str,
    use_context: bool = True,
    model_name: str = "google/byt5-small",
    checkpoint_dir: str = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_source_length: int = 128,
    max_target_length: int = 32,
    seed: int = 42,
):
    """
    Train and evaluate ByT5 model for context-aware morphological tag prediction.
    
    Args:
        train_file: Path to training data (lemma\ttags\tform\tcontext)
        dev_file: Path to dev data
        test_file: Path to test data
        output_dir: Directory to save model and results
        use_context: Whether to use context in input
        model_name: HuggingFace model name or checkpoint path
        checkpoint_dir: If provided, continue training from this checkpoint
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_source_length: Max length for input sequences
        max_target_length: Max length for target sequences
        seed: Random seed
    """
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data...")
    train_data = load_context_data(train_file)
    dev_data = load_context_data(dev_file)
    test_data = load_context_data(test_file)
    
    # Limit training data to 5k examples for faster training
    if len(train_data) > 5000:
        print(f"Limiting training data from {len(train_data)} to 5000 examples")
        train_data = train_data[:5000]
    
    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    # Build examples
    print(f"Building examples (use_context={use_context})...")
    train_examples = build_training_examples(train_data, use_context=use_context)
    dev_examples = build_training_examples(dev_data, use_context=use_context)
    test_examples = build_training_examples(test_data, use_context=use_context)
    
    print(f"Built {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test examples")
    
    # Show sample
    print("\nSample training example:")
    print(f"Input: {train_examples[0]['input'][:100]}...")
    print(f"Target: {train_examples[0]['target']}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    dev_dataset = Dataset.from_list(dev_examples)
    
    # Tokenize
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_source_length, max_target_length),
        batched=False,
        remove_columns=[c for c in train_dataset.column_names if c in ('input', 'target')],
        desc="Tokenizing train"
    )
    dev_dataset = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_source_length, max_target_length),
        batched=False,
        remove_columns=[c for c in dev_dataset.column_names if c in ('input', 'target')],
        desc="Tokenizing dev"
    )
    print(f"Tokenization complete.")
    
    # Training arguments (matching byt5_finetune.py)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=False,
        report_to=[],
        seed=seed,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Training steps: {len(train_dataset) // batch_size * num_epochs}")
    print(f"Warmup steps: {training_args.warmup_steps}")
    print(f"Base learning rate: {learning_rate}")
    
    # Sanity check on a sample batch
    import time
    sample_indices = list(range(min(8, len(train_dataset))))
    sample_batch = data_collator([train_dataset[i] for i in sample_indices])
    labels = sample_batch["labels"]
    if isinstance(labels, torch.Tensor):
        non_ignored = (labels != -100).sum(dim=1).tolist()
        print(f"Non-ignored label tokens per sample (first {len(non_ignored)}): {non_ignored}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()})
        print(f"Sanity loss on first batch: {float(out.loss):.6f}")
    model.train()
    
    train_start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint_dir if checkpoint_dir and os.path.exists(checkpoint_dir) else None)
    train_elapsed = time.time() - train_start
    print(f"Training time: {train_elapsed:.2f} seconds")
    
    # Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\nModel saved to: {final_model_dir}")
    
    # Evaluate on dev and test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("\n" + "="*50)
    print("Evaluating on DEV set...")
    print("="*50)
    dev_results = evaluate_model(model, tokenizer, dev_examples, device, max_source_length, batch_size=batch_size)
    
    print("\n" + "="*50)
    print("Evaluating on TEST set...")
    print("="*50)
    test_results = evaluate_model(model, tokenizer, test_examples, device, max_source_length, batch_size=batch_size)
    
    # Save predictions
    dev_pred_file = os.path.join(output_dir, "dev_predictions.txt")
    test_pred_file = os.path.join(output_dir, "test_predictions.txt")
    
    with open(dev_pred_file, 'w', encoding='utf-8') as f:
        for inp, pred, ref in zip([ex['input'] for ex in dev_examples], dev_results['predictions'], dev_results['references']):
            f.write(f"{inp}\t{pred}\t{ref}\n")
    
    with open(test_pred_file, 'w', encoding='utf-8') as f:
        for inp, pred, ref in zip([ex['input'] for ex in test_examples], test_results['predictions'], test_results['references']):
            f.write(f"{inp}\t{pred}\t{ref}\n")
    
    print(f"\nPredictions saved to:")
    print(f"  - {dev_pred_file}")
    print(f"  - {test_pred_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "results_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Context-Aware ByT5 Training Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Use context: {use_context}\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Max source length: {max_source_length}\n")
        f.write(f"  Seed: {seed}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Dev Lemma Accuracy: {dev_results.get('lemma_accuracy', 0):.4f}\n")
        f.write(f"  Dev Lemma Mean Levenshtein: {dev_results.get('mean_levenshtein', 0):.4f}\n")
        f.write(f"  Dev MSD Accuracy: {dev_results.get('msd_accuracy', 0):.4f}\n")
        f.write(f"  Dev MSD F1: {dev_results.get('msd_f1', 0):.4f}\n")
        f.write(f"  Test Lemma Accuracy: {test_results.get('lemma_accuracy', 0):.4f}\n")
        f.write(f"  Test Lemma Mean Levenshtein: {test_results.get('mean_levenshtein', 0):.4f}\n")
        f.write(f"  Test MSD Accuracy: {test_results.get('msd_accuracy', 0):.4f}\n")
        f.write(f"  Test MSD F1: {test_results.get('msd_f1', 0):.4f}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return {
        'dev': dev_results,
        'test': test_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train ByT5 for context-aware verb morphological tag prediction"
    )
    
    # Data arguments
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--dev', type=str, required=True,
                       help='Path to dev data file')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test data file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='google/byt5-small',
                       help='Model name or path (default: google/byt5-small)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint directory to resume from')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for model and results')
    
    # Training arguments
    parser.add_argument('--no-context', action='store_true',
                       help='Train without context (baseline comparison)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--max-source-length', type=int, default=128,
                       help='Max source sequence length (default: 128)')
    parser.add_argument('--max-target-length', type=int, default=32,
                       help='Max target sequence length (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Run training and evaluation
    results = train_and_evaluate(
        train_file=args.train,
        dev_file=args.dev,
        test_file=args.test,
        output_dir=args.output_dir,
        use_context=not args.no_context,
        model_name=args.model,
        checkpoint_dir=args.checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
    )
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()
