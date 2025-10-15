import os
import time
import argparse
import unicodedata
import re

data_path = "../data"
prompt_direct = "Generate the inflected form for: {lemma} {features}"
prompt_inverse = "Generate the lemma and morphological tags for the following inflected verb: {inflected}"
model_name = "google/byt5-small"

def load_data(file, prompt, inverse=False):
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Strip whitespace and skip empty targets
            lemma, features, inflected = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not inflected:
                continue
            if not inverse:
                # Direct task: (lemma, features) -> inflected
                input_str = prompt.format(lemma=lemma, features=features)
                target_str = inflected
            else:
                # Inverse task: inflected -> (lemma, features)
                input_str = prompt.format(inflected=inflected)
                target_str = f"{lemma} {features}"
            data.append({"input": input_str, "target": target_str})
    return data

def load_test_data(file, prompt, inverse=False):
    test_data = []
    gold_forms = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lemma, features, inflected = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not inflected:
                continue
            if not inverse:
                # Direct: input from lemma+features, gold is inflected
                input_str = prompt.format(lemma=lemma, features=features)
                gold_str = inflected
            else:
                # Inverse: input from inflected, gold is "lemma features"
                input_str = prompt.format(inflected=inflected)
                gold_str = f"{lemma} {features}"
            test_data.append(input_str)
            gold_forms.append(gold_str)
    return test_data, gold_forms

def normalize(s):
    # Unicode normalize, lowercase, strip spaces
    return unicodedata.normalize('NFC', s).strip().lower()

def evaluate(predictions, gold_forms, output_path, debug_mismatches=5):
    correct = 0
    total = len(predictions)
    mismatches = []
    for pred, gold in zip(predictions, gold_forms):
        if normalize(pred) == normalize(gold):
            correct += 1
        elif len(mismatches) < debug_mismatches:
            mismatches.append((pred, gold))
    accuracy = correct / total if total > 0 else 0.0
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})\n")
        if mismatches:
            f.write("Mismatches (prediction | gold):\n")
            for pred, gold in mismatches:
                f.write(f"{pred} | {gold}\n")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total}) written to {output_path}")
    if mismatches:
        print("  Example mismatches:")
        for pred, gold in mismatches:
            print(f"    Pred: '{pred}' | Gold: '{gold}'")

# Standalone evaluation mode
def evaluate_predictions_file(predictions_file, debug_mismatches=5):
    predictions = []
    gold_forms = []
    # Read predictions file and extract 2nd column (predictions) and 3rd column (gold)
    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            predictions.append(parts[1])  # 2nd column: prediction
            gold_forms.append(parts[2])   # 3rd column: gold
    evaluate(predictions, gold_forms, predictions_file, debug_mismatches)

def preprocess(example, tokenizer, target_max_length=32):
    # Tokenize inputs without padding; collator will pad dynamically.
    model_inputs = tokenizer(example["input"], max_length=128, truncation=True)

    # Tokenize targets properly for seq2seq. Prefer the `text_target` API when available.
    try:
        labels_enc = tokenizer(text_target=example["target"], max_length=target_max_length, truncation=True)
    except TypeError:
        # Backward compatibility for older Transformers
        try:
            with tokenizer.as_target_tokenizer():
                labels_enc = tokenizer(example["target"], max_length=target_max_length, truncation=True)
        except AttributeError:
            labels_enc = tokenizer(example["target"], max_length=target_max_length, truncation=True)

    model_inputs["labels"] = labels_enc["input_ids"]
    return model_inputs

def train_and_predict_all(selected_langs=None, inverse=False):
    import torch
    import random
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset
    # Set fixed random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GPU check
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    device = torch.device("cuda")
    languages = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.trn')]
    languages.sort()
    if selected_langs:
        wanted = [l.strip() for l in selected_langs if l.strip()]
        missing = [l for l in wanted if l not in languages]
        languages = [l for l in languages if l in wanted]
        if missing:
            print(f"Warning: requested languages not found and will be skipped: {', '.join(missing)}")
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        train_file = os.path.join(data_path, f"{lang}.trn")
        dev_file = os.path.join(data_path, f"{lang}.dev")
        test_file = os.path.join(data_path, f"{lang}.tst")

        # Select prompt and target length based on mode
        if not inverse:
            prompt = prompt_direct
            target_max_len = 32
            out_dir = f"./byt5-{lang}"
            out_pred = f"output/predictions_{lang}_finetuned.txt"
        else:
            prompt = prompt_inverse
            target_max_len = 64
            out_dir = f"./byt5-{lang}-inverse"
            out_pred = f"output/predictions_{lang}_inverse_finetuned.txt"

        train_data = load_data(train_file, prompt, inverse=inverse)
        dev_data = load_data(dev_file, prompt, inverse=inverse)
        test_data, gold_forms = load_test_data(test_file, prompt, inverse=inverse)

        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize & drop raw text columns so the data collator doesn't see string fields (avoids tensor conversion error)
        train_dataset = train_dataset.map(
            lambda x: preprocess(x, tokenizer, target_max_length=target_max_len),
            batched=False,
            remove_columns=[c for c in train_dataset.column_names if c in ("input", "target")]
        )
        dev_dataset = dev_dataset.map(
            lambda x: preprocess(x, tokenizer, target_max_length=target_max_len),
            batched=False,
            remove_columns=[c for c in dev_dataset.column_names if c in ("input", "target")]
        )

        # Add data collator for proper batching
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=1e-4,  # Conservative learning rate for ByT5
            warmup_steps=100,    # Reduced warmup steps (was too high)
            weight_decay=0.01,   # Weight decay for regularization
            lr_scheduler_type="linear",  # Explicit scheduler type
            logging_steps=10,
            save_steps=100,
            save_total_limit=1,
            evaluation_strategy="steps",  # Correct parameter name
            eval_steps=50,
            fp16=False,
            report_to=[],
            seed=seed,
            predict_with_generate=True,  # Important for seq2seq
            generation_max_length=target_max_len,    # Match target length
            remove_unused_columns=False,  # Keep all columns for seq2seq
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )

        print(f"  Fine-tuning ByT5 for {lang} ({'inverse' if inverse else 'direct'})...")
        print(f"  Training steps: {len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Base learning rate: {args.learning_rate}")

        # Sanity check: ensure labels are not all -100 and model computes non-zero loss on a small batch
        import torch
        sample_indices = list(range(min(8, len(train_dataset))))
        sample_batch = data_collator([train_dataset[i] for i in sample_indices])
        labels = sample_batch["labels"]
        if isinstance(labels, torch.Tensor):
            non_ignored = (labels != -100).sum(dim=1).tolist()
            print(f"  Non-ignored label tokens per sample (first {len(non_ignored)}): {non_ignored}")
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()})
            print(f"  Sanity loss on first batch: {float(out.loss):.6f}")
        model.train()
        train_start = time.time()
        trainer.train()
        train_elapsed = time.time() - train_start
        print(f"  Training time: {train_elapsed:.2f} seconds")

        # Predict on test set
        print(f"  Predicting on test set for {lang}...")
        batch_size = 16
        predictions = []
        pred_start = time.time()
        for i in tqdm(range(0, len(test_data), batch_size), desc=f"Predict-{lang}", unit="batch"):
            batch_inputs = test_data[i:i+batch_size]
            inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=target_max_len)
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        pred_elapsed = time.time() - pred_start
        print(f"  Prediction time: {pred_elapsed:.2f} seconds")

        # Save predictions
        output_path = out_pred
        with open(output_path, "w", encoding="utf-8") as f:
            for inp, pred, gold in zip(test_data, predictions, gold_forms):
                f.write(f"{inp}\t{pred}\t{gold}\n")
        print(f"  Saved predictions to {output_path}")
        # Evaluate and write accuracy
        evaluate(predictions, gold_forms, output_path)

def predict_with_trained_models(inverse=False):
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    device = torch.device("cuda")
    languages = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.trn')]
    for lang in languages:
        print(f"\nPredicting with trained model for language: {lang}")
        test_file = os.path.join(data_path, f"{lang}.tst")
        # Select prompt and model dir based on mode
        if not inverse:
            prompt = prompt_direct
            base_dir = f"./byt5-{lang}"
            out_pred = f"output/predictions_{lang}_reloaded.txt"
            gen_max_len = 32
        else:
            prompt = prompt_inverse
            base_dir = f"./byt5-{lang}-inverse"
            out_pred = f"output/predictions_{lang}_inverse_reloaded.txt"
            gen_max_len = 64

        test_data, gold_forms = load_test_data(test_file, prompt, inverse=inverse)

        model_dir = f"{base_dir}/checkpoint-3750"  # or just base_dir if you kept only the last checkpoint
        if not os.path.exists(model_dir):
            model_dir = base_dir  # fallback to main output dir

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

        batch_size = 16
        predictions = []
        pred_start = time.time()
        for i in tqdm(range(0, len(test_data), batch_size), desc=f"Predict-{lang}", unit="batch"):
            batch_inputs = test_data[i:i+batch_size]
            inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=gen_max_len)
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        pred_elapsed = time.time() - pred_start
        print(f"  Prediction time: {pred_elapsed:.2f} seconds")

        # Save predictions
        output_path = out_pred
        with open(output_path, "w", encoding="utf-8") as f:
            for inp, pred, gold in zip(test_data, predictions, gold_forms):
                f.write(f"{inp}\t{pred}\t{gold}\n")
        print(f"  Saved predictions to {output_path}")
        # Evaluate and write accuracy
        evaluate(predictions, gold_forms, output_path)

def main():
    parser = argparse.ArgumentParser(description="ByT5 Morphological Inflection Trainer/Evaluator")
    parser.add_argument('--train', action='store_true', help='Train and predict for all (or selected) languages')
    parser.add_argument('--evaluate', type=str, help='Evaluate predictions file (provide path)')
    parser.add_argument('--predict', action='store_true', help='Predict using already trained models')
    parser.add_argument('--langs', type=str, help='Comma-separated list of language codes to train (e.g. eng,ita,por)')
    parser.add_argument('--inverse', action='store_true', help='Use inverse task (inflected -> lemma + tags)')
    args = parser.parse_args()

    if args.train:
        langs = args.langs.split(',') if args.langs else None
        train_and_predict_all(selected_langs=langs, inverse=args.inverse)
    elif args.predict:
        predict_with_trained_models(inverse=args.inverse)
    elif args.evaluate:
        evaluate_predictions_file(args.evaluate)
    else:
        print("No action specified. Use --train, --predict, or --evaluate <file>.")

if __name__ == "__main__":
    main()
