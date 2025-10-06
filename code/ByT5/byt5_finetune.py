import os
import time
import argparse
import unicodedata
import re

data_path = "../data"
prompt = "Generate the inflected form for the lemma: '{lemma}', according to the following morphological descriptions: '{features}'"
model_name = "google/byt5-small"

def load_data(file, prompt):
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lemma, features, target = parts[0], parts[1], parts[2]
            input_str = prompt.format(lemma=lemma, features=features)
            data.append({"input": input_str, "target": target})
    return data

def load_test_data(file, prompt):
    test_data = []
    gold_forms = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lemma, features, gold = parts[0], parts[1], parts[2]
            input_str = prompt.format(lemma=lemma, features=features)
            test_data.append(input_str)
            gold_forms.append(gold)
    return test_data, gold_forms

def normalize(s):
    # Unicode normalize, lowercase, strip spaces
    return unicodedata.normalize('NFC', s).strip().lower()

def evaluate(predictions, gold_forms, output_path, debug_mismatches=5):
    correct = 0
    total = len(gold_forms)
    mismatches = []
    formatted_predictions = []
    formatted_gold = []
    # If predictions are lines from a file, handle tab-splitting
    for pred in predictions:
        parts = pred.strip().split("\t")
        # Prediction is always in the second column, gold in the third
        if len(parts) >= 3:
            formatted_predictions.append(parts[1])
            formatted_gold.append(parts[2])
        elif len(parts) == 2:
            formatted_predictions.append(parts[1])
            formatted_gold.append("")  # No gold available
        else:
            formatted_predictions.append(parts[0])
            formatted_gold.append("")
    # Truncate to shortest length to avoid mismatch
    min_len = min(len(formatted_predictions), len(formatted_gold))
    formatted_predictions = formatted_predictions[:min_len]
    formatted_gold = formatted_gold[:min_len]
    for pred, gold in zip(formatted_predictions, formatted_gold):
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
    import re
    predictions = []
    # Infer tst file path from predictions file name
    m = re.search(r'predictions_([a-z]+)_finetuned', predictions_file)
    if not m:
        print("Could not infer language from predictions file name.")
        return
    lang = m.group(1)
    tst_file = os.path.join(data_path, f"{lang}.tst")
    gold_forms = []
    with open(tst_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gold_forms.append(parts[2])
    # Get predictions from predictions file (second or third column)
    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            pred = parts[2] if len(parts) > 2 else parts[1]
            predictions.append(pred)
    # Truncate to shortest length to avoid mismatch
    min_len = min(len(predictions), len(gold_forms))
    predictions = predictions[:min_len]
    gold_forms = gold_forms[:min_len]
    evaluate(predictions, gold_forms, predictions_file, debug_mismatches)

def preprocess(example, tokenizer):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=32)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=32)
    inputs["labels"] = targets["input_ids"]
    return inputs

def train_and_predict_all():
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
    from datasets import Dataset
    # GPU check
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    device = torch.device("cuda")
    languages = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.trn')]
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        if lang == "eng":
            continue
        train_file = os.path.join(data_path, f"{lang}.trn")
        dev_file = os.path.join(data_path, f"{lang}.dev")
        test_file = os.path.join(data_path, f"{lang}.tst")

        train_data = load_data(train_file, prompt)
        dev_data = load_data(dev_file, prompt)
        test_data, gold_forms = load_test_data(test_file, prompt)

        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), batched=False)
        dev_dataset = dev_dataset.map(lambda x: preprocess(x, tokenizer), batched=False)

        args = Seq2SeqTrainingArguments(
            output_dir=f"./byt5-{lang}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            save_total_limit=1,
            evaluation_strategy="steps",
            eval_steps=50,
            fp16=True,
            report_to=[],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        print(f"  Fine-tuning ByT5 for {lang}...")
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
                output_ids = model.generate(**inputs, max_length=32)
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        pred_elapsed = time.time() - pred_start
        print(f"  Prediction time: {pred_elapsed:.2f} seconds")

        # Save predictions
        output_path = f"output/predictions_{lang}_finetuned.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for inp, pred, gold in zip(test_data, predictions, gold_forms):
                f.write(f"{inp}\t{pred}\t{gold}\n")
        print(f"  Saved predictions to {output_path}")
        # Evaluate and write accuracy
        evaluate(predictions, gold_forms, output_path)

def predict_with_trained_models():
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
        test_data, gold_forms = load_test_data(test_file, prompt)

        model_dir = f"./byt5-{lang}/checkpoint-3750"  # or just "./byt5-{lang}" if you kept only the last checkpoint
        if not os.path.exists(model_dir):
            model_dir = f"./byt5-{lang}"  # fallback to main output dir

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

        batch_size = 16
        predictions = []
        pred_start = time.time()
        for i in tqdm(range(0, len(test_data), batch_size), desc=f"Predict-{lang}", unit="batch"):
            batch_inputs = test_data[i:i+batch_size]
            inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=32)
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        pred_elapsed = time.time() - pred_start
        print(f"  Prediction time: {pred_elapsed:.2f} seconds")

        # Save predictions
        output_path = f"output/predictions_{lang}_reloaded.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for inp, pred, gold in zip(test_data, predictions, gold_forms):
                f.write(f"{inp}\t{pred}\t{gold}\n")
        print(f"  Saved predictions to {output_path}")
        # Evaluate and write accuracy
        evaluate(predictions, gold_forms, output_path)

def main():
    parser = argparse.ArgumentParser(description="ByT5 Morphological Inflection Trainer/Evaluator")
    parser.add_argument('--train', action='store_true', help='Train and predict for all languages')
    parser.add_argument('--evaluate', type=str, help='Evaluate predictions file (provide path)')
    parser.add_argument('--predict', action='store_true', help='Predict using already trained models')
    args = parser.parse_args()

    if args.train:
        train_and_predict_all()
    elif args.predict:
        predict_with_trained_models()
    elif args.evaluate:
        evaluate_predictions_file(args.evaluate)
    else:
        print("No action specified. Use --train, --predict, or --evaluate <file>.")

if __name__ == "__main__":
    main()
