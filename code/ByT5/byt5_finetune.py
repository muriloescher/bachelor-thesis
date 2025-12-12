import os
import time
import argparse
import unicodedata
import re

data_path = "../data/ud"
# Default (forward) prompt: lemma+features -> inflected form
prompt = "Generate the inflected form for: {lemma} {features}"
# Inverse prompt: inflected form -> lemma+features
inverse_prompt = "Generate the lemma and morphological tags for the following inflected verb: {form}"
model_name = "google/byt5-small"

def load_data(file, prompt, inverse=False):
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Strip whitespace and skip empty targets
            lemma, features, target = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if inverse:
                # Inverse task: input is the inflected surface form; target is "lemma features"
                inflected = target
                if not inflected:
                    continue
                input_str = inverse_prompt.format(form=inflected)
                inv_target = f"{lemma} {features}".strip()
                if not inv_target:
                    continue
                data.append({"input": input_str, "target": inv_target})
            else:
                # Forward task: input is "lemma features"; target is the inflected surface form
                if not target:
                    continue
                input_str = prompt.format(lemma=lemma, features=features)
                data.append({"input": input_str, "target": target})
    return data

def load_test_data(file, prompt, inverse=False):
    test_data = []
    gold_forms = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lemma, features, gold = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if inverse:
                # Inverse task: input is the inflected form (gold); expected output is "lemma features"
                inflected = gold
                if not inflected:
                    continue
                input_str = inverse_prompt.format(form=inflected)
                expected = f"{lemma} {features}".strip()
                if not expected:
                    continue
                test_data.append(input_str)
                gold_forms.append(expected)
            else:
                # Forward task: input is lemma+features; expected output is inflected form
                if not gold:
                    continue
                input_str = prompt.format(lemma=lemma, features=features)
                test_data.append(input_str)
                gold_forms.append(gold)
    return test_data, gold_forms

def normalize(s):
    # Unicode normalize, lowercase, strip spaces
    return unicodedata.normalize('NFC', s).strip().lower()

def read_triples(file):
    """Read raw triples (lemma, features, form) from a TSV file. Skips ill-formed lines."""
    triples = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lemma, features, form = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not lemma and not features and not form:
                continue
            triples.append((lemma, features, form))
    return triples

def build_forward_examples(triples):
    """Map triples to forward examples using the global forward prompt."""
    data = []
    for lemma, features, form in triples:
        if not form:
            continue
        input_str = prompt.format(lemma=lemma, features=features)
        data.append({"input": input_str, "target": form})
    return data

def build_inverse_examples(triples):
    """Map triples to inverse examples using the global inverse prompt."""
    data = []
    for lemma, features, form in triples:
        inflected = form
        if not inflected:
            continue
        input_str = inverse_prompt.format(form=inflected)
        inv_target = f"{lemma} {features}".strip()
        if not inv_target:
            continue
        data.append({"input": input_str, "target": inv_target})
    return data

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

def evaluate_inverse(predictions, gold_forms, output_path, debug_mismatches=5):
    """Inverse task metrics: lemma accuracy + mean Levenshtein; MSD exact-set accuracy + micro-F1."""
    total = len(predictions)
    if total == 0:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\nInverse evaluation (lemma/MSD): no samples.\n")
        print("  Inverse metrics: no samples")
        return

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

# Standalone evaluation mode
def evaluate_predictions_file(predictions_file, debug_mismatches=5, inverse=False):
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
    if inverse:
        evaluate_inverse(predictions, gold_forms, predictions_file, debug_mismatches)
    else:
        evaluate(predictions, gold_forms, predictions_file, debug_mismatches)

def preprocess(example, tokenizer):
    # Tokenize inputs without padding; collator will pad dynamically.
    model_inputs = tokenizer(example["input"], max_length=128, truncation=True)

    # Tokenize targets properly for seq2seq. Prefer the `text_target` API when available.
    try:
        labels_enc = tokenizer(text_target=example["target"], max_length=32, truncation=True)
    except TypeError:
        # Backward compatibility for older Transformers
        try:
            with tokenizer.as_target_tokenizer():
                labels_enc = tokenizer(example["target"], max_length=32, truncation=True)
        except AttributeError:
            labels_enc = tokenizer(example["target"], max_length=32, truncation=True)

    model_inputs["labels"] = labels_enc["input_ids"]
    return model_inputs

def train_and_predict_all(selected_langs=None, inverse=False, bidirectional=False, resume=False, resume_steps=None):
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
        mode_desc = 'bidirectional' if bidirectional else ('inverse' if inverse else 'forward')
        if bidirectional and inverse:
            print("Note: --bidirectional overrides --inverse; proceeding with bidirectional training.")
        print(f"\nProcessing language: {lang} ({mode_desc})")
        train_file = os.path.join(data_path, f"{lang}.trn")
        dev_file = os.path.join(data_path, f"{lang}.dev")
        test_file = os.path.join(data_path, f"{lang}.tst")

        # Build datasets according to direction
        if bidirectional:
            # Create a 50/50 mix of forward and inverse examples for train/dev
            train_triples = read_triples(train_file)
            dev_triples = read_triples(dev_file)

            # Deterministic random sampling to select which examples become forward vs inverse.
            # This avoids blocky / sorted splits (e.g. all verbs then all nouns) by shuffling
            # the triples with a fixed seed and then splitting 50/50. The seed is set above
            # for reproducibility.
            rng = random.Random(seed)
            # Shuffle copies so original lists remain intact if needed elsewhere
            train_shuf = train_triples.copy()
            dev_shuf = dev_triples.copy()
            rng.shuffle(train_shuf)
            rng.shuffle(dev_shuf)

            half_tr = len(train_shuf) // 2
            half_dv = len(dev_shuf) // 2
            train_data = build_forward_examples(train_shuf[:half_tr]) + build_inverse_examples(train_shuf[half_tr:])
            dev_data = build_forward_examples(dev_shuf[:half_dv]) + build_inverse_examples(dev_shuf[half_dv:])

            # Prepare both test directions
            test_fwd_inputs, test_fwd_gold = load_test_data(test_file, prompt, inverse=False)
            test_inv_inputs, test_inv_gold = load_test_data(test_file, inverse_prompt, inverse=True)
        else:
            use_prompt = inverse_prompt if inverse else prompt
            train_data = load_data(train_file, use_prompt, inverse=inverse)
            dev_data = load_data(dev_file, use_prompt, inverse=inverse)
            test_data, gold_forms = load_test_data(test_file, use_prompt, inverse=inverse)

        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize & drop raw text columns so the data collator doesn't see string fields (avoids tensor conversion error)
        train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), batched=False, remove_columns=[c for c in train_dataset.column_names if c in ("input","target")])
        dev_dataset = dev_dataset.map(lambda x: preprocess(x, tokenizer), batched=False, remove_columns=[c for c in dev_dataset.column_names if c in ("input","target")])
        
        # Add data collator for proper batching
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            output_dir=f"./byt5-{'bidir-' if bidirectional else ('inverse-' if inverse else '')}{lang}",
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
            generation_max_length=32,    # Match target length
            remove_unused_columns=False,  # Keep all columns for seq2seq
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )

        print(f"  Fine-tuning ByT5 for {lang} ({mode_desc})...")
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
        # Optionally resume from a checkpoint for this language
        resume_path = None
        out_dir = args.output_dir
        if resume_steps is not None:
            candidate = os.path.join(out_dir, f"checkpoint-{resume_steps}")
            if os.path.isdir(candidate):
                resume_path = candidate
            else:
                print(f"  Requested checkpoint-{resume_steps} not found in {out_dir}; starting fresh.")
        elif resume:
            # Find latest checkpoint-* directory
            ckpts = []
            if os.path.isdir(out_dir):
                for name in os.listdir(out_dir):
                    path = os.path.join(out_dir, name)
                    if os.path.isdir(path) and name.startswith("checkpoint-"):
                        try:
                            step = int(name.split('-', 1)[1])
                        except ValueError:
                            step = -1
                        ckpts.append((step, path))
            if ckpts:
                ckpts.sort(key=lambda x: x[0])
                resume_path = ckpts[-1][1]
                print(f"  Resuming from latest checkpoint: {resume_path}")

        train_start = time.time()
        if resume_path:
            trainer.train(resume_from_checkpoint=resume_path)
        else:
            trainer.train()
        train_elapsed = time.time() - train_start
        print(f"  Training time: {train_elapsed:.2f} seconds")

        # Predict on test set
        if bidirectional:
            # Forward direction
            print(f"  Predicting on test set for {lang} (forward)...")
            batch_size = 16
            predictions_fwd = []
            pred_start = time.time()
            for i in tqdm(range(0, len(test_fwd_inputs), batch_size), desc=f"Predict-{lang}-fwd", unit="batch"):
                batch_inputs = test_fwd_inputs[i:i+batch_size]
                inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_length=32)
                batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                predictions_fwd.extend(batch_preds)
            pred_elapsed = time.time() - pred_start
            print(f"  Prediction time (forward): {pred_elapsed:.2f} seconds")

            out_fwd = f"output/predictions_{lang}_finetuned_bidir_forward.txt"
            with open(out_fwd, "w", encoding="utf-8") as f:
                for inp, pred, gold in zip(test_fwd_inputs, predictions_fwd, test_fwd_gold):
                    f.write(f"{inp}\t{pred}\t{gold}\n")
            print(f"  Saved predictions to {out_fwd}")
            evaluate(predictions_fwd, test_fwd_gold, out_fwd)

            # Inverse direction
            print(f"  Predicting on test set for {lang} (inverse)...")
            predictions_inv = []
            pred_start = time.time()
            for i in tqdm(range(0, len(test_inv_inputs), batch_size), desc=f"Predict-{lang}-inv", unit="batch"):
                batch_inputs = test_inv_inputs[i:i+batch_size]
                inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_length=32)
                batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                predictions_inv.extend(batch_preds)
            pred_elapsed = time.time() - pred_start
            print(f"  Prediction time (inverse): {pred_elapsed:.2f} seconds")

            out_inv = f"output/predictions_{lang}_finetuned_bidir_inverse.txt"
            with open(out_inv, "w", encoding="utf-8") as f:
                for inp, pred, gold in zip(test_inv_inputs, predictions_inv, test_inv_gold):
                    f.write(f"{inp}\t{pred}\t{gold}\n")
            print(f"  Saved predictions to {out_inv}")
            evaluate_inverse(predictions_inv, test_inv_gold, out_inv)
        else:
            print(f"  Predicting on test set for {lang} ({'inverse' if inverse else 'forward'})...")
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
            output_path = f"output/predictions_{lang}_finetuned{'_inverse' if inverse else ''}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for inp, pred, gold in zip(test_data, predictions, gold_forms):
                    f.write(f"{inp}\t{pred}\t{gold}\n")
            print(f"  Saved predictions to {output_path}")
            # Evaluate and write metrics
            if inverse:
                evaluate_inverse(predictions, gold_forms, output_path)
            else:
                evaluate(predictions, gold_forms, output_path)

def predict_with_trained_models(inverse=False, bidirectional=False, selected_langs=None, inverse_only=False, test_file_override=None):
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        if bidirectional and inverse:
            print("Note: --bidirectional overrides --inverse; proceeding with bidirectional predictions.")
        if inverse_only and bidirectional:
            mode_desc = 'bidirectional (inverse only)'
        else:
            mode_desc = 'bidirectional' if bidirectional else ('inverse' if inverse else 'forward')
        print(f"\nPredicting with trained model for language: {lang} ({mode_desc})")
        
        # Use custom test file if provided, otherwise use default
        if test_file_override:
            test_file = test_file_override
        else:
            test_file = os.path.join(data_path, f"{lang}.tst")
        if bidirectional:
            test_fwd_inputs, test_fwd_gold = load_test_data(test_file, prompt, inverse=False)
            test_inv_inputs, test_inv_gold = load_test_data(test_file, inverse_prompt, inverse=True)
        else:
            use_prompt = inverse_prompt if inverse else prompt
            test_data, gold_forms = load_test_data(test_file, use_prompt, inverse=inverse)

        base_dir = f"./byt5-{'bidir-' if bidirectional else ('inverse-' if inverse else '')}{lang}"
        model_dir = f"{base_dir}/checkpoint-3750"
        if not os.path.exists(model_dir):
            model_dir = base_dir  # fallback to main output dir

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

        if bidirectional:
            batch_size = 16
            # Forward predictions (skip if inverse_only is set)
            if not inverse_only:
                predictions_fwd = []
                pred_start = time.time()
                for i in tqdm(range(0, len(test_fwd_inputs), batch_size), desc=f"Predict-{lang}-fwd", unit="batch"):
                    batch_inputs = test_fwd_inputs[i:i+batch_size]
                    inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, max_length=32)
                    batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    predictions_fwd.extend(batch_preds)
                pred_elapsed = time.time() - pred_start
                print(f"  Prediction time (forward): {pred_elapsed:.2f} seconds")

                out_fwd = f"output/predictions_{lang}_reloaded_bidir_forward.txt"
                with open(out_fwd, "w", encoding="utf-8") as f:
                    for inp, pred, gold in zip(test_fwd_inputs, predictions_fwd, test_fwd_gold):
                        f.write(f"{inp}\t{pred}\t{gold}\n")
                print(f"  Saved predictions to {out_fwd}")
                evaluate(predictions_fwd, test_fwd_gold, out_fwd)

            # Inverse predictions
            predictions_inv = []
            pred_start = time.time()
            for i in tqdm(range(0, len(test_inv_inputs), batch_size), desc=f"Predict-{lang}-inv", unit="batch"):
                batch_inputs = test_inv_inputs[i:i+batch_size]
                inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_length=32)
                batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                predictions_inv.extend(batch_preds)
            pred_elapsed = time.time() - pred_start
            print(f"  Prediction time (inverse): {pred_elapsed:.2f} seconds")

            out_inv = f"output/predictions_{lang}_reloaded_bidir_inverse.txt"
            with open(out_inv, "w", encoding="utf-8") as f:
                for inp, pred, gold in zip(test_inv_inputs, predictions_inv, test_inv_gold):
                    f.write(f"{inp}\t{pred}\t{gold}\n")
            print(f"  Saved predictions to {out_inv}")
            evaluate_inverse(predictions_inv, test_inv_gold, out_inv)
        else:
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
            output_path = f"output/predictions_{lang}_reloaded{'_inverse' if inverse else ''}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for inp, pred, gold in zip(test_data, predictions, gold_forms):
                    f.write(f"{inp}\t{pred}\t{gold}\n")
            print(f"  Saved predictions to {output_path}")
            # Evaluate and write metrics per language
            if inverse:
                evaluate_inverse(predictions, gold_forms, output_path)
            else:
                evaluate(predictions, gold_forms, output_path)

def main():
    parser = argparse.ArgumentParser(description="ByT5 Morphological Inflection Trainer/Evaluator")
    parser.add_argument('--train', action='store_true', help='Train and predict for all (or selected) languages')
    parser.add_argument('--evaluate', type=str, help='Evaluate predictions file (provide path)')
    parser.add_argument('--predict', action='store_true', help='Predict using already trained models')
    parser.add_argument('--langs', type=str, help='Comma-separated list of language codes to train (e.g. eng,ita,por)')
    parser.add_argument('--inverse', action='store_true', help='Use inverse task (inflected -> lemma+features)')
    parser.add_argument('--bidirectional', action='store_true', help='Train on a 50/50 mix of forward and inverse; evaluate both directions')
    parser.add_argument('--inverse-only', action='store_true', help='For bidirectional models: predict only inverse direction')
    parser.add_argument('--test-file', type=str, help='Custom test file path (overrides default data/<lang>.tst)')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint in each language dir')
    parser.add_argument('--resume-steps', type=int, help='Resume training from a specific checkpoint-<steps> in each language dir')
    args = parser.parse_args()

    if args.train:
        langs = args.langs.split(',') if args.langs else None
        train_and_predict_all(selected_langs=langs, inverse=args.inverse, bidirectional=args.bidirectional, resume=args.resume, resume_steps=args.resume_steps)
    elif args.predict:
        langs = args.langs.split(',') if args.langs else None
        predict_with_trained_models(inverse=args.inverse, bidirectional=args.bidirectional, selected_langs=langs, inverse_only=args.inverse_only, test_file_override=args.test_file)
    elif args.evaluate:
        evaluate_predictions_file(args.evaluate, inverse=args.inverse)
    else:
        print("No action specified. Use --train, --predict, or --evaluate <file>.")

if __name__ == "__main__":
    main()
