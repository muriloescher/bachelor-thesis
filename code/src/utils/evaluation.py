"""Utility functions for evaluation metrics."""
import unicodedata


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


def _msd_tokens(msd: str, exclude_tags=None):
    """Lowercased ';'-separated feature tags as a set."""
    if not msd:
        return set()
    tags = {t.strip().lower() for t in msd.split(';') if t.strip()}
    if exclude_tags:
        tags = tags - {t.lower() for t in exclude_tags}
    return tags


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


def evaluate_forward(predictions, gold_forms, output_path=None, debug_mismatches=5):
    """
    Evaluate forward task (inflection): exact match accuracy.
    
    Args:
        predictions: List of predicted forms
        gold_forms: List of gold standard forms
        output_path: Optional path to write results
        debug_mismatches: Number of mismatches to show
        
    Returns:
        dict with accuracy and count metrics
    """
    correct = 0
    total = len(predictions)
    edit_sum = 0
    mismatches = []
    
    for pred, gold in zip(predictions, gold_forms):
        pred_n = normalize(pred)
        gold_n = normalize(gold)

        if pred_n == gold_n:
            correct += 1
        elif len(mismatches) < debug_mismatches:
            mismatches.append((pred, gold))

        edit_sum += _levenshtein(pred_n, gold_n)
    
    accuracy = correct / total if total > 0 else 0.0
    mean_lev = edit_sum / total if total > 0 else 0.0
    
    results = {
        'accuracy': accuracy,
        'mean_levenshtein': mean_lev,
        'correct': correct,
        'total': total,
        'mismatches': mismatches
    }
    
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"\nForward Task Evaluation:\n")
            f.write(f"  Accuracy: {accuracy:.4f} ({correct}/{total})\n")
            f.write(f"  Mean Levenshtein distance: {mean_lev:.4f}\n")
            if mismatches:
                f.write("  Example Mismatches (prediction | gold):\n")
                for pred, gold in mismatches:
                    f.write(f"    {pred} | {gold}\n")
    
    return results


def evaluate_inverse(predictions, gold_forms, output_path=None, debug_mismatches=5, exclude_tags=None):
    """
    Evaluate inverse task (analysis): lemma accuracy + MSD accuracy + micro-F1.
    
    Args:
        predictions: List of predicted "lemma FEATURES" strings
        gold_forms: List of gold standard "lemma FEATURES" strings
        output_path: Optional path to write results
        debug_mismatches: Number of mismatches to show
        exclude_tags: Optional list of tags to exclude from comparison (e.g., ['FIN'])
        
    Returns:
        dict with lemma and MSD metrics
    """
    total = len(predictions)
    if total == 0:
        return {
            'lemma_accuracy': 0.0,
            'mean_levenshtein': 0.0,
            'msd_accuracy': 0.0,
            'msd_precision': 0.0,
            'msd_recall': 0.0,
            'msd_f1': 0.0
        }

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
        pred_tags = _msd_tokens(pred_msd_raw, exclude_tags)
        gold_tags = _msd_tokens(gold_msd_raw, exclude_tags)

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

    results = {
        'lemma_accuracy': lemma_acc,
        'lemma_correct': lemma_correct,
        'mean_levenshtein': mean_lev,
        'msd_accuracy': msd_acc,
        'msd_correct': msd_exact_correct,
        'msd_precision': prec,
        'msd_recall': rec,
        'msd_f1': f1,
        'total': total,
        'mismatches': mismatches
    }

    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\nInverse Task Evaluation (Lemma/MSD):\n")
            f.write(f"  Lemma accuracy: {lemma_acc:.4f} ({lemma_correct}/{total})\n")
            f.write(f"  Lemma mean Levenshtein distance: {mean_lev:.4f}\n")
            f.write(f"  MSD accuracy (exact set match): {msd_acc:.4f} ({msd_exact_correct}/{total})\n")
            f.write(f"  MSD micro-precision: {prec:.4f}  micro-recall: {rec:.4f}  micro-F1: {f1:.4f}\n")
            if mismatches:
                f.write("  Example Mismatches (prediction | gold):\n")
                for pred, gold in mismatches:
                    f.write(f"    {pred} | {gold}\n")

    return results
