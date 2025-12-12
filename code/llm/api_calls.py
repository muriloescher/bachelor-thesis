import requests
import json
import os
import time
import unicodedata
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"
QWEN_MODEL = "qwen/qwen3-8b"

HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

EXAMPLE_INPUT = "gosta\tAquele cliente gosta apenas de vinho branco ."
EXAMPLE_PREDICTION = "gostar\tV;IND;SG;3;PRS"
PROMPT = "Generate the lemma and morphological tags for the following inflected verb and context. IMPORTANT: Provide ONLY the answer in the exact format 'lemma<tab>tags' with NO explanations, NO commentary, NO additional text: "


# ============================================================================
# Helper functions for inverse task evaluation (from byt5_context_finetune.py)
# ============================================================================

def normalize(s: str) -> str:
    """Normalize string: Unicode NFD, lowercase, strip."""
    return unicodedata.normalize('NFD', s).lower().strip()


def _split_lemma_msd(s: str):
    """Split 'lemma FEATURES' into (lemma, MSD)."""
    parts = s.strip().split(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return parts[0], ""
    return "", ""


def _msd_tokens(msd: str):
    """Parse semicolon-separated tags into set."""
    if not msd:
        return set()
    return set(t.strip() for t in msd.split(';') if t.strip())


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance."""
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


def evaluate_inverse(predictions, gold_forms, output_file, debug_mismatches=5):
    """
    Compute inverse task metrics: lemma accuracy + mean Levenshtein; MSD exact-set accuracy + micro-F1.
    
    Args:
        predictions: List of predicted "lemma tags" strings
        gold_forms: List of gold "lemma tags" strings
        output_file: File handle to write results
        debug_mismatches: Number of mismatches to show
        
    Returns:
        Dict with metrics
    """
    total = len(predictions)
    if total == 0:
        output_file.write("\nInverse evaluation (lemma/MSD): no samples.\n")
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

    output_file.write("\nInverse evaluation (lemma/MSD):\n")
    output_file.write(f"  Lemma accuracy: {lemma_acc:.4f} ({lemma_correct}/{total})\n")
    output_file.write(f"  Lemma mean Levenshtein distance: {mean_lev:.4f}\n")
    output_file.write(f"  MSD accuracy (exact set match): {msd_acc:.4f} ({msd_exact_correct}/{total})\n")
    output_file.write(f"  MSD micro-precision: {prec:.4f}  micro-recall: {rec:.4f}  micro-F1: {f1:.4f}\n")
    if mismatches:
        output_file.write("  Mismatches (prediction | gold):\n")
        for pred, gold in mismatches:
            output_file.write(f"    {pred} | {gold}\n")
    
    return {
        'lemma_accuracy': lemma_acc,
        'msd_accuracy': msd_acc,
        'msd_f1': f1,
        'mean_levenshtein': mean_lev,
        'msd_precision': prec,
        'msd_recall': rec
    }


def load_test_data(filepath):
    """Load test data from file with format: lemma\ttags\tform\tcontext"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 4:
                lemma, tags, form, context = parts
                data.append({
                    'lemma': lemma,
                    'tags': tags,
                    'form': form,
                    'context': context,
                    'line_num': line_num
                })
    return data


def create_prompt(form, context):
    """Create the prompt with one example"""
    ex_form, ex_context = EXAMPLE_INPUT.split('\t')
    
    messages = [
        {
            "role": "user",
            "content": f"Based on this example:\nInput: {ex_form}\nContext: {ex_context}\nPrediction: {EXAMPLE_PREDICTION}\n\n{PROMPT}{form}\t{context}\n\nAnswer (lemma and tags only, no explanation):"
        }
    ]
    
    return messages


def query_model(model_name, form, context, temperature=0.0):
    """Query the LLM model"""
    payload = {
        "model": model_name,
        "messages": create_prompt(form, context),
        "temperature": temperature,
        "max_tokens":2000
    }
    
    try:
        response = requests.post(
            url=URL,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=60  # Increased timeout for reasoning models
        )
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0]['message']
            # Get content, which contains the actual answer
            prediction = message.get('content', '').strip()
            if prediction:
                return prediction
            else:
                print(f"\nEmpty content for model {model_name}")
                print(f"Full message: {json.dumps(message, indent=2)}")
                return None
        else:
            print(f"\nUnexpected response format for model {model_name}:")
            print(f"Response: {json.dumps(result, indent=2)}")
            return None
    except requests.exceptions.Timeout:
        print(f"\nRequest timeout for model {model_name} after 60 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nAPI request failed for model {model_name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"\nUnexpected error for model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_prediction(prediction):
    """Parse the model prediction into lemma and tags"""
    if not prediction:
        return None, None
    
    # Try to extract lemma and tags from the prediction
    parts = prediction.split('\t')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    
    # If no tab, try space-separated
    parts = prediction.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    
    return None, None


def evaluate_model(model_name, test_file, output_dir):
    """Evaluate a model on dev and test sets with inverse task metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_short_name = model_name.split('/')[-1]
    
    # Summary file for results
    summary_file = os.path.join(output_dir, f"{model_short_name}_results_summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as summary_f:
        summary_f.write(f"Results for model: {model_name}\n")
        summary_f.write(f"{'='*60}\n\n")
        
        for split_name, filepath in [('test', test_file)]:
            print(f"\nProcessing {split_name} set: {filepath}")
            summary_f.write(f"\n{split_name.upper()} SET\n")
            summary_f.write(f"{'-'*60}\n")
            
            if not os.path.exists(filepath):
                msg = f"File not found: {filepath}"
                print(msg)
                summary_f.write(msg + "\n")
                continue
            
            # Load data
            data = load_test_data(filepath)
            print(f"Loaded {len(data)} examples")
            
            # Output file for predictions
            pred_file = os.path.join(output_dir, f"{model_short_name}_{split_name}_predictions.txt")
            
            predictions = []
            gold_targets = []
            
            with open(pred_file, 'w', encoding='utf-8') as pred_f:
                for item in tqdm(data, desc=f"  {split_name.upper()}", unit="example"):
                    # Query model
                    prediction = query_model(model_name, item['form'], item['context'])
                    pred_lemma, pred_tags = parse_prediction(prediction)
                    
                    # Write to file: input | prediction | gold
                    input_str = f"form: {item['form']} | context: {item['context']}"
                    gold = f"{item['lemma']} {item['tags']}"
                    pred = prediction if prediction else "ERROR"
                    
                    pred_f.write(f"{input_str}\t{pred}\t{gold}\n")
                    pred_f.flush()
                    
                    # Store for evaluation
                    predictions.append(pred)
                    gold_targets.append(gold)
                    
                    # Rate limiting (adjust as needed)
                    time.sleep(0.5)
            
            print(f"  Predictions saved to: {pred_file}")
            
            # Compute inverse task metrics
            print(f"  Computing inverse task metrics...")
            metrics = evaluate_inverse(predictions, gold_targets, summary_f)
            
            # Print metrics to console
            if metrics:
                print(f"  {split_name.upper()} - Lemma accuracy: {metrics['lemma_accuracy']:.4f}")
                print(f"  {split_name.upper()} - Lemma mean Levenshtein: {metrics['mean_levenshtein']:.4f}")
                print(f"  {split_name.upper()} - MSD accuracy: {metrics['msd_accuracy']:.4f}")
                print(f"  {split_name.upper()} - MSD F1: {metrics['msd_f1']:.4f} (P={metrics['msd_precision']:.4f}, R={metrics['msd_recall']:.4f})")
    
    print(f"\nResults summary saved to: {summary_file}")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "ud"
    output_dir = script_dir / "output"
    
    test_file = data_dir / "pt_verbs_context.tst"
    
    # Check API key
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Evaluate both models
    #for model in [LLAMA_MODEL, QWEN_MODEL]:
    for model in [QWEN_MODEL]:
        try:
            evaluate_model(model, str(test_file), str(output_dir))
        except Exception as e:
            print(f"\nError evaluating {model}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()