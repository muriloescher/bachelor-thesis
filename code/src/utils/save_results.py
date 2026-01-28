"""
Utility functions for saving experiment results in JSON and CSV formats.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def save_results_json(
    output_dir: str,
    model_type: str,
    model_name: str,
    language: str,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    test_examples: Optional[int] = None,
    split: str = "test"
) -> Path:
    """
    Save results to JSON file.
    
    Args:
        output_dir: Directory to save the JSON file
        model_type: Type of model (e.g., 'llm', 'byt5', 'nonneural')
        model_name: Name or identifier of the model
        language: Language code (e.g., 'por', 'kat', 'eng')
        metrics: Dictionary containing metrics (structure depends on model_type)
        config: Optional configuration dictionary
        test_examples: Optional number of test examples
        split: Data split ('test', 'dev', etc.)
    
    Returns:
        Path to the saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "model_type": model_type,
        "model_name": model_name,
        "language": language,
        "split": split,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    if config is not None:
        result_data["configuration"] = config
    
    if test_examples is not None:
        result_data["test_examples"] = test_examples
    
    json_file = output_dir / "results_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    
    return json_file


def append_to_central_csv(
    csv_path: str,
    model_type: str,
    model_name: str,
    language: str,
    split: str,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    test_examples: Optional[int] = None
) -> None:
    """
    Append results to the central CSV file.
    
    Args:
        csv_path: Path to the central CSV file
        model_type: Type of model
        model_name: Name or identifier of the model
        language: Language code
        split: Data split
        metrics: Dictionary containing metrics
        config: Optional configuration dictionary
        test_examples: Optional number of test examples
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV row
    row = {
        'model_type': model_type,
        'model_name': model_name,
        'language': language,
        'split': split,
        'test_examples': test_examples if test_examples is not None else '',
        'lemma_accuracy': '',
        'lemma_mean_levenshtein': '',
        'lemma_correct': '',
        'lemma_total': '',
        'msd_accuracy': '',
        'msd_f1': '',
        'msd_precision': '',
        'msd_recall': '',
        'msd_correct': '',
        'msd_total': '',
        'use_context': '',
        'epochs': '',
        'batch_size': '',
        'learning_rate': '',
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract metrics based on structure
    if 'lemma' in metrics:
        row['lemma_accuracy'] = metrics['lemma'].get('accuracy', '')
        row['lemma_mean_levenshtein'] = metrics['lemma'].get('mean_levenshtein', '')
        row['lemma_correct'] = metrics['lemma'].get('correct', '')
        row['lemma_total'] = metrics['lemma'].get('total', '')
    else:
        # Direct lemma metrics
        row['lemma_accuracy'] = metrics.get('lemma_accuracy', '')
        row['lemma_mean_levenshtein'] = metrics.get('mean_levenshtein', '')
    
    if 'msd' in metrics:
        row['msd_accuracy'] = metrics['msd'].get('accuracy', '')
        row['msd_f1'] = metrics['msd'].get('f1', '')
        row['msd_precision'] = metrics['msd'].get('precision', '')
        row['msd_recall'] = metrics['msd'].get('recall', '')
        row['msd_correct'] = metrics['msd'].get('correct', '')
        row['msd_total'] = metrics['msd'].get('total', '')
    else:
        # Direct MSD metrics
        row['msd_accuracy'] = metrics.get('msd_accuracy', '')
        row['msd_f1'] = metrics.get('msd_f1', '')
    
    # Add config if provided
    if config:
        row['use_context'] = config.get('use_context', '')
        row['epochs'] = config.get('epochs', '')
        row['batch_size'] = config.get('batch_size', '')
        row['learning_rate'] = config.get('learning_rate', '')
    
    # Check if file exists
    file_exists = csv_path.exists()
    
    # Write to CSV
    fieldnames = [
        'model_type', 'model_name', 'language', 'split', 'test_examples',
        'lemma_accuracy', 'lemma_mean_levenshtein', 'lemma_correct', 'lemma_total',
        'msd_accuracy', 'msd_f1', 'msd_precision', 'msd_recall', 'msd_correct', 'msd_total',
        'use_context', 'epochs', 'batch_size', 'learning_rate', 'timestamp'
    ]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def save_llm_results(
    output_dir: str,
    model_name: str,
    language: str,
    metrics: Dict[str, Any],
    test_examples: int,
    central_csv_path: str = None
) -> None:
    """
    Save LLM results in both JSON and CSV formats.
    
    Args:
        output_dir: Directory to save results
        model_name: LLM model name
        language: Language code
        metrics: Dictionary with 'lemma_accuracy', 'mean_levenshtein', 'msd_accuracy', 'msd_f1', etc.
        test_examples: Number of test examples
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    """
    if central_csv_path is None:
        central_csv_path = Path(output_dir).parent / "all_results.csv"
    
    # Structure metrics for JSON
    json_metrics = {
        "lemma": {
            "accuracy": metrics.get('lemma_accuracy', 0.0),
            "mean_levenshtein": metrics.get('mean_levenshtein', 0.0),
            "correct": metrics.get('lemma_correct', 0),
            "total": metrics.get('total', 0)
        },
        "msd": {
            "accuracy": metrics.get('msd_accuracy', 0.0),
            "f1": metrics.get('msd_f1', 0.0),
            "precision": metrics.get('msd_precision', 0.0),
            "recall": metrics.get('msd_recall', 0.0),
            "correct": metrics.get('msd_correct', 0),
            "total": metrics.get('total', 0)
        }
    }
    
    # Save JSON
    save_results_json(
        output_dir=output_dir,
        model_type="llm",
        model_name=model_name,
        language=language,
        metrics=json_metrics,
        test_examples=test_examples,
        split="test"
    )
    
    # Append to CSV
    append_to_central_csv(
        csv_path=central_csv_path,
        model_type="llm",
        model_name=model_name,
        language=language,
        split="test",
        metrics=json_metrics,
        test_examples=test_examples
    )


def save_byt5_results(
    output_dir: str,
    model_name: str,
    language: str,
    dev_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    config: Dict[str, Any],
    central_csv_path: str = None
) -> None:
    """
    Save ByT5 results in both JSON and CSV formats.
    
    Args:
        output_dir: Directory to save results
        model_name: Model name or checkpoint path
        language: Language code
        dev_metrics: Dev set metrics
        test_metrics: Test set metrics
        config: Configuration dictionary (use_context, epochs, batch_size, learning_rate, etc.)
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    """
    if central_csv_path is None:
        central_csv_path = Path(output_dir).parent / "all_results.csv"
    
    # Structure metrics for JSON
    json_metrics = {
        "dev": {
            "lemma": {
                "accuracy": dev_metrics.get('lemma_accuracy', 0.0),
                "mean_levenshtein": dev_metrics.get('mean_levenshtein', 0.0)
            },
            "msd": {
                "accuracy": dev_metrics.get('msd_accuracy', 0.0),
                "f1": dev_metrics.get('msd_f1', 0.0)
            }
        },
        "test": {
            "lemma": {
                "accuracy": test_metrics.get('lemma_accuracy', 0.0),
                "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0)
            },
            "msd": {
                "accuracy": test_metrics.get('msd_accuracy', 0.0),
                "f1": test_metrics.get('msd_f1', 0.0)
            }
        }
    }
    
    # Save JSON
    save_results_json(
        output_dir=output_dir,
        model_type="byt5",
        model_name=model_name,
        language=language,
        metrics=json_metrics,
        config=config
    )
    
    # Append to CSV (one row for dev, one for test)
    for split, split_metrics in [("dev", json_metrics["dev"]), ("test", json_metrics["test"])]:
        append_to_central_csv(
            csv_path=central_csv_path,
            model_type="byt5",
            model_name=model_name,
            language=language,
            split=split,
            metrics=split_metrics,
            config=config
        )
