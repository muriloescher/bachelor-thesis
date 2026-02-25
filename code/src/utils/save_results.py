"""
Utility functions for saving experiment results in JSON and CSV formats.

This module provides a unified interface for saving results from all model types:
- Non-neural baseline
- Neural baseline (transducer)
- ByT5 models (forward, inverse, bidirectional, context)
- LLM models

Results are saved in two formats:
1. JSON: One file per language in the model's output directory (results_{lang}.json)
2. CSV: Central file at results/all_results.csv with all experiments
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


def save_results_json(
    output_dir: str,
    model_type: str,
    model_name: str,
    language: str,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    test_examples: Optional[int] = None,
    direction: Optional[str] = None
) -> Path:
    """
    Save results to a language-specific JSON file.
    
    This replaces any existing results for the same language, allowing for
    clean re-runs without accumulating stale data.
    
    Args:
        output_dir: Directory to save the JSON file
        model_type: Type of model (e.g., 'llm', 'byt5', 'nonneural', 'neural')
        model_name: Name or identifier of the model
        language: Language code (e.g., 'por', 'kat', 'eng')
        metrics: Dictionary containing metrics (see structure below)
        config: Optional configuration dictionary
        test_examples: Optional number of test examples
        direction: Optional direction for bidirectional models ('forward', 'inverse', or None)
    
    Metrics structure:
        {
            "dev": {  # Optional, only if dev set is evaluated
                "lemma": {"accuracy": float, "mean_levenshtein": float, "correct": int, "total": int},
                "msd": {"accuracy": float, "f1": float, "precision": float, "recall": float, "correct": int, "total": int}
            },
            "test": {  # Required
                "lemma": {...},
                "msd": {...}
            }
        }
        
        For bidirectional models:
        {
            "forward": {"dev": {...}, "test": {...}},
            "inverse": {"dev": {...}, "test": {...}}
        }
    
    Returns:
        Path to the saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use language-specific filename
    json_file = output_dir / f"results_{language}.json"
    
    result_data = {
        "model_type": model_type,
        "model_name": model_name,
        "language": language,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    if direction is not None:
        result_data["direction"] = direction
    
    if config is not None:
        result_data["configuration"] = config
    
    if test_examples is not None:
        result_data["test_examples"] = test_examples
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return json_file


def append_to_central_csv(
    csv_path: str,
    model_type: str,
    model_name: str,
    language: str,
    split: str,
    metrics: Dict[str, Any],
    direction: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    test_examples: Optional[int] = None
) -> None:
    """
    Append a single row to the central CSV file.
    
    Args:
        csv_path: Path to the central CSV file
        model_type: Type of model (e.g., 'llm', 'byt5', 'nonneural', 'neural')
        model_name: Name or identifier of the model
        language: Language code
        split: Data split ('dev' or 'test')
        metrics: Dictionary containing metrics for this split
        direction: Optional direction ('forward', 'inverse', or None for uni-directional)
        config: Optional configuration dictionary
        test_examples: Optional number of test examples
    
    Metrics structure for this split:
        {
            "lemma": {"accuracy": float, "mean_levenshtein": float, "correct": int, "total": int},
            "msd": {"accuracy": float, "f1": float, "precision": float, "recall": float, "correct": int, "total": int}
        }
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV row with default values
    row = {
        'model_type': model_type,
        'model_name': model_name,
        'language': language,
        'direction': direction if direction else '',
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
    
    # Extract lemma metrics
    if 'lemma' in metrics:
        row['lemma_accuracy'] = metrics['lemma'].get('accuracy', '')
        row['lemma_mean_levenshtein'] = metrics['lemma'].get('mean_levenshtein', '')
        row['lemma_correct'] = metrics['lemma'].get('correct', '')
        row['lemma_total'] = metrics['lemma'].get('total', '')
    
    # Extract MSD metrics
    if 'msd' in metrics:
        row['msd_accuracy'] = metrics['msd'].get('accuracy', '')
        row['msd_f1'] = metrics['msd'].get('f1', '')
        row['msd_precision'] = metrics['msd'].get('precision', '')
        row['msd_recall'] = metrics['msd'].get('recall', '')
        row['msd_correct'] = metrics['msd'].get('correct', '')
        row['msd_total'] = metrics['msd'].get('total', '')
    
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
        'model_type', 'model_name', 'language', 'direction', 'split', 'test_examples',
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
) -> Path:
    """
    Save LLM results in both JSON and CSV formats.
    
    Args:
        output_dir: Directory to save results
        model_name: LLM model name
        language: Language code
        metrics: Dictionary with 'lemma_accuracy', 'mean_levenshtein', 'msd_accuracy', 'msd_f1', etc.
        test_examples: Number of test examples
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    
    Returns:
        Path to the saved JSON file
    """
    if central_csv_path is None:
        from ..utils.paths import get_project_root
        central_csv_path = get_project_root() / "results" / "all_results.csv"
    
    # Structure metrics for JSON (test split only for LLMs)
    json_metrics = {
        "test": {
            "lemma": {
                "accuracy": metrics.get('lemma_accuracy', 0.0),
                "mean_levenshtein": metrics.get('mean_levenshtein', 0.0),
                "correct": metrics.get('lemma_correct', 0),
                "total": metrics.get('total', test_examples)
            },
            "msd": {
                "accuracy": metrics.get('msd_accuracy', 0.0),
                "f1": metrics.get('msd_f1', 0.0),
                "precision": metrics.get('msd_precision', 0.0),
                "recall": metrics.get('msd_recall', 0.0),
                "correct": metrics.get('msd_correct', 0),
                "total": metrics.get('total', test_examples)
            }
        }
    }
    
    # Save JSON (language-specific file)
    json_path = save_results_json(
        output_dir=output_dir,
        model_type="llm",
        model_name=model_name,
        language=language,
        metrics=json_metrics,
        test_examples=test_examples
    )
    
    # Append to CSV
    append_to_central_csv(
        csv_path=central_csv_path,
        model_type="llm",
        model_name=model_name,
        language=language,
        split="test",
        metrics=json_metrics["test"],
        test_examples=test_examples
    )
    
    return json_path


def save_byt5_results(
    output_dir: str,
    model_name: str,
    language: str,
    direction: str,
    test_metrics: Dict[str, Any],
    config: Dict[str, Any],
    central_csv_path: str = None
) -> Path:
    """
    Save ByT5 results in both JSON and CSV formats.
    
    Handles uni-directional (forward/inverse) and bidirectional models properly.
    For bidirectional models, call this function once per direction.
    Only test metrics are saved (no dev metrics).
    
    Args:
        output_dir: Directory to save results
        model_name: Model name or checkpoint path
        language: Language code
        direction: 'forward', 'inverse', or None for uni-directional models
        test_metrics: Test set metrics (required)
        config: Configuration dictionary (use_context, epochs, batch_size, learning_rate, etc.)
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    
    Returns:
        Path to the saved JSON file
    """
    if central_csv_path is None:
        from ..utils.paths import get_project_root
        central_csv_path = get_project_root() / "results" / "all_results.csv"
    
    # Read existing JSON if it exists (for bidirectional models)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    json_file = output_dir_path / f"results_{language}.json"
    
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            json_metrics = existing_data.get('metrics', {})
    else:
        json_metrics = {}
    
    # Structure metrics for this direction (test only)
    # Handle both forward (accuracy only) and inverse (lemma + MSD) metrics
    direction_metrics = {}
    
    if 'accuracy' in test_metrics and 'lemma_accuracy' not in test_metrics:
        # Forward direction - only accuracy
        direction_metrics["test"] = {
            "accuracy": test_metrics.get('accuracy', 0.0),
            "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
            "correct": test_metrics.get('correct', 0),
            "total": test_metrics.get('total', 0)
        }
    else:
        # Inverse direction - lemma + MSD analysis
        direction_metrics["test"] = {
            "lemma": {
                "accuracy": test_metrics.get('lemma_accuracy', 0.0),
                "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
                "correct": test_metrics.get('lemma_correct', 0),
                "total": test_metrics.get('total', 0)
            },
            "msd": {
                "accuracy": test_metrics.get('msd_accuracy', 0.0),
                "f1": test_metrics.get('msd_f1', 0.0),
                "precision": test_metrics.get('msd_precision', 0.0),
                "recall": test_metrics.get('msd_recall', 0.0),
                "correct": test_metrics.get('msd_correct', 0),
                "total": test_metrics.get('total', 0)
            }
        }
    
    
    # Add to JSON metrics
    if direction:
        # Bidirectional model - add this direction's metrics
        json_metrics[direction] = direction_metrics
    else:
        # Uni-directional model - metrics at top level
        json_metrics = direction_metrics
    
    # Save JSON (language-specific file, updated with new/additional direction)
    json_path = save_results_json(
        output_dir=output_dir,
        model_type="byt5",
        model_name=model_name,
        language=language,
        metrics=json_metrics,
        config=config,
        direction=direction
    )
    
    # Append to CSV (test only)
    # Format metrics for CSV based on direction
    if 'accuracy' in direction_metrics["test"] and 'lemma' not in direction_metrics["test"]:
        # Forward direction - wrap in lemma structure for CSV
        csv_metrics = {
            "lemma": {
                "accuracy": direction_metrics["test"].get('accuracy', 0.0),
                "mean_levenshtein": direction_metrics["test"].get('mean_levenshtein', 0.0),
                "correct": direction_metrics["test"].get('correct', 0),
                "total": direction_metrics["test"].get('total', 0)
            }
        }
    else:
        # Inverse direction - already in correct format
        csv_metrics = direction_metrics["test"]
    
    append_to_central_csv(
        csv_path=central_csv_path,
        model_type="byt5",
        model_name=model_name,
        language=language,
        split="test",
        metrics=csv_metrics,
        direction=direction,
        config=config
    )
    
    return json_path


def save_nonneural_results(
    output_dir: str,
    model_name: str,
    language: str,
    test_metrics: Dict[str, Any],
    central_csv_path: str = None
) -> Path:
    """
    Save non-neural baseline results in both JSON and CSV formats.
    
    Non-neural baseline only does forward direction (inflection), so only
    accuracy metrics are saved, not lemma/MSD analysis metrics.
    
    Args:
        output_dir: Directory to save results
        model_name: Model name (e.g., 'nonneural-baseline')
        language: Language code
        test_metrics: Test set metrics with 'accuracy', 'correct', 'total'
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    
    Returns:
        Path to the saved JSON file
    """
    if central_csv_path is None:
        from ..utils.paths import get_project_root
        central_csv_path = get_project_root() / "results" / "all_results.csv"
    
    # Structure metrics for JSON (test split only, forward direction only)
    json_metrics = {
        "test": {
            "accuracy": test_metrics.get('accuracy', 0.0),
            "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
            "correct": test_metrics.get('correct', 0),
            "total": test_metrics.get('total', 0)
        }
    }
    
    # Save JSON (language-specific file)
    json_path = save_results_json(
        output_dir=output_dir,
        model_type="nonneural",
        model_name=model_name,
        language=language,
        metrics=json_metrics
    )
    
    # Prepare metrics for CSV (forward-only format)
    csv_metrics = {
        "lemma": {
            "accuracy": test_metrics.get('accuracy', 0.0),
            "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
            "correct": test_metrics.get('correct', 0),
            "total": test_metrics.get('total', 0)
        }
    }
    
    # Append to CSV
    append_to_central_csv(
        csv_path=central_csv_path,
        model_type="nonneural",
        model_name=model_name,
        language=language,
        split="test",
        metrics=csv_metrics,
        direction="forward",
        test_examples=test_metrics.get('total', 0)
    )
    
    return json_path


def save_neural_results(
    output_dir: str,
    model_name: str,
    language: str,
    test_metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    central_csv_path: str = None
) -> Path:
    """
    Save neural baseline (transducer) results in both JSON and CSV formats.
    
    Neural baseline only does forward direction (inflection), so only
    accuracy metrics are saved, not lemma/MSD analysis metrics.
    
    Args:
        output_dir: Directory to save results
        model_name: Model name (e.g., 'neural-transducer')
        language: Language code
        test_metrics: Test set metrics with 'accuracy', 'correct', 'total'
        config: Optional configuration dictionary (epochs, etc.)
        central_csv_path: Path to central CSV file (defaults to results/all_results.csv)
    
    Returns:
        Path to the saved JSON file
    """
    if central_csv_path is None:
        from ..utils.paths import get_project_root
        central_csv_path = get_project_root() / "results" / "all_results.csv"
    
    # Structure metrics for JSON (test split only, forward direction only)
    json_metrics = {
        "test": {
            "accuracy": test_metrics.get('accuracy', 0.0),
            "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
            "correct": test_metrics.get('correct', 0),
            "total": test_metrics.get('total', 0)
        }
    }
    
    # Save JSON (language-specific file)
    json_path = save_results_json(
        output_dir=output_dir,
        model_type="neural",
        model_name=model_name,
        language=language,
        metrics=json_metrics,
        config=config
    )
    
    # Prepare metrics for CSV (forward-only format)
    csv_metrics = {
        "lemma": {
            "accuracy": test_metrics.get('accuracy', 0.0),
            "mean_levenshtein": test_metrics.get('mean_levenshtein', 0.0),
            "correct": test_metrics.get('correct', 0),
            "total": test_metrics.get('total', 0)
        }
    }
    
    # Append to CSV
    append_to_central_csv(
        csv_path=central_csv_path,
        model_type="neural",
        model_name=model_name,
        language=language,
        split="test",
        metrics=csv_metrics,
        direction="forward",
        config=config,
        test_examples=test_metrics.get('total', 0)
    )
    
    return json_path
