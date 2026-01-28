#!/usr/bin/env python3
"""
Main runner script for morphological inflection/analysis experiments.

This script provides a unified interface to train and evaluate all models
(non-neural, neural baseline, ByT5 variants, LLM) on multiple languages.

Usage:
    python run.py --model byt5_bidirectional --language por --train
    python run.py --model byt5_bidirectional --language por --predict
    python run.py --model byt5_bidirectional --language por,eng,ita --train --predict
    python run.py --model llm --language por --predict --test-only
"""

import argparse
import sys
import yaml
from pathlib import Path

from src.utils import get_config_path, get_project_root


def load_config(config_type, name):
    """Load a YAML configuration file."""
    config_path = get_config_path(config_type, name)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_byt5(model_config, lang_config, args):
    """Run ByT5 model training/prediction."""
    from src.models.byt5 import ByT5Model
    
    model = ByT5Model(model_config, lang_config)
    
    if args.train:
        model.train()
    
    if args.predict:
        inverse_only = model_config.get('prediction', {}).get('inverse_only', False)
        if hasattr(args, 'inverse_only') and args.inverse_only:
            inverse_only = True
        model.predict(checkpoint=args.checkpoint, inverse_only=inverse_only)


def run_llm(model_config, lang_config, args):
    """Run LLM model prediction."""
    from src.models.llm import LLMModel
    
    model = LLMModel(model_config, lang_config)
    
    if args.train:
        model.train()
    
    if args.predict:
        model.predict()


def run_nonneural(model_config, lang_config, args):
    """Run non-neural baseline."""
    from src.models.nonneural import NonNeuralModel
    
    model = NonNeuralModel(model_config, lang_config)
    
    if args.train:
        model.train()
    
    if args.predict:
        model.predict(is_test=True)


def run_neural(model_config, lang_config, args):
    """Run neural baseline (transducer)."""
    from src.models.neural_baseline import NeuralBaselineModel
    
    model = NeuralBaselineModel(model_config, lang_config)
    
    if args.train:
        model.train()
    
    if args.predict:
        model.predict(checkpoint=args.checkpoint)


MODEL_RUNNERS = {
    'byt5': run_byt5,
    'byt5_context': run_byt5,
    'llm': run_llm,
    'nonneural': run_nonneural,
    'neural_baseline': run_neural
}


def main():
    parser = argparse.ArgumentParser(
        description='Unified runner for morphological inflection/analysis experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ByT5 bidirectional model for Portuguese
  python run.py --model byt5_bidirectional --language por --train
  
  # Predict with trained model
  python run.py --model byt5_bidirectional --language por --predict
  
  # Train and predict for multiple languages
  python run.py --model byt5_forward --language por,eng,ita --train --predict
  
  # Predict with specific checkpoint
  python run.py --model byt5_inverse --language por --predict --checkpoint 3750
  
  # Bidirectional model, inverse direction only
  python run.py --model byt5_bidirectional --language por --predict --inverse-only
  
  # LLM evaluation
  python run.py --model llm --language por --predict
        """
    )
    
    parser.add_argument('--model',
                       help='Model config name (e.g., byt5_forward, byt5_inverse, byt5_bidirectional, byt5_context, llm, nonneural, neural_baseline)')
    parser.add_argument('--language',
                       help='Language code(s), comma-separated (e.g., por or por,eng,ita)')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions with trained model')
    parser.add_argument('--checkpoint', type=int,
                       help='Specific checkpoint number to load for prediction')
    parser.add_argument('--inverse-only', action='store_true',
                       help='For bidirectional models: predict only inverse direction')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model configurations')
    parser.add_argument('--list-languages', action='store_true',
                       help='List available language configurations')
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_models:
        models_dir = get_project_root() / 'configs' / 'models'
        print("\nAvailable model configurations:")
        for config_file in sorted(models_dir.glob('*.yaml')):
            print(f"  - {config_file.stem}")
        return 0
    
    if args.list_languages:
        langs_dir = get_project_root() / 'configs' / 'languages'
        print("\nAvailable language configurations:")
        for config_file in sorted(langs_dir.glob('*.yaml')):
            lang_cfg = yaml.safe_load(config_file.read_text())
            code = lang_cfg['language']['code']
            name = lang_cfg['language']['name']
            print(f"  - {code:6s} ({name})")
        return 0
    
    # Validate required arguments for normal operation
    if not args.model:
        parser.error("--model is required")
    if not args.language:
        parser.error("--language/--lang is required")
    
    # Validate that at least one action is specified
    if not (args.train or args.predict):
        parser.error("At least one of --train or --predict must be specified")
    
    # Load model configuration
    try:
        model_config = load_config('models', args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nRun 'python run.py --list-models' to see available models.")
        return 1
    
    # Get model type
    model_type = model_config.get('model_type')
    if not model_type:
        print(f"Error: Model config {args.model} missing 'model_type' field")
        return 1
    
    # Get model runner
    runner = MODEL_RUNNERS.get(model_type)
    if not runner:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Supported types: {', '.join(MODEL_RUNNERS.keys())}")
        return 1
    
    # Parse languages
    languages = [lang.strip() for lang in args.language.split(',')]
    
    # Run for each language
    for lang_code in languages:
        try:
            lang_config = load_config('languages', lang_code)
        except FileNotFoundError:
            print(f"\nWarning: No config found for language '{lang_code}', skipping...")
            print(f"Run 'python run.py --list-languages' to see available languages.")
            continue
        
        try:
            runner(model_config, lang_config, args)
        except Exception as e:
            print(f"\nError processing {lang_code} with {args.model}:")
            print(f"  {type(e).__name__}: {e}")
            if hasattr(args, 'debug') and args.debug:
                raise
            continue
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
