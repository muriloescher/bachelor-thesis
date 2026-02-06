#!/usr/bin/env python3
"""
Prepare UniMorph data with consistent train/dev/test splits.

This script takes original UniMorph datasets and creates randomized
10k/1k/1k train/dev/test splits for consistency across all languages.

Usage:
    python prepare_unimorph_data.py <language_code> <input_file>
    
Example:
    python prepare_unimorph_data.py por unimorph-data/por
    python prepare_unimorph_data.py azg unimorph-data/azg
"""

import argparse
import random
from pathlib import Path


def is_verb(features):
    """
    Check if a UniMorph feature string indicates a verb.
    
    Args:
        features: UniMorph feature string (e.g., "V;IND;PRS;3;SG")
        
    Returns:
        True if the word is a verb (contains V or V.PTCP, etc.)
    """
    return features.startswith('V') or features.startswith('V.')


def load_unimorph_data(input_file):
    """
    Load UniMorph data from file, separating verbs from other word types.
    
    Args:
        input_file: Path to UniMorph data file
        
    Returns:
        Tuple of (verbs, non_verbs) where each is a list of tuples (lemma, features, form)
    """
    verbs = []
    non_verbs = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                lemma, form, features = parts[0], parts[1], parts[2]
                entry = (lemma, features, form)
                
                if is_verb(features):
                    verbs.append(entry)
                else:
                    non_verbs.append(entry)
    
    return verbs, non_verbs


def group_by_lemma(data, max_forms_per_lemma=None, seed=42):
    """
    Group data by lemma to ensure no lemma appears in multiple splits.
    Optionally limits the number of forms per lemma to increase lemma diversity.
    
    Args:
        data: List of tuples (lemma, features, form)
        max_forms_per_lemma: Maximum number of forms to keep per lemma (None = keep all)
        seed: Random seed for sampling forms
        
    Returns:
        Dictionary mapping lemma -> list of (lemma, features, form) tuples
    """
    random.seed(seed)
    lemma_groups = {}
    
    for entry in data:
        lemma = entry[0]
        if lemma not in lemma_groups:
            lemma_groups[lemma] = []
        lemma_groups[lemma].append(entry)
    
    # Limit forms per lemma if specified
    if max_forms_per_lemma is not None:
        for lemma in lemma_groups:
            if len(lemma_groups[lemma]) > max_forms_per_lemma:
                lemma_groups[lemma] = random.sample(lemma_groups[lemma], max_forms_per_lemma)
    
    return lemma_groups


def create_splits(verbs, non_verbs, train_size=10000, dev_size=1000, test_size=1000, seed=42, max_forms_per_lemma=None):
    """
    Create lemma-based train/dev/test splits, prioritizing verbs.
    
    IMPORTANT: Splits by lemma (not individual examples) to prevent data leakage.
    All forms of a lemma go into the same split.
    
    Args:
        verbs: List of verb data tuples
        non_verbs: List of non-verb data tuples
        train_size: Target training set size
        dev_size: Target dev set size
        test_size: Target test set size
        seed: Random seed for reproducibility
        max_forms_per_lemma: Maximum forms per lemma (None = unlimited, e.g., 15 or 20)
        
    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Group by lemma and optionally limit forms
    verb_groups = group_by_lemma(verbs, max_forms_per_lemma, seed)
    nonverb_groups = group_by_lemma(non_verbs, max_forms_per_lemma, seed)
    
    total_verb_forms = sum(len(forms) for forms in verb_groups.values())
    total_nonverb_forms = sum(len(forms) for forms in nonverb_groups.values())
    
    print(f"  Verb lemmas: {len(verb_groups)} (total {total_verb_forms} forms, avg {total_verb_forms/len(verb_groups) if verb_groups else 0:.1f} forms/lemma)")
    print(f"  Non-verb lemmas: {len(nonverb_groups)} (total {total_nonverb_forms} forms, avg {total_nonverb_forms/len(nonverb_groups) if nonverb_groups else 0:.1f} forms/lemma)")
    if max_forms_per_lemma:
        print(f"  Max forms per lemma: {max_forms_per_lemma}")
    
    # Check if we have enough data to reach target sizes
    total_needed = train_size + dev_size + test_size
    total_available = total_verb_forms + total_nonverb_forms
    
    if total_available < total_needed:
        print(f"\n  ⚠️  WARNING: Only {total_available} forms available, but need {total_needed}")
        print(f"  Consider:")
        if max_forms_per_lemma:
            print(f"    - Removing or increasing --max-forms (currently {max_forms_per_lemma})")
        print(f"    - Reducing split sizes")
        print(f"  Proceeding with available data...")
    
    # Get list of lemmas and shuffle them
    verb_lemmas = list(verb_groups.keys())
    random.shuffle(verb_lemmas)
    
    nonverb_lemmas = list(nonverb_groups.keys())
    random.shuffle(nonverb_lemmas)
    
    # Strategy: Split lemmas to get approximately the target number of examples
    # Start by allocating lemmas to splits, trying to reach target sizes
    
    train_verb_lemmas = []
    dev_verb_lemmas = []
    test_verb_lemmas = []
    
    train_count = 0
    dev_count = 0
    test_count = 0
    
    # Distribute verb lemmas across splits proportionally
    # Calculate target ratios
    total_target = train_size + dev_size + test_size
    train_ratio = train_size / total_target
    dev_ratio = dev_size / total_target
    
    for lemma in verb_lemmas:
        lemma_size = len(verb_groups[lemma])
        
        # Calculate how far each split is from its target ratio
        total_assigned = train_count + dev_count + test_count
        if total_assigned > 0:
            train_current_ratio = train_count / total_assigned
            dev_current_ratio = dev_count / total_assigned
            test_current_ratio = test_count / total_assigned
        else:
            train_current_ratio = dev_current_ratio = test_current_ratio = 0
        
        # Assign to the split that's furthest below its target ratio
        train_deficit = train_ratio - train_current_ratio
        dev_deficit = dev_ratio - dev_current_ratio
        test_deficit = (1 - train_ratio - dev_ratio) - test_current_ratio
        
        if train_deficit >= dev_deficit and train_deficit >= test_deficit:
            train_verb_lemmas.append(lemma)
            train_count += lemma_size
        elif dev_deficit >= test_deficit:
            dev_verb_lemmas.append(lemma)
            dev_count += lemma_size
        else:
            test_verb_lemmas.append(lemma)
            test_count += lemma_size
    
    # If we still need more data, add non-verb lemmas (using same proportional strategy)
    train_nonverb_lemmas = []
    dev_nonverb_lemmas = []
    test_nonverb_lemmas = []
    
    for lemma in nonverb_lemmas:
        lemma_size = len(nonverb_groups[lemma])
        
        # Calculate how far each split is from its target ratio
        total_assigned = train_count + dev_count + test_count
        if total_assigned > 0:
            train_current_ratio = train_count / total_assigned
            dev_current_ratio = dev_count / total_assigned
            test_current_ratio = test_count / total_assigned
        else:
            train_current_ratio = dev_current_ratio = test_current_ratio = 0
        
        # Assign to the split that's furthest below its target ratio
        train_deficit = train_ratio - train_current_ratio
        dev_deficit = dev_ratio - dev_current_ratio
        test_deficit = (1 - train_ratio - dev_ratio) - test_current_ratio
        
        if train_deficit >= dev_deficit and train_deficit >= test_deficit:
            train_nonverb_lemmas.append(lemma)
            train_count += lemma_size
        elif dev_deficit >= test_deficit:
            dev_nonverb_lemmas.append(lemma)
            dev_count += lemma_size
        else:
            test_nonverb_lemmas.append(lemma)
            test_count += lemma_size
        
        # Stop if all splits have enough
        if train_count >= train_size and dev_count >= dev_size and test_count >= test_size:
            break
    
    # Collect all examples from assigned lemmas
    train_data = []
    for lemma in train_verb_lemmas:
        train_data.extend(verb_groups[lemma])
    for lemma in train_nonverb_lemmas:
        train_data.extend(nonverb_groups[lemma])
    random.shuffle(train_data)
    
    dev_data = []
    for lemma in dev_verb_lemmas:
        dev_data.extend(verb_groups[lemma])
    for lemma in dev_nonverb_lemmas:
        dev_data.extend(nonverb_groups[lemma])
    random.shuffle(dev_data)
    
    test_data = []
    for lemma in test_verb_lemmas:
        test_data.extend(verb_groups[lemma])
    for lemma in test_nonverb_lemmas:
        test_data.extend(nonverb_groups[lemma])
    random.shuffle(test_data)
    
    # Trim to exact sizes (sample randomly if over, keep all if under)
    if len(train_data) > train_size:
        train_data = random.sample(train_data, train_size)
    if len(dev_data) > dev_size:
        dev_data = random.sample(dev_data, dev_size)
    if len(test_data) > test_size:
        test_data = random.sample(test_data, test_size)
    
    # Sort by lemma for readability (groups all forms of same lemma together)
    train_data.sort(key=lambda x: x[0])
    dev_data.sort(key=lambda x: x[0])
    test_data.sort(key=lambda x: x[0])
    
    # Report statistics
    print(f"\n  Split by lemma (no overlap):")
    print(f"  Train: {len(train_verb_lemmas)} verb lemmas + {len(train_nonverb_lemmas)} non-verb lemmas = {len(train_data)} examples")
    print(f"  Dev:   {len(dev_verb_lemmas)} verb lemmas + {len(dev_nonverb_lemmas)} non-verb lemmas = {len(dev_data)} examples")
    print(f"  Test:  {len(test_verb_lemmas)} verb lemmas + {len(test_nonverb_lemmas)} non-verb lemmas = {len(test_data)} examples")
    
    return train_data, dev_data, test_data


def save_split(data, output_file):
    """
    Save data split to file.
    
    Args:
        data: List of tuples (lemma, features, form)
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for lemma, features, form in data:
            f.write(f"{lemma}\t{features}\t{form}\n")
    
    print(f"Saved {len(data)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare UniMorph data with consistent train/dev/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_unimorph_data.py por path/to/unimorph/por
  python prepare_unimorph_data.py azg path/to/unimorph/azg
  python prepare_unimorph_data.py dsb path/to/unimorph/dsb --train 8000 --dev 1000 --test 1000
        """
    )
    
    parser.add_argument('language', help='Language code (e.g., por, azg, dsb)')
    parser.add_argument('input_file', help='Path to original UniMorph data file')
    parser.add_argument('--train', type=int, default=10000, help='Training set size (default: 10000)')
    parser.add_argument('--dev', type=int, default=1000, help='Dev set size (default: 1000)')
    parser.add_argument('--test', type=int, default=1000, help='Test set size (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max-forms', type=int, default=None, help='Max forms per lemma to increase diversity (default: None = unlimited, try 15-20 for high-inflection languages)')
    parser.add_argument('--output-dir', default='data/unimorph', help='Output directory (default: data/unimorph)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    verbs, non_verbs = load_unimorph_data(input_path)
    print(f"Loaded {len(verbs) + len(non_verbs)} examples ({len(verbs)} verbs, {len(non_verbs)} non-verbs)")
    
    # Create splits
    print(f"\nCreating splits (train={args.train}, dev={args.dev}, test={args.test})...")
    train_data, dev_data, test_data = create_splits(
        verbs,
        non_verbs,
        train_size=args.train,
        dev_size=args.dev,
        test_size=args.test,
        seed=args.seed,
        max_forms_per_lemma=args.max_forms
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    print(f"\nSaving splits to {output_dir}...")
    
    save_split(train_data, output_dir / f"{args.language}.trn")
    save_split(dev_data, output_dir / f"{args.language}.dev")
    save_split(test_data, output_dir / f"{args.language}.tst")
    
    print("\nDone!")
    print(f"  Train: {len(train_data)} examples -> {args.language}.trn")
    print(f"  Dev:   {len(dev_data)} examples -> {args.language}.dev")
    print(f"  Test:  {len(test_data)} examples -> {args.language}.tst")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
