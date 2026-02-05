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


def create_splits(verbs, non_verbs, train_size=10000, dev_size=1000, test_size=1000, seed=42):
    """
    Create randomized train/dev/test splits, prioritizing verbs.
    
    Splits verbs and non-verbs proportionally across train/dev/test to maintain balance.
    
    Args:
        verbs: List of verb data tuples
        non_verbs: List of non-verb data tuples
        train_size: Number of training examples
        dev_size: Number of dev examples
        test_size: Number of test examples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle both datasets
    shuffled_verbs = verbs.copy()
    random.shuffle(shuffled_verbs)
    
    shuffled_non_verbs = non_verbs.copy()
    random.shuffle(shuffled_non_verbs)
    
    # Calculate how much data we have
    total_needed = train_size + dev_size + test_size
    total_verbs = len(shuffled_verbs)
    total_available = total_verbs + len(shuffled_non_verbs)
    
    # Report verb availability
    print(f"  Verbs available: {total_verbs}")
    print(f"  Non-verbs available: {len(shuffled_non_verbs)}")
    print(f"  Total needed: {total_needed}")
    
    # Check if we need to adjust sizes
    if total_available < total_needed:
        print(f"  ⚠️  Warning: Only {total_available} total examples available")
        print(f"  Adjusting split sizes proportionally...")
        ratio = total_available / total_needed
        train_size = int(train_size * ratio)
        dev_size = int(dev_size * ratio)
        test_size = total_available - train_size - dev_size
        total_needed = train_size + dev_size + test_size
    
    # Determine split strategy
    if total_verbs >= total_needed:
        # We have enough verbs - use only verbs, split proportionally
        print(f"  ✓ Using only verbs (sufficient for all splits)")
        
        # Split verbs proportionally across train/dev/test
        verb_train = shuffled_verbs[:train_size]
        verb_dev = shuffled_verbs[train_size:train_size + dev_size]
        verb_test = shuffled_verbs[train_size + dev_size:train_size + dev_size + test_size]
        
        train_data = verb_train
        dev_data = verb_dev
        test_data = verb_test
        
        print(f"  Composition: 100% verbs in each split")
    else:
        # Need to supplement with non-verbs - split both proportionally
        needed_non_verbs = total_needed - total_verbs
        verb_ratio = total_verbs / total_needed
        
        print(f"  ⚠️  Using all {total_verbs} verbs + {needed_non_verbs} non-verbs")
        print(f"  Verb ratio: {verb_ratio*100:.1f}%")
        
        # Calculate how many verbs go in each split (proportional to split size)
        verb_train_size = int(train_size * verb_ratio)
        verb_dev_size = int(dev_size * verb_ratio)
        verb_test_size = total_verbs - verb_train_size - verb_dev_size  # remainder
        
        # Calculate how many non-verbs go in each split
        nonverb_train_size = train_size - verb_train_size
        nonverb_dev_size = dev_size - verb_dev_size
        nonverb_test_size = test_size - verb_test_size
        
        # Split verbs
        verb_train = shuffled_verbs[:verb_train_size]
        verb_dev = shuffled_verbs[verb_train_size:verb_train_size + verb_dev_size]
        verb_test = shuffled_verbs[verb_train_size + verb_dev_size:]
        
        # Split non-verbs
        nonverb_train = shuffled_non_verbs[:nonverb_train_size]
        nonverb_dev = shuffled_non_verbs[nonverb_train_size:nonverb_train_size + nonverb_dev_size]
        nonverb_test = shuffled_non_verbs[nonverb_train_size + nonverb_dev_size:nonverb_train_size + nonverb_dev_size + nonverb_test_size]
        
        # Combine and shuffle each split to mix verbs and non-verbs
        train_data = verb_train + nonverb_train
        random.shuffle(train_data)
        
        dev_data = verb_dev + nonverb_dev
        random.shuffle(dev_data)
        
        test_data = verb_test + nonverb_test
        random.shuffle(test_data)
        
        print(f"  Train: {len(verb_train)} verbs + {len(nonverb_train)} non-verbs")
        print(f"  Dev:   {len(verb_dev)} verbs + {len(nonverb_dev)} non-verbs")
        print(f"  Test:  {len(verb_test)} verbs + {len(nonverb_test)} non-verbs")
    
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
        seed=args.seed
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
