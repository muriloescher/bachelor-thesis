#!/usr/bin/env python3
"""
Split Amharic CoNLL-U file into train/dev/test and convert to UniMorph format.

This script:
1. Reads the am_att-ud-test.conllu file
2. Splits it into train (80%), dev (10%), test (10%)
3. Converts each split to UniMorph format
"""

import random
from pathlib import Path
from conllu_to_unimorph import convert_conllu_to_unimorph


def read_conllu_sentences(input_path):
    """
    Read all sentences from a CoNLL-U file.
    
    Returns:
        List of sentence blocks (each block is a list of lines)
    """
    sentences = []
    current_sentence = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if not line:  # Empty line marks end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        
        # Add last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def write_conllu_sentences(sentences, output_path):
    """Write sentences to a CoNLL-U file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # Empty line after each sentence


def split_data(sentences, train_ratio=0.8, dev_ratio=0.1, seed=42):
    """
    Split sentences into train/dev/test sets.
    
    Args:
        sentences: List of sentence blocks
        train_ratio: Fraction for training (default 0.8)
        dev_ratio: Fraction for development (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_sentences, dev_sentences, test_sentences)
    """
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    train_set = shuffled[:train_end]
    dev_set = shuffled[train_end:dev_end]
    test_set = shuffled[dev_end:]
    
    return train_set, dev_set, test_set


def main():
    # Set up paths
    data_dir = Path(__file__).parent
    input_file = data_dir / 'am_att-ud-test.conllu'
    
    print("="*60)
    print("Splitting and Converting Amharic Data")
    print("="*60)
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return 1
    
    # Read all sentences
    print(f"\nReading {input_file.name}...")
    sentences = read_conllu_sentences(input_file)
    print(f"Total sentences: {len(sentences)}")
    
    # Split into train/dev/test
    print("\nSplitting data (80/10/10)...")
    train, dev, test = split_data(sentences, train_ratio=0.8, dev_ratio=0.1, seed=42)
    print(f"  Train: {len(train)} sentences ({len(train)/len(sentences)*100:.1f}%)")
    print(f"  Dev:   {len(dev)} sentences ({len(dev)/len(sentences)*100:.1f}%)")
    print(f"  Test:  {len(test)} sentences ({len(test)/len(sentences)*100:.1f}%)")
    
    # Write CoNLL-U splits
    print("\nWriting CoNLL-U splits...")
    conllu_files = {
        'train': data_dir / 'am_att-ud-train.conllu',
        'dev': data_dir / 'am_att-ud-dev.conllu',
        'test': data_dir / 'am_att-ud-test-split.conllu'
    }
    
    write_conllu_sentences(train, conllu_files['train'])
    write_conllu_sentences(dev, conllu_files['dev'])
    write_conllu_sentences(test, conllu_files['test'])
    
    for split, path in conllu_files.items():
        print(f"  {split}: {path.name}")
    
    # Convert to UniMorph format
    print("\nConverting to UniMorph format...")
    unimorph_files = {
        'train': (conllu_files['train'], data_dir / 'amh.trn'),
        'dev': (conllu_files['dev'], data_dir / 'amh.dev'),
        'test': (conllu_files['test'], data_dir / 'amh.tst')
    }
    
    for split, (conllu_path, unimorph_path) in unimorph_files.items():
        count = convert_conllu_to_unimorph(conllu_path, unimorph_path)
        print(f"  {split}: {count} verbs -> {unimorph_path.name}")
    
    print("\n" + "="*60)
    print("Done! Files created:")
    print("="*60)
    print("\nCoNLL-U files:")
    for split, path in conllu_files.items():
        print(f"  {path}")
    print("\nUniMorph files:")
    for split, (_, path) in unimorph_files.items():
        print(f"  {path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
