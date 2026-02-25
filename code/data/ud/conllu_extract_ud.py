#!/usr/bin/env python3
"""
Extract verb forms from CoNLL-U files in UD format.

This script extracts the first verb from each sentence in a CoNLL-U file
and outputs it in tab-separated format: lemma \t features \t form \t sentence

Features are kept in UD format, with values extracted and joined by semicolons.
Example: Mood=Ind|VerbForm=Fin|Voice=Cau -> Ind;Fin;Cau
"""

import argparse
from pathlib import Path


def convert_ud_features(features_str):
    """
    Convert UD feature string to semicolon-separated values.

    Splits on '|', extracts values after '=', joins with ';'.
    Example: Mood=Ind|VerbForm=Fin|Voice=Cau -> Ind;Fin;Cau

    Args:
        features_str: UD morphological features string (pipe-separated key=value pairs)

    Returns:
        Semicolon-separated feature values, or '_' if no features
    """
    if not features_str or features_str == '_':
        return '_'

    values = []
    for feature in features_str.split('|'):
        feature = feature.strip()
        if '=' in feature:
            values.append(feature.split('=', 1)[1])
        elif feature:
            values.append(feature)

    return ';'.join(values)


def extract_verbs_from_conllu(input_path, output_path):
    """
    Extract first verb from each sentence in a CoNLL-U file.

    Args:
        input_path: Path to input CoNLL-U file
        output_path: Path to output tab-separated file

    Returns:
        Number of sentences processed
    """
    results = []
    in_sentence = False
    sentence_text = None
    verb_data = None

    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip('\n')

            if line.startswith('# sent_id'):
                in_sentence = True
                sentence_text = None
                verb_data = None
            elif not line and in_sentence:
                if verb_data and sentence_text:
                    results.append(verb_data + (sentence_text,))
                in_sentence = False
            elif in_sentence:
                if line.startswith('# text = '):
                    sentence_text = line[9:].strip()
                elif line.startswith('#'):
                    continue
                else:
                    parts = line.split('\t')
                    if len(parts) < 7:
                        continue

                    token_id = parts[0]

                    # Skip multiword tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
                    if '-' in token_id or '.' in token_id:
                        continue

                    form = parts[1]
                    lemma = parts[2]
                    upos = parts[3]
                    features = parts[5]

                    if upos == 'VERB' and verb_data is None:
                        features_converted = convert_ud_features(features)
                        verb_data = (lemma, features_converted, form)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for lemma, features, form, sentence in results:
            fout.write(f"{lemma}\t{features}\t{form}\t{sentence}\n")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description='Extract verb forms from CoNLL-U files in UD feature format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python conllu_extract_ud.py am_att-ud-train.conllu amh.trn
  python conllu_extract_ud.py am_att-ud-dev.conllu amh.dev
  python conllu_extract_ud.py am_att-ud-test.conllu amh.tst
        """
    )

    parser.add_argument('input', type=str, help='Input CoNLL-U file')
    parser.add_argument('output', type=str, help='Output tab-separated file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    if args.verbose:
        print(f"Processing {input_path}...")

    count = extract_verbs_from_conllu(input_path, output_path)

    if args.verbose:
        print(f"Successfully processed {count} sentences")
        print(f"Output written to {output_path}")
    else:
        print(f"{input_path.name}: {count} sentences -> {output_path.name}")

    return 0


if __name__ == '__main__':
    exit(main())
