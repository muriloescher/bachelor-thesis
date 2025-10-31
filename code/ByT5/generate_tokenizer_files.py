import os
import argparse
from typing import List

# Minimal utility to save a tokenizer into model directories so prediction can load locally.
# Usage examples:
#   python3 generate_tokenizer_files.py --inverse --langs eng,por
#   python3 generate_tokenizer_files.py --dirs ./byt5-inverse-eng ./byt5-inverse-por
#   python3 generate_tokenizer_files.py --langs eng --base-model google/byt5-small
#
# This script does not modify your training code. It only writes tokenizer files
# (tokenizer.json, tokenizer_config.json, special_tokens_map.json, etc.) into the
# specified directories using the base model's tokenizer.

def discover_langs(data_path: str) -> List[str]:
    return sorted([f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.trn')])


def main():
    parser = argparse.ArgumentParser("Generate tokenizer files for ByT5 model directories")
    parser.add_argument("--langs", type=str, help="Comma-separated language codes (e.g. eng,por)")
    parser.add_argument("--inverse", action="store_true", help="Target inverse dirs (byt5-inverse-<lang>)")
    parser.add_argument("--dirs", nargs="*", help="Explicit target directories to write tokenizer files into")
    parser.add_argument("--base-model", type=str, default="google/byt5-small", help="Tokenizer source model")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tokenizer files if present")
    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, "../data"))

    targets: List[str] = []
    if args.dirs:
        targets.extend(args.dirs)

    langs = []
    if args.langs:
        langs = [l.strip() for l in args.langs.split(',') if l.strip()]
    elif not args.dirs:
        # If no dirs and no langs provided, target all languages found in data
        try:
            langs = discover_langs(data_path)
        except FileNotFoundError:
            print(f"Data path not found: {data_path}")
            langs = []

    if langs:
        for lang in langs:
            dir_name = f"byt5-{'inverse-' if args.inverse else ''}{lang}"
            targets.append(os.path.join(script_dir, dir_name))

    if not targets:
        print("No target directories resolved. Provide --dirs or --langs (or ensure ../data exists).")
        return

    # Lazy import to avoid requiring transformers unless needed
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("Transformers is required. Please install it in your environment (pip install transformers).")
        print(f"Details: {e}")
        return

    print(f"Loading tokenizer from base model: {args.base_model}")
    try:
        tok = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as e:
        print(f"Failed to load tokenizer from '{args.base_model}': {e}")
        return

    wrote_any = False
    for tgt in targets:
        try:
            os.makedirs(tgt, exist_ok=True)
            # Detect existing tokenizer files
            existing = any(
                os.path.exists(os.path.join(tgt, fname))
                for fname in (
                    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                    'spiece.model', 't5_tokenizer.model'
                )
            )
            if existing and not args.overwrite:
                print(f"[skip] {tgt}: tokenizer files already present (use --overwrite to replace)")
                continue
            tok.save_pretrained(tgt)
            print(f"[ok]   {tgt}: saved tokenizer files")
            wrote_any = True
        except Exception as e:
            print(f"[fail] {tgt}: {e}")

    if not wrote_any:
        print("No tokenizer files written. Nothing to do.")


if __name__ == "__main__":
    main()
