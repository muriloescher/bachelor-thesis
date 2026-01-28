#!/usr/bin/env python3
"""
Convert CoNLL-U files to UniMorph format.

This script extracts the first verb from each sentence in a CoNLL-U file
and outputs it in UniMorph format: lemma \t features \t form \t sentence
"""

import argparse
from pathlib import Path


def reorder_unimorph_tags(features_str, upos):
    """
    Reorder morphological features to match UniMorph canonical ordering.
    
    UniMorph ordering (by category):
    1. POS tag (V, N, ADJ, etc.)
    2. Mood/VerbForm (IND, IMP, COND, SBJV, MASD, INF, PART, etc.)
    3. Tense/Aspect (PRS, PST, FUT, PRF, PFV, IPFV, etc.)
    4. Voice (ACT, PASS, MID, etc.)
    5. Agreement features (person/number for subjects/objects)
    6. Case (NOM, ACC, DAT, ERG, GEN, etc.)
    7. Other features (Gender, Animacy, etc.)
    
    Args:
        features_str: Morphological features string (semicolon-separated tags)
        upos: Universal POS tag
        
    Returns:
        Reordered features string in UniMorph format
    """
    if not features_str or features_str == '_':
        return upos
    
    # Parse semicolon-separated tags
    tags = features_str.split(';')
    
    # Categories for ordering
    pos_tag = None
    mood_tags = []  # IND, IMP, COND, SBJV, INF, PART, CONV, MASD, FIN, etc.
    tense_aspect_tags = []  # PRS, PST, FUT, PRF, PFV, IPFV, etc.
    voice_tags = []  # ACT, PASS, MID
    agreement_tags = []  # NOM(...), ACC(...), DAT(...), ERG(...)
    case_tags = []  # For non-verbs
    other_tags = []
    
    # Known categories
    mood_verbform = {'IND', 'IMP', 'COND', 'SBJV', 'INF', 'PART', 'CONV', 'MASD', 'NMLZ', 'FIN'}
    tense_aspect = {'PRS', 'PST', 'FUT', 'PRF', 'PFV', 'IPFV'}
    voice = {'ACT', 'PASS', 'MID'}
    case_markers = {'NOM', 'ACC', 'DAT', 'ERG', 'GEN', 'INS', 'VOC', 'ABL', 'LOC'}
    
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        
        # POS tag (usually first)
        if tag in {'V', 'N', 'ADJ', 'VERB'}:
            pos_tag = 'V' if tag == 'VERB' else tag
        # Mood/VerbForm
        elif tag in mood_verbform:
            mood_tags.append(tag)
        # Tense/Aspect
        elif tag in tense_aspect:
            tense_aspect_tags.append(tag)
        # Voice
        elif tag in voice:
            voice_tags.append(tag)
        # Agreement/Case with parentheses (e.g., NOM(2,SG), ACC(3,PL))
        elif '(' in tag:
            # Check if it's a case marker
            base = tag.split('(')[0]
            if base in case_markers:
                agreement_tags.append(tag)
            else:
                other_tags.append(tag)
        # Case markers without parentheses
        elif tag in case_markers:
            case_tags.append(tag)
        else:
            other_tags.append(tag)
    
    # Build result in canonical order
    result = []
    
    if pos_tag:
        result.append(pos_tag)
    
    result.extend(mood_tags)
    result.extend(tense_aspect_tags)
    result.extend(voice_tags)
    result.extend(agreement_tags)
    result.extend(case_tags)
    result.extend(other_tags)
    
    return ';'.join(result)


def convert_conllu_to_unimorph(input_path, output_path):
    """
    Convert a CoNLL-U file to UniMorph format.
    
    Args:
        input_path: Path to input CoNLL-U file
        output_path: Path to output UniMorph file
        
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
            
            # Check if we're starting a new sentence block
            if line.startswith('# sent_id'):
                in_sentence = True
                sentence_text = None
                verb_data = None
            # Empty line marks end of sentence block
            elif not line and in_sentence:
                if verb_data and sentence_text:
                    results.append(verb_data + (sentence_text,))
                in_sentence = False
            # Process lines within a sentence block
            elif in_sentence:
                # Extract sentence text from comment
                if line.startswith('# text = '):
                    sentence_text = line[9:].strip()
                # Skip other comments
                elif line.startswith('#'):
                    continue
                # Process token line
                else:
                    parts = line.split('\t')
                    if len(parts) < 7:
                        continue
                    
                    token_id = parts[0]
                    
                    # Skip multiword tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
                    if '-' in token_id or '.' in token_id:
                        continue
                    
                    form = parts[1]        # Inflected word form
                    lemma = parts[2]       # Lemma (base form/infinitive)
                    upos = parts[3]        # Universal POS tag
                    features = parts[5]    # Morphological features
                    
                    # Check if this is a verb and we haven't found one yet
                    if upos == 'VERB' and verb_data is None:
                        # Reorder features to match UniMorph canonical ordering
                        # print(f"Converting verb: {form} ({lemma}) with features: {features}, upos: {upos}")
                        features_reordered = reorder_unimorph_tags(features, upos)
                        verb_data = (lemma, features_reordered, form)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as fout:
        for lemma, features, form, sentence in results:
            fout.write(f"{lemma}\t{features}\t{form}\t{sentence}\n")
    
    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description='Convert CoNLL-U files to UniMorph format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python conllu_to_unimorph.py ka_gnc-ud-train.conllu kat.trn
  
  # Process all files for a language
  python conllu_to_unimorph.py ka_gnc-ud-train.conllu kat.trn
  python conllu_to_unimorph.py ka_gnc-ud-dev.conllu kat.dev
  python conllu_to_unimorph.py ka_gnc-ud-test.conllu kat.tst
        """
    )
    
    parser.add_argument('input', type=str, help='Input CoNLL-U file')
    parser.add_argument('output', type=str, help='Output UniMorph file')
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
    
    count = convert_conllu_to_unimorph(input_path, output_path)
    
    if args.verbose:
        print(f"Successfully processed {count} sentences")
        print(f"Output written to {output_path}")
    else:
        print(f"{input_path.name}: {count} sentences -> {output_path.name}")
    
    return 0


if __name__ == '__main__':
    exit(main())
