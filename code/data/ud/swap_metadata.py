#!/usr/bin/env python3
"""
Script to swap the order of # text and # sent_id lines in CoNLL-U files.
"""

import sys
import re

def swap_metadata_lines(input_file, output_file=None):
    """
    Swap # text and # sent_id lines in a CoNLL-U file.
    
    Args:
        input_file: Path to input CoNLL-U file
        output_file: Path to output file (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    result = []
    
    while i < len(lines):
        # Check if current line is # text and next is # sent_id
        if (i + 1 < len(lines) and 
            lines[i].startswith('# text =') and 
            lines[i + 1].startswith('# sent_id =')):
            # Swap them
            result.append(lines[i + 1])
            result.append(lines[i])
            i += 2
        # Check if current line is # sent_id and next is # text
        elif (i + 1 < len(lines) and 
              lines[i].startswith('# sent_id =') and 
              lines[i + 1].startswith('# text =')):
            # Already in desired order, but let's be consistent
            result.append(lines[i])
            result.append(lines[i + 1])
            i += 2
        else:
            # Not a metadata pair, just add the line
            result.append(lines[i])
            i += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(result)
    
    print(f"Successfully swapped metadata lines in {input_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python swap_metadata.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    swap_metadata_lines(input_file, output_file)
