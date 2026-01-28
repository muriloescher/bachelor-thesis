"""Data loading utilities for morphological datasets."""
import os
from typing import List, Tuple, Dict


def load_data(file_path: str, has_context: bool = False) -> List[Tuple]:
    """
    Load morphological data from TSV file.
    
    Args:
        file_path: Path to data file
        has_context: Whether data includes context column
        
    Returns:
        List of tuples: (lemma, features, form) or (lemma, features, form, context)
    """
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if has_context:
                if len(parts) < 4:
                    continue
                lemma, features, form, context = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if lemma or features or form:
                    data.append((lemma, features, form, context))
            else:
                if len(parts) < 3:
                    continue
                lemma, features, form = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if lemma or features or form:
                    data.append((lemma, features, form))
    return data


def build_forward_examples(data: List[Tuple], prompt_template: str, has_context: bool = False) -> List[Dict]:
    """
    Build forward task examples (lemma+features -> form).
    
    Args:
        data: List of data tuples
        prompt_template: Prompt template with {lemma}, {features}, optional {context}
        has_context: Whether data includes context
        
    Returns:
        List of dicts with 'input' and 'target' keys
    """
    examples = []
    for item in data:
        if has_context:
            lemma, features, form, context = item
            if not form:
                continue
            input_str = prompt_template.format(lemma=lemma, features=features, context=context)
            examples.append({"input": input_str, "target": form})
        else:
            lemma, features, form = item
            if not form:
                continue
            input_str = prompt_template.format(lemma=lemma, features=features)
            examples.append({"input": input_str, "target": form})
    return examples


def build_inverse_examples(data: List[Tuple], prompt_template: str, has_context: bool = False) -> List[Dict]:
    """
    Build inverse task examples (form -> lemma+features).
    
    Args:
        data: List of data tuples
        prompt_template: Prompt template with {form}, optional {context}
        has_context: Whether data includes context
        
    Returns:
        List of dicts with 'input' and 'target' keys
    """
    examples = []
    for item in data:
        if has_context:
            lemma, features, form, context = item
            if not form:
                continue
            input_str = prompt_template.format(form=form, context=context)
            target = f"{lemma} {features}".strip()
            if target:
                examples.append({"input": input_str, "target": target})
        else:
            lemma, features, form = item
            if not form:
                continue
            input_str = prompt_template.format(form=form)
            target = f"{lemma} {features}".strip()
            if target:
                examples.append({"input": input_str, "target": target})
    return examples


def load_test_data_forward(file_path: str, prompt_template: str, has_context: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load test data for forward task.
    
    Returns:
        Tuple of (test_inputs, gold_outputs)
    """
    data = load_data(file_path, has_context)
    test_inputs = []
    gold_outputs = []
    
    for item in data:
        if has_context:
            lemma, features, form, context = item
            if not form:
                continue
            input_str = prompt_template.format(lemma=lemma, features=features, context=context)
            test_inputs.append(input_str)
            gold_outputs.append(form)
        else:
            lemma, features, form = item
            if not form:
                continue
            input_str = prompt_template.format(lemma=lemma, features=features)
            test_inputs.append(input_str)
            gold_outputs.append(form)
    
    return test_inputs, gold_outputs


def load_test_data_inverse(file_path: str, prompt_template: str, has_context: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load test data for inverse task.
    
    Returns:
        Tuple of (test_inputs, gold_outputs)
    """
    data = load_data(file_path, has_context)
    test_inputs = []
    gold_outputs = []
    
    for item in data:
        if has_context:
            lemma, features, form, context = item
            if not form:
                continue
            input_str = prompt_template.format(form=form, context=context)
            target = f"{lemma} {features}".strip()
            if target:
                test_inputs.append(input_str)
                gold_outputs.append(target)
        else:
            lemma, features, form = item
            if not form:
                continue
            input_str = prompt_template.format(form=form)
            target = f"{lemma} {features}".strip()
            if target:
                test_inputs.append(input_str)
                gold_outputs.append(target)
    
    return test_inputs, gold_outputs
