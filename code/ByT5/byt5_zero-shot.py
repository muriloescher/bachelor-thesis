import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import time

# Only use GPU
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    exit()
device = torch.device("cuda")

model_size = "base"  # Change to "small" or "base" if needed

# Load ByT5 model and tokenizer
model_name = f"google/byt5-{model_size}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)

import glob

# Path to data folder
data_path = "../data"

# Find all .tst files (languages)
test_files = sorted(glob.glob(f"{data_path}/*.tst"))
languages = [f.split('/')[-1].split('.')[0] for f in test_files]

# List of prompts to test
prompts = [
    "{lemma} {features}",
    "Generate the inflected form for: {lemma} {features}",
    "Generate the inflected form for the lemma: \"{lemma}\", according to the following morphological descriptions: \"{features}\"",
    "Your task is to inflect verbs based on their lemma and morphological features. You are going to be given a verb lemma followed by its morphological features, following the schema adopted in Unimorph 4.0. Provide the correct inflected form of the verb based on these features. Here is an example format: \"lemma features\" -> \"inflected_form\". Now, please inflect the following verb: \"{lemma} {features}\"",
]

batch_size = 16

for lang, test_path in zip(languages, test_files):
    # Read test set
    with open(test_path, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]
    print(f"\nEvaluating language: {lang} ({test_path})")
    for prompt in prompts:
        print(f"  Using prompt: '{prompt}'")
        output_path = f"output/predictions_{model_size}_{lang}_{prompts.index(prompt)}.txt"
        start_time = time.time()
        num_examples = len(test_lines)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i in tqdm(range(0, num_examples, batch_size), desc=f"{lang}-{prompts.index(prompt)}", unit="batch"):
                batch_lines = test_lines[i:i+batch_size]
                batch_inputs = []
                for line in batch_lines:
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    lemma, features = parts[0], parts[1]
                    input_str = prompt.format(lemma=lemma, features=features)
                    batch_inputs.append(input_str)
                if not batch_inputs:
                    continue
                inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_length=32)
                predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for inp, pred in zip(batch_inputs, predictions):
                    out_f.write(f"{inp}\t{pred}\n")
        elapsed = time.time() - start_time
        print(f"    Total time: {elapsed:.2f} seconds. Output: {output_path}")
