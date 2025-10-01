import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    exit()

tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-small"
)
model = model.to(torch.float16)
model = model.to("cuda")

input_ids = tokenizer("summarize: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))

print("Number of GPUs:", torch.cuda.device_count())
print("Current GPU device:", torch.cuda.current_device())
print("Device:", model.device)