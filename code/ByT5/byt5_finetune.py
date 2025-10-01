from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

# TODO: Replace with your actual data loading logic
# Example data for demonstration
train_data = [
    {"input": "andar V;PRS;3;SG", "target": "anda"},
    {"input": "andar V;PST;3;SG", "target": "andou"},
]
dev_data = [
    {"input": "andar V;PRS;1;SG", "target": "ando"},
    {"input": "andar V;PST;1;SG", "targe    pip install --upgrade torch torchvision transformerst": "andei"},
]

train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)

model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(example):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=32)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=32)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess, batched=False)
dev_dataset = dev_dataset.map(preprocess, batched=False)

args = Seq2SeqTrainingArguments(
    output_dir="./byt5-test",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=50,
    fp16=True,  # Set to False if your GPU does not support fp16
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()
