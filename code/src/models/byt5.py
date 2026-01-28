"""ByT5 model training and prediction."""
import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

from ..data import (
    load_data,
    build_forward_examples,
    build_inverse_examples,
    load_test_data_forward,
    load_test_data_inverse
)
from ..utils import evaluate_forward, evaluate_inverse, ensure_dir, resolve_path


def preprocess(example, tokenizer):
    """Preprocess example for seq2seq training."""
    model_inputs = tokenizer(example["input"], max_length=128, truncation=True)
    
    try:
        labels_enc = tokenizer(text_target=example["target"], max_length=32, truncation=True)
    except TypeError:
        try:
            with tokenizer.as_target_tokenizer():
                labels_enc = tokenizer(example["target"], max_length=32, truncation=True)
        except AttributeError:
            labels_enc = tokenizer(example["target"], max_length=32, truncation=True)
    
    model_inputs["labels"] = labels_enc["input_ids"]
    return model_inputs


class ByT5Model:
    """ByT5 model for morphological tasks."""
    
    def __init__(self, config, lang_config, mode='train'):
        """
        Initialize ByT5 model.
        
        Args:
            config: Model configuration dict
            lang_config: Language configuration dict
            mode: 'train' or 'predict'
        """
        self.config = config
        self.lang_config = lang_config
        self.mode = mode
        self.task = config['task']
        self.lang_code = lang_config['language']['code']
        
        # Set seed
        seed = config['training'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. ByT5 requires GPU.")
        self.device = torch.device("cuda")
        
        # Get output directory
        output_template = config.get('output_dir_template', 'results/byt5-{task}-{lang}')
        self.output_dir = output_template.format(task=self.task, lang=self.lang_code)
        ensure_dir(self.output_dir)
        
        # Determine if context is used from model config
        self.data_source = config.get('data_source', 'unimorph')
        self.has_context = (self.data_source == 'ud')
        
    def train(self):
        """Train the model."""
        print(f"\n{'='*60}")
        print(f"Training ByT5 ({self.task}) for {self.lang_code}")
        print(f"{'='*60}")
        
        # Load data
        data_config = self.lang_config['data']
        
        # Choose data files based on model's data source
        if self.has_context:
            # Use UD data (with context)
            train_file = resolve_path(data_config['ud']['train'])
            dev_file = resolve_path(data_config['ud']['dev'])
        else:
            # Use UniMorph data (no context)
            train_file = resolve_path(data_config['unimorph']['train'])
            dev_file = resolve_path(data_config['unimorph']['dev'])
        
        print(f"Loading training data from {train_file}")
        train_data_raw = load_data(str(train_file), self.has_context)
        dev_data_raw = load_data(str(dev_file), self.has_context)
        
        # Build examples based on task
        prompts = self.config['prompts']
        
        if self.task == 'forward':
            train_examples = build_forward_examples(train_data_raw, prompts['forward'], self.has_context)
            dev_examples = build_forward_examples(dev_data_raw, prompts['forward'], self.has_context)
        elif self.task == 'inverse':
            train_examples = build_inverse_examples(train_data_raw, prompts['inverse'], self.has_context)
            dev_examples = build_inverse_examples(dev_data_raw, prompts['inverse'], self.has_context)
        elif self.task == 'bidirectional':
            # 50/50 mix
            rng = random.Random(self.config['training'].get('seed', 42))
            train_shuf = train_data_raw.copy()
            dev_shuf = dev_data_raw.copy()
            rng.shuffle(train_shuf)
            rng.shuffle(dev_shuf)
            
            half_tr = len(train_shuf) // 2
            half_dv = len(dev_shuf) // 2
            
            train_examples = (build_forward_examples(train_shuf[:half_tr], prompts['forward'], self.has_context) +
                            build_inverse_examples(train_shuf[half_tr:], prompts['inverse'], self.has_context))
            dev_examples = (build_forward_examples(dev_shuf[:half_dv], prompts['forward'], self.has_context) +
                          build_inverse_examples(dev_shuf[half_dv:], prompts['inverse'], self.has_context))
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        print(f"Training examples: {len(train_examples)}")
        print(f"Dev examples: {len(dev_examples)}")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        dev_dataset = Dataset.from_list(dev_examples)
        
        # Load tokenizer and model
        model_name = self.config['model_name']
        # Replace {lang} placeholder with actual language code
        if '{lang}' in model_name:
            model_name = model_name.format(lang=self.lang_code)
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda x: preprocess(x, tokenizer),
            batched=False,
            remove_columns=[c for c in train_dataset.column_names if c in ("input", "target")]
        )
        dev_dataset = dev_dataset.map(
            lambda x: preprocess(x, tokenizer),
            batched=False,
            remove_columns=[c for c in dev_dataset.column_names if c in ("input", "target")]
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        
        # Training arguments
        train_cfg = self.config['training']
        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            **train_cfg
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )
        
        print(f"Starting training...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds")
        
        # Save final model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        return self.output_dir
    
    def predict(self, checkpoint=None, inverse_only=False):
        """
        Make predictions on test set.
        
        Args:
            checkpoint: Specific checkpoint to load (default: latest or main dir)
            inverse_only: For bidirectional models, only predict inverse direction
        """
        print(f"\n{'='*60}")
        print(f"Predicting with ByT5 ({self.task}) for {self.lang_code}")
        print(f"{'='*60}")
        
        # Find model directory
        if checkpoint:
            model_dir = os.path.join(self.output_dir, f"checkpoint-{checkpoint}")
        else:
            # Try checkpoint-3750 first, then fall back to output_dir
            checkpoint_dir = os.path.join(self.output_dir, "checkpoint-3750")
            model_dir = checkpoint_dir if os.path.exists(checkpoint_dir) else self.output_dir
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        print(f"Loading model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        model.eval()
        
        # Load test data
        data_config = self.lang_config['data']
        if self.has_context:
            # Use UD test data when context is needed
            test_file = resolve_path(data_config['ud']['test'])
        else:
            # Use UniMorph test data by default
            test_file = resolve_path(data_config['unimorph']['test'])
        
        print(f"Loading test data from {test_file}")
        
        prompts = self.config['prompts']
        pred_cfg = self.config['prediction']
        batch_size = pred_cfg.get('batch_size', 16)
        
        results = {}
        
        # Predict based on task
        if self.task == 'bidirectional':
            # Forward direction
            if not inverse_only:
                print("Predicting forward direction...")
                test_inputs, gold_outputs = load_test_data_forward(str(test_file), prompts['forward'], self.has_context)
                predictions = self._batch_predict(model, tokenizer, test_inputs, batch_size)
                
                # Save and evaluate
                output_file = os.path.join(self.output_dir, f"predictions_{self.lang_code}_forward.txt")
                self._save_predictions(output_file, test_inputs, predictions, gold_outputs)
                results['forward'] = evaluate_forward(predictions, gold_outputs, output_file)
                print(f"  Forward accuracy: {results['forward']['accuracy']:.4f}")
            
            # Inverse direction
            print("Predicting inverse direction...")
            test_inputs, gold_outputs = load_test_data_inverse(str(test_file), prompts['inverse'], self.has_context)
            predictions = self._batch_predict(model, tokenizer, test_inputs, batch_size)
            
            # Save and evaluate
            output_file = os.path.join(self.output_dir, f"predictions_{self.lang_code}_inverse.txt")
            self._save_predictions(output_file, test_inputs, predictions, gold_outputs)
            results['inverse'] = evaluate_inverse(predictions, gold_outputs, output_file)
            print(f"  Inverse lemma accuracy: {results['inverse']['lemma_accuracy']:.4f}")
            print(f"  Inverse MSD F1: {results['inverse']['msd_f1']:.4f}")
            
        elif self.task == 'inverse':
            test_inputs, gold_outputs = load_test_data_inverse(str(test_file), prompts['inverse'], self.has_context)
            predictions = self._batch_predict(model, tokenizer, test_inputs, batch_size)
            
            output_file = os.path.join(self.output_dir, f"predictions_{self.lang_code}_inverse.txt")
            self._save_predictions(output_file, test_inputs, predictions, gold_outputs)
            results = evaluate_inverse(predictions, gold_outputs, output_file)
            print(f"  Lemma accuracy: {results['lemma_accuracy']:.4f}")
            print(f"  MSD F1: {results['msd_f1']:.4f}")
            
        else:  # forward
            test_inputs, gold_outputs = load_test_data_forward(str(test_file), prompts['forward'], self.has_context)
            predictions = self._batch_predict(model, tokenizer, test_inputs, batch_size)
            
            output_file = os.path.join(self.output_dir, f"predictions_{self.lang_code}_forward.txt")
            self._save_predictions(output_file, test_inputs, predictions, gold_outputs)
            results = evaluate_forward(predictions, gold_outputs, output_file)
            print(f"  Accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def _batch_predict(self, model, tokenizer, inputs, batch_size):
        """Make predictions in batches."""
        predictions = []
        max_length = self.config['prediction'].get('max_length', 32)
        
        for i in tqdm(range(0, len(inputs), batch_size), desc="Predicting", unit="batch"):
            batch_inputs = inputs[i:i+batch_size]
            tokenized = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = model.generate(**tokenized, max_length=max_length)
            
            batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        
        return predictions
    
    def _save_predictions(self, output_file, inputs, predictions, gold):
        """Save predictions to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for inp, pred, g in zip(inputs, predictions, gold):
                f.write(f"{inp}\t{pred}\t{g}\n")
        print(f"Saved predictions to {output_file}")
