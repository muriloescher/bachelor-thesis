"""LLM-based morphological analysis using OpenRouter API."""
import os
import time
import json
import unicodedata
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv

from ..data import load_data
from ..utils import evaluate_inverse, ensure_dir, resolve_path
from ..utils.save_results import save_llm_results

# Load environment variables
load_dotenv()

# OpenRouter API configuration
URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models
AVAILABLE_MODELS = {
    'llama': "meta-llama/llama-3.1-8b-instruct",
    'qwen': "qwen/qwen3-8b"
}


class LLMModel:
    """LLM model for morphological analysis via API calls."""
    
    def __init__(self, config, lang_config):
        """
        Initialize LLM model.
        
        Args:
            config: Model configuration dict
            lang_config: Language configuration dict
        """
        self.config = config
        self.lang_config = lang_config
        self.lang_code = lang_config['language']['code']
        
        # Get API key
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Get model name
        model_shortname = config.get('model_name', 'qwen')
        self.model_name = AVAILABLE_MODELS.get(model_shortname, model_shortname)
        
        # Get output directory
        output_template = config.get('output_dir_template', 'results/llm-{model}-{lang}')
        self.output_dir = output_template.format(
            model=model_shortname,
            lang=self.lang_code
        )
        ensure_dir(self.output_dir)
        
        # Get prompt settings
        self.example_input = config.get('example_input', '')
        self.example_prediction = config.get('example_prediction', '')
        self.prompt_template = config.get('prompt_template', '')
        self.temperature = config.get('temperature', 0.0)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
        self.max_tokens = config.get('max_tokens', 2000)
        
    def train(self):
        """LLM models don't require training - they are used zero-shot."""
        print(f"\n{'='*60}")
        print(f"LLM Model: {self.model_name}")
        print(f"Language: {self.lang_code}")
        print(f"Note: LLM models are used zero-shot (no training required)")
        print(f"Use --predict to run evaluation")
        print(f"{'='*60}")
    
    def _create_prompt(self, form, context):
        """
        Create the prompt for the LLM.
        
        Args:
            form: Inflected verb form
            context: Sentence context
            
        Returns:
            List of message dicts for the API
        """
        ex_form, ex_context = self.example_input.split('\t')
        
        prompt_content = (
            f"Based on this example:\n"
            f"Input: {ex_form}\n"
            f"Context: {ex_context}\n"
            f"Prediction: {self.example_prediction}\n\n"
            f"{self.prompt_template}{form}\t{context}\n\n"
            f"Answer (lemma and tags only, no explanation):"
        )
        
        return [{"role": "user", "content": prompt_content}]
    
    def _query_model(self, form, context):
        """
        Query the LLM API.
        
        Args:
            form: Inflected verb form
            context: Sentence context
            
        Returns:
            Predicted string or None if error
        """
        payload = {
            "model": self.model_name,
            "messages": self._create_prompt(form, context),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url=URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                prediction = message.get('content', '').strip()
                if prediction:
                    return prediction
            
            return None
            
        except Exception as e:
            print(f"\nAPI Error: {e}")
            return None
    
    def _parse_prediction(self, prediction):
        """
        Parse LLM prediction into lemma and tags.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            Tuple of (lemma, tags)
        """
        if not prediction:
            return "", ""
        
        # Clean up the prediction - remove newlines, tabs, extra whitespace
        pred = prediction.strip().replace('\n', ' ')
        # Normalize multiple spaces to single space
        pred = ' '.join(pred.split())
        
        # Remove common markdown/formatting
        if pred.startswith('`') and pred.endswith('`'):
            pred = pred.strip('`')
        
        # Try tab-separated format first
        if '\t' in pred:
            parts = pred.split('\t', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        
        # Try space-separated format
        parts = pred.split(None, 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        elif len(parts) == 1:
            return parts[0].strip(), ""
        
        return "", ""
    
    def predict(self, checkpoint=None, inverse_only=True):
        """
        Run LLM predictions on test set.
        
        Args:
            checkpoint: Ignored (for API compatibility)
            inverse_only: Ignored (LLM only does inverse task)
        """
        print(f"\n{'='*60}")
        print(f"Running LLM predictions for {self.lang_code}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        
        # Load test data - use UD data with context
        data_config = self.lang_config['data']
        test_file = resolve_path(data_config['ud']['test'])
        
        print(f"Loading test data from {test_file}")
        test_data = load_data(str(test_file), has_context=True)
        print(f"Test examples: {len(test_data)}")
        
        # Prepare for predictions
        predictions = []
        gold_targets = []
        
        output_file = Path(self.output_dir) / f"predictions_{self.lang_code}_inverse.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for lemma, tags, form, context in tqdm(test_data, desc="Predicting"):
                # Query model
                prediction = self._query_model(form, context)
                pred_lemma, pred_tags = self._parse_prediction(prediction)
                
                # Write to file
                input_str = f"form: {form} | context: {context}"
                gold = f"{lemma} {tags}"
                pred = f"{pred_lemma} {pred_tags}" if pred_lemma or pred_tags else "ERROR"
                
                f.write(f"{input_str}\t{pred}\t{gold}\n")
                f.flush()
                
                # Store for evaluation
                predictions.append(pred)
                gold_targets.append(gold)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
        
        print(f"\nPredictions saved to: {output_file}")
        
        # Evaluate
        print("\nComputing metrics...")
        metrics = evaluate_inverse(predictions, gold_targets, str(output_file))
        
        if metrics:
            print(f"\nTest Results:")
            print(f"  Lemma accuracy: {metrics['lemma_accuracy']:.4f}")
            print(f"  Lemma mean Levenshtein: {metrics['mean_levenshtein']:.4f}")
            print(f"  MSD accuracy: {metrics['msd_accuracy']:.4f}")
            print(f"  MSD F1: {metrics['msd_f1']:.4f}")
            
            # Save results in JSON and CSV formats
            save_llm_results(
                output_dir=str(self.output_dir),
                model_name=self.model_name,
                language=self.lang_code,
                metrics=metrics,
                test_examples=len(test_data)
            )
            
            # Also save legacy TXT format for backward compatibility
            summary_file = Path(self.output_dir) / "results_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"LLM Results ({self.model_name})\n")
                f.write("="*50 + "\n\n")
                f.write(f"Language: {self.lang_code}\n")
                f.write(f"Test examples: {len(test_data)}\n\n")
                f.write("Results:\n")
                f.write(f"  Lemma Accuracy: {metrics['lemma_accuracy']:.4f} ({metrics['lemma_correct']}/{metrics['total']})\n")
                f.write(f"  Lemma Mean Levenshtein: {metrics['mean_levenshtein']:.4f}\n")
                f.write(f"  MSD Accuracy: {metrics['msd_accuracy']:.4f} ({metrics['msd_correct']}/{metrics['total']})\n")
                f.write(f"  MSD Micro-Precision: {metrics['msd_precision']:.4f}\n")
                f.write(f"  MSD Micro-Recall: {metrics['msd_recall']:.4f}\n")
                f.write(f"  MSD F1: {metrics['msd_f1']:.4f}\n")
            
            print(f"\nResults saved to:")
            print(f"  - JSON: {Path(self.output_dir) / 'results_summary.json'}")
            print(f"  - CSV: {Path(self.output_dir).parent / 'all_results.csv'}")
            print(f"  - TXT: {summary_file}")
        
        return metrics
