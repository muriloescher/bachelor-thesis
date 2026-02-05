"""Non-neural baseline wrapper."""
import os
import re
import subprocess
from pathlib import Path
from ..data import load_data
from ..utils import resolve_path, ensure_dir, evaluate_forward, save_nonneural_results


class NonNeuralModel:
    """Wrapper for non-neural baseline system."""
    
    def __init__(self, config, lang_config):
        """
        Initialize non-neural baseline.
        
        Args:
            config: Model configuration dict
            lang_config: Language configuration dict
        """
        self.config = config
        self.lang_config = lang_config
        self.lang_code = lang_config['language']['code']
        
        # Get baseline script path
        self.baseline_script = resolve_path('baselines/nonneural/nonneural.py')
        if not self.baseline_script.exists():
            raise FileNotFoundError(f"Non-neural baseline script not found: {self.baseline_script}")
        
        # Get output directory
        output_template = config.get('output_dir_template', 'results/nonneural-{lang}')
        self.output_dir = output_template.format(lang=self.lang_code)
        ensure_dir(self.output_dir)
    
    def train(self):
        """Non-neural baseline doesn't have separate training - it learns rules during prediction."""
        print(f"\n{'='*60}")
        print(f"Non-neural baseline for {self.lang_code}")
        print(f"Note: This baseline learns rules during prediction (no separate training)")
        print(f"Use --predict to run the baseline")
        print(f"{'='*60}")
    
    
    def predict(self, is_test=True):
        """
        Run non-neural baseline prediction on test set.
        
        Args:
            is_test: Always use test set (default True for consistency)
        """
        print(f"\n{'='*60}")
        print(f"Running non-neural baseline for {self.lang_code}")
        print(f"{'='*60}")
        
        # Get data paths - use unimorph data (3-column format: lemma\tmsd\tform)
        data_config = self.lang_config['data']
        train_file = resolve_path(data_config['unimorph']['train'])
        test_file = resolve_path(data_config['unimorph']['test'])
        data_dir = train_file.parent
        
        # Check that files are named correctly for baseline
        # Baseline expects: {lang}.trn, {lang}.tst
        expected_train = data_dir / f"{self.lang_code}.trn"
        expected_test = data_dir / f"{self.lang_code}.tst"
        
        if not expected_train.exists() or not expected_test.exists():
            print(f"\n⚠️  Warning: Expected files {expected_train.name} and {expected_test.name}")
            print(f"   Found: {train_file.name} and {test_file.name}")
            print(f"   The baseline script may fail. Check data file naming.")
        
        print(f"Data directory: {data_dir}")
        print(f"Training file: {train_file.name}")
        print(f"Test file: {test_file.name}")
        
        # Run the baseline script
        # It will process the language and print accuracy
        cmd = [
            'python3',
            str(self.baseline_script),
            '--path', str(data_dir) + '/',
            '--lang', self.lang_code,  # Specify which language to run
            '--test',  # Always evaluate on test
            '--output'  # Generate output file
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.baseline_script.parent))
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"Non-neural baseline failed with return code {result.returncode}")
        
        # Parse accuracy from output
        # Format: "por: 0.12345"
        accuracy = None
        for line in result.stdout.split('\n'):
            if self.lang_code in line and ':' in line:
                match = re.search(r':\s*([0-9.]+)', line)
                if match:
                    accuracy = float(match.group(1))
                    break
        
        if accuracy is None:
            print(f"⚠️  Could not parse accuracy from output")
            accuracy = 0.0
        
        # Load test data to get gold forms and count
        # load_data returns tuples: (lemma, features, form)
        test_data = load_data(str(test_file))
        gold_forms = [item[2] for item in test_data]  # form is at index 2
        total = len(gold_forms)
        correct = int(accuracy * total)
        
        # Read predictions from output file
        output_file_baseline = data_dir / f"{self.lang_code}.out"
        predictions = []
        
        if output_file_baseline.exists():
            with open(output_file_baseline, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            predictions.append(parts[2])  # predicted form
        
        # Save predictions to our output directory
        pred_output_file = Path(self.output_dir) / f"predictions_{self.lang_code}_forward.txt"
        with open(pred_output_file, 'w', encoding='utf-8') as f:
            for pred, gold in zip(predictions, gold_forms):
                f.write(f"{pred}\t{gold}\n")
        
        print(f"\nPredictions saved to {pred_output_file}")
        
        # Evaluate using our evaluation function for consistency
        if predictions:
            results = evaluate_forward(predictions, gold_forms, str(pred_output_file))
            print(f"\n✅ Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        else:
            print(f"\n✅ Accuracy from baseline: {accuracy:.4f}")
            results = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        
        # Save results using new system
        save_nonneural_results(
            output_dir=self.output_dir,
            model_name="nonneural-baseline",
            language=self.lang_code,
            test_metrics=results
        )
        
        print(f"\n✅ Results saved:")
        print(f"   JSON: {self.output_dir}/results_{self.lang_code}.json")
        print(f"   CSV: results/all_results.csv")
        
        return results