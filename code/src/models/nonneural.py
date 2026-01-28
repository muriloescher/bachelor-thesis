"""Non-neural baseline wrapper."""
import os
import subprocess
from pathlib import Path
from ..utils import resolve_path, ensure_dir


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
    
    def predict(self, test=False, checkpoint=None):
        """
        Run non-neural baseline prediction.
        
        The baseline expects files named: {lang}.trn, {lang}.dev, {lang}.tst
        in a single directory and outputs {lang}.out
        
        Args:
            test: If True, evaluate on test set; otherwise use dev set
            checkpoint: Ignored (for API compatibility)
        """
        print(f"\n{'='*60}")
        print(f"Running non-neural baseline for {self.lang_code}")
        print(f"{'='*60}")
        
        # Get data paths - use unimorph data (3-column format)
        data_config = self.lang_config['data']
        train_file = resolve_path(data_config['unimorph']['train'])
        data_dir = train_file.parent
        
        print(f"Data directory: {data_dir}")
        print(f"Language code: {self.lang_code}")
        print(f"Evaluation mode: {'test' if test else 'dev'}")
        
        # Run the baseline script
        # It expects: --path <dir_with_trailing_slash> --lang <code> [--test] --output
        cmd = [
            'python3',
            str(self.baseline_script),
            '--path', str(data_dir) + '/',
            '--lang', self.lang_code,
        ]
        
        if test:
            cmd.append('--test')
        cmd.append('--output')
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"Non-neural baseline failed with return code {result.returncode}")
        
        # Move output file to our output directory
        expected_output = data_dir / f"{self.lang_code}.out"
        output_file = Path(self.output_dir) / f"{self.lang_code}_{'test' if test else 'dev'}.out"
        
        if expected_output.exists():
            import shutil
            shutil.copy(str(expected_output), str(output_file))
            print(f"\nPredictions saved to {output_file}")
        
        return output_file