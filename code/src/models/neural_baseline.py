"""Neural transducer baseline wrapper for the new system."""
import os
import sys
import subprocess
from pathlib import Path

from ..utils import ensure_dir, resolve_path


class NeuralBaselineModel:
    """Neural transducer baseline model wrapper."""
    
    def __init__(self, config, lang_config, mode='train'):
        """
        Initialize neural baseline.
        
        Args:
            config: Model configuration dict
            lang_config: Language configuration dict
            mode: 'train' or 'predict'
        """
        self.config = config
        self.lang_config = lang_config
        self.mode = mode
        self.lang_code = lang_config['language']['code']
        
        # Get baseline directory path
        project_root = Path(__file__).parent.parent.parent
        self.baseline_dir = project_root / 'baselines' / 'neural' / 'neural-transducer-master'
        self.train_script = self.baseline_dir / 'src' / 'train.py'
        self.decode_script = self.baseline_dir / 'src' / 'sigmorphon19-task1-decode.py'
        
        if not self.baseline_dir.exists():
            raise FileNotFoundError(f"Neural baseline directory not found: {self.baseline_dir}")
        
        # Get output directory
        output_template = config.get('output_dir_template', 'results/neural-{lang}')
        self.output_dir = output_template.format(lang=self.lang_code)
        ensure_dir(self.output_dir)
        
        # Determine data source (always use unimorph for neural baseline)
        self.data_source = 'unimorph'
    
    def train(self):
        """Train the model."""
        print(f"\n{'='*60}")
        print(f"Training neural baseline for {self.lang_code}")
        print(f"{'='*60}")
        
        # Get data paths
        data_config = self.lang_config['data']
        data_source_cfg = data_config.get(self.data_source, data_config['unimorph'])
        
        train_file = resolve_path(data_source_cfg['train'])
        dev_file = resolve_path(data_source_cfg['dev'])
        
        # Training configuration
        train_cfg = self.config.get('training', {})
        
        # Build command
        cmd = [
            sys.executable,
            str(self.train_script),
            '--dataset', str(train_file),
            '--dev', str(dev_file),
            '--model', str(Path(self.output_dir) / 'model'),
            '--epochs', str(train_cfg.get('epochs', 20)),
            '--patience', str(train_cfg.get('patience', 5)),
            '--batch-size', str(train_cfg.get('batch_size', 20)),
            '--dropout', str(train_cfg.get('dropout', 0.2)),
            '--enc-layers', str(train_cfg.get('enc_layers', 1)),
            '--dec-layers', str(train_cfg.get('dec_layers', 1)),
            '--enc-hidden-size', str(train_cfg.get('enc_hidden_size', 256)),
            '--dec-hidden-size', str(train_cfg.get('dec_hidden_size', 256)),
            '--embed-size', str(train_cfg.get('embed_size', 256)),
        ]
        
        # Add optional arguments
        if train_cfg.get('use_attention', True):
            cmd.append('--attention')
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run training
        result = subprocess.run(
            cmd,
            cwd=str(self.baseline_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"Training completed successfully")
            print(f"Model saved to {self.output_dir}")
        else:
            print(f"Training failed:")
            print(result.stderr)
            raise RuntimeError("Neural baseline training failed")
        
        return self.output_dir
    
    def predict(self, checkpoint=None):
        """
        Make predictions on test set.
        
        Args:
            checkpoint: Specific checkpoint to load (not used for this baseline)
        """
        print(f"\n{'='*60}")
        print(f"Predicting with neural baseline for {self.lang_code}")
        print(f"{'='*60}")
        
        # Get data paths
        data_config = self.lang_config['data']
        data_source_cfg = data_config.get(self.data_source, data_config['unimorph'])
        
        test_file = resolve_path(data_source_cfg['test'])
        model_path = Path(self.output_dir) / 'model'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Please train first.")
        
        # Output file
        output_file = Path(self.output_dir) / f"predictions_{self.lang_code}.txt"
        
        # Prediction configuration
        pred_cfg = self.config.get('prediction', {})
        
        # Build command
        cmd = [
            sys.executable,
            str(self.decode_script),
            '--dataset', str(test_file),
            '--model', str(model_path),
            '--output', str(output_file),
            '--beam-size', str(pred_cfg.get('beam_size', 5)),
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run prediction
        result = subprocess.run(
            cmd,
            cwd=str(self.baseline_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"Predictions saved to {output_file}")
            
            # Parse accuracy from output if available
            results = {}
            for line in result.stdout.split('\n'):
                if 'accuracy' in line.lower():
                    try:
                        acc = float(line.split(':')[-1].strip().strip('%')) / 100
                        results['accuracy'] = acc
                        print(f"Accuracy: {acc:.4f}")
                    except:
                        pass
            
            return results
        else:
            print(f"Prediction failed:")
            print(result.stderr)
            raise RuntimeError("Neural baseline prediction failed")
