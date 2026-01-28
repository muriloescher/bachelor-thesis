# Morphological Inflection/Analysis Experiments

This repository contains code for training and evaluating models on morphological inflection and analysis tasks, supporting multiple languages and model architectures.

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

The repository uses a unified runner script (`run.py`) with YAML configuration files:

```bash
# List available models and languages
python run.py --list-models
python run.py --list-languages

# Train a model
python run.py --model byt5_bidirectional --language por --train

# Make predictions with a trained model
python run.py --model byt5_bidirectional --language por --predict

# Train and predict in one command
python run.py --model byt5_forward --language por --train --predict

# Multiple languages at once
python run.py --model byt5_inverse --language por,eng,ita --train --predict
```

## Project Structure

```
code/
├── run.py                      # Main entry point
├── configs/                    # Configuration files
│   ├── models/                 # Model configurations
│   │   ├── byt5_forward.yaml
│   │   ├── byt5_inverse.yaml
│   │   ├── byt5_bidirectional.yaml
│   │   ├── byt5_context.yaml
│   │   ├── llm.yaml
│   │   ├── nonneural.yaml
│   │   └── neural_baseline.yaml
│   └── languages/              # Language configurations
│       ├── por.yaml
│       ├── eng.yaml
│       └── ita.yaml
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   ├── data/                   # Data loading utilities
│   └── utils/                  # Shared utilities
├── data/                       # Datasets
│   ├── unimorph/              # UniMorph data
│   └── ud/                    # Universal Dependencies data
└── results/                    # Output directory (created automatically)
```

## Available Models

### ByT5 Models

**Forward (Inflection)**: `byt5_forward`
- Task: lemma + features → inflected form
- Standard morphological inflection

**Inverse (Analysis)**: `byt5_inverse`
- Task: inflected form → lemma + features
- Morphological analysis/segmentation

**Bidirectional**: `byt5_bidirectional`
- Task: 50/50 mix of forward and inverse
- Evaluates both directions

**Context-Aware**: `byt5_context`
- Task: Bidirectional with sentential context
- Uses 4-column data format (lemma, features, form, context)

### Baseline Models

**Non-Neural Baseline**: `nonneural`
- Classical string alignment approach

**Neural Baseline**: `neural_baseline`
- Neural transducer model

**LLM Baseline**: `llm`
- Large language models (Llama, Qwen) via OpenRouter API

## Configuration

### Model Configurations

Model configs (`configs/models/*.yaml`) define:
- Model architecture and hyperparameters
- Training settings (batch size, learning rate, epochs)
- Prediction settings
- Task-specific prompts

Example (`byt5_bidirectional.yaml`):
```yaml
model_type: byt5
task: bidirectional

model_name: google/byt5-small
output_dir_template: "results/byt5-bidir-{lang}"

training:
  per_device_train_batch_size: 8
  num_train_epochs: 3
  learning_rate: 1.0e-4
  # ... more settings

prompts:
  forward: "Generate the inflected form for: {lemma} {features}"
  inverse: "Generate the lemma and morphological tags for the following inflected verb: {form}"
```

### Language Configurations

Language configs (`configs/languages/*.yaml`) define:
- Language metadata
- Data file paths
- Data format specifications

Example (`por.yaml`):
```yaml
language:
  code: por
  name: Portuguese
  family: Romance

data:
  source: unimorph
  train: "data/unimorph/por.trn"
  dev: "data/unimorph/por.dev"
  test: "data/unimorph/por.tst"
  
  # Alternative UD data (for context models)
  ud:
    train: "data/ud/pt_verbs_context.trn"
    dev: "data/ud/pt_verbs_context.dev"
    test: "data/ud/pt_verbs_context.tst"

format:
  columns: 3  # lemma, features, form
  separator: "\t"
  has_context: false
```

## Common Workflows

### 1. Training a New Model

```bash
# Train ByT5 bidirectional model for Portuguese
python run.py --model byt5_bidirectional --language por --train
```

Output:
- Model checkpoints: `results/byt5-bidir-por/checkpoint-*/`
- Final model: `results/byt5-bidir-por/`

### 2. Evaluating a Trained Model

```bash
# Predict with trained model
python run.py --model byt5_bidirectional --language por --predict
```

Output:
- Predictions: `results/byt5-bidir-por/predictions_por_*.txt`
- Each prediction file includes evaluation metrics

### 3. Inverse-Only Prediction (Bidirectional Models)

```bash
# Only evaluate inverse direction
python run.py --model byt5_bidirectional --language por --predict --inverse-only
```

Useful for comparing baseline vs. context-aware models on analysis tasks.

### 4. Using Specific Checkpoints

```bash
# Predict using checkpoint-3750
python run.py --model byt5_inverse --language por --predict --checkpoint 3750
```

### 5. Multi-Language Experiments

```bash
# Train and evaluate on multiple languages
python run.py --model byt5_forward --language por,eng,ita --train --predict
```

### 6. Context-Aware Models

```bash
# Train context-aware model (uses UD data)
python run.py --model byt5_context --language por --train --predict
```

This automatically uses the UD data paths specified in the language config.

## Adding New Languages

1. Create a language config file in `configs/languages/`:

```yaml
# configs/languages/fra.yaml
language:
  code: fra
  name: French
  family: Romance

data:
  source: unimorph
  train: "data/unimorph/fra.trn"
  dev: "data/unimorph/fra.dev"
  test: "data/unimorph/fra.tst"

format:
  columns: 3
  separator: "\t"
  has_context: false
```

2. Add your data files to `data/unimorph/` or `data/ud/`

3. Run experiments:

```bash
python run.py --model byt5_bidirectional --language fra --train --predict
```

## Data Format

### Standard Format (3 columns)

```
lemma<TAB>features<TAB>form
walk<TAB>V;PST<TAB>walked
run<TAB>V;PST<TAB>ran
```

### Context Format (4 columns)

```
lemma<TAB>features<TAB>form<TAB>context
walk<TAB>V;PST<TAB>walked<TAB>I walked to the store yesterday.
run<TAB>V;PST<TAB>ran<TAB>She ran faster than everyone.
```

### Conversion from Universal-Dependencies to Unimorph format

```
cd code/data/ud-compatibility-master
python3 -m ud_compatibility.marry convert --ud ../ud/NAME_OF_DATASET.conllu -l LANGUAGE_TAG_IF_GIVEN
```
Result is saved in ```data/ud```. 
The name goes for ex. from ```pt_petrogold-ud-train.connlu``` to ```pt_petrogold-um-train.connlu``` (changes "ud" substring to "um").

If the order each data block is that it starts with ```# text ``` instead of ```# sent_id```, run:
```
cd code/data/ud
python3 swap_metadata.py NAME_OF_DATASET.conllu
```

At last, to extract the tags, now in Unimorph format, from the UD data structure, run: 
```
cd code/data/ud
python3 conllu_to_unimorph.py NAME_OF_DATASET.conllu TARGET_FILE
```
Example: 
```
cd code/data/ud
python3 conllu_to_unimorph.py pt_petrogold-um-test.conllu por.tst
```

## Evaluation Metrics

### Forward Task (Inflection)
- **Accuracy**: Exact match percentage

### Inverse Task (Analysis)
- **Lemma Accuracy**: Exact lemma match percentage
- **Lemma Levenshtein**: Mean edit distance for lemmas
- **MSD Accuracy**: Exact morphological tag set match
- **MSD Micro-F1**: Precision/recall/F1 over all tags

## Output Files

Each prediction run creates:

1. **Predictions file**: `predictions_{lang}_{direction}.txt`
   - Format: `input<TAB>prediction<TAB>gold`
   
2. **Metrics**: Appended to the predictions file
   - Accuracy scores
   - Example mismatches for debugging

## Reproducibility

To reproduce all results from scratch:

```bash
# Train all ByT5 variants on all languages
for model in byt5_forward byt5_inverse byt5_bidirectional; do
    for lang in por eng ita; do
        python run.py --model $model --language $lang --train --predict
    done
done

# Train context-aware model for Portuguese
python run.py --model byt5_context --language por --train --predict
```

All experiments use fixed random seeds (seed=42) for reproducibility.

## Environment Variables

For LLM experiments, create a `.env` file in `code/`:

```
OPENROUTER_API_KEY=your_api_key_here
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in model config:
```yaml
training:
  per_device_train_batch_size: 4  # reduced from 8
```

### Model Not Found
Check that training completed successfully:
```bash
ls results/byt5-bidir-por/
```

Should contain model files or a `checkpoint-*/` directory.

### Data File Not Found
Verify paths in language config are relative to `code/` directory:
```yaml
data:
  train: "data/unimorph/por.trn"  # Relative to code/
```

## Advanced Usage

### Custom Model Configurations

Copy an existing config and modify:

```bash
cp configs/models/byt5_bidirectional.yaml configs/models/byt5_custom.yaml
# Edit byt5_custom.yaml with your settings
python run.py --model byt5_custom --language por --train
```

### Using Pre-trained Checkpoints

Models are saved in `results/{model}-{lang}/`. To use a pre-trained model:

```bash
# Just run prediction (skips training)
python run.py --model byt5_bidirectional --language por --predict
```

The runner automatically finds the trained model in the results directory.

## Citation

If you use this code, please cite the relevant papers for each component:
- ByT5: [Xue et al., 2021]
- UniMorph: [McCarthy et al., 2020]
- Universal Dependencies: [Nivre et al., 2020]

## License

[Add your license information here]

## Contact

[Add your contact information here]
