# Bachelor Thesis – Reproducibility README

This repository contains:

- `code/`: the experiment code for morphological **inflection** (forward) and **analysis** (inverse) across multiple languages and model types (baselines, ByT5, LLM baselines).
- `thesis/`: the LaTeX source used to write the thesis.

The main goal of this README is to provide *everything needed to reproduce the results* (from prepared data → running the unified runner → generating the thesis tables).

---

## Project structure (high-level)

```
bachelor-thesis/
├── README.md
├── code/
│   ├── run.py                         # unified runner (train/predict)
│   ├── requirements.txt               # Python dependencies
│   ├── configs/
│   │   ├── models/                    # model configs (ByT5, baselines, LLM)
│   │   └── languages/                 # language + dataset path configs
│   ├── data/
│   │   ├── unimorph/                  # 3-col data: lemma, features, form
│   │   └── ud/                        # 4-col data: lemma, features, form, context
│   ├── results/                       # produced experiment outputs + CSV log
│   └── src/                           # implementations (models, data, utils)
└── thesis/
  ├── main.tex                       # LaTeX entrypoint
  ├── chapters/                      # thesis chapters
  ├── tables/                        # generated LaTeX table fragments
  └── build/                         # compiled PDF output (latexmk)
```

## 1) Environment setup

### System requirements

- **Python**: tested with **Python 3.12** (see `code/pyvenv.cfg`).
- (Recommended) **GPU** for ByT5 training: an NVIDIA GPU with CUDA speeds things up significantly. CPU-only runs work for small tests but can be very slow.
- For the **neural baseline** and some convenience scripts, you need a **bash** environment.

### Create a virtual environment (Linux/macOS)

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Sanity check

From `code/`:

```bash
python run.py --list-models
python run.py --list-languages
```

If those commands work, your environment and config discovery are set up correctly.

---

## 2) Data layout and (re)generation

The unified runner expects the following prepared data files:

- UniMorph-like (3 columns): `code/data/unimorph/<lang>.{trn,dev,tst}`
  - Format: `lemma<TAB>features<TAB>form`
- UD-with-context (4 columns): `code/data/ud/<lang>.{trn,dev,tst}`
  - Format: `lemma<TAB>features<TAB>form<TAB>context_sentence`

This repo already contains prepared files in those locations. If you want to **regenerate them from the included originals**, use the scripts below.

### 2.1 UniMorph splits (train/dev/test)

The repository includes original UniMorph files at:

- `code/data/unimorph/original/<lang>`

To regenerate splits for one language:

```bash
cd code
python prepare_unimorph_data.py por data/unimorph/original/por
```

This creates:

- `code/data/unimorph/por.trn`
- `code/data/unimorph/por.dev`
- `code/data/unimorph/por.tst`

The split is **lemma-based** (no lemma leakage across splits) and uses a fixed seed by default.

To regenerate all languages in one go, there is also:

- `code/prepare_all_unimorph.sh`

Note: this is a bash script, so on Windows it is easiest to run it in **WSL**.

### 2.2 UD → (lemma, tags, form, context) files

Prepared UD `.trn/.dev/.tst` files live in `code/data/ud/`.

The folder also contains the original treebanks as `.conllu` files (e.g. `pt_petrogold-ud-train.conllu`) and conversion helpers:

- `code/data/ud/conllu_to_unimorph.py` (extracts the first verb per sentence, reorders tags, outputs 4 columns)
- `code/data/ud/conllu_extract_ud.py` (extracts the first verb per sentence, keeps UD-style tags)
- `code/data/ud/swap_metadata.py` (fixes ordering of `# sent_id` / `# text` lines if needed)

Example conversion (Portuguese train split):

```bash
cd code/data/ud
python conllu_to_unimorph.py pt_petrogold-ud-train.conllu por.trn
python conllu_to_unimorph.py pt_petrogold-ud-dev.conllu   por.dev
python conllu_to_unimorph.py pt_petrogold-ud-test.conllu  por.tst
```

If your `.conllu` blocks start with `# text = ...` before `# sent_id = ...`, you can normalize them:

```bash
cd code/data/ud
python swap_metadata.py pt_petrogold-ud-train.conllu
```

---

## 3) Running experiments (unified runner)

All experiments are run via the **unified runner**:

- `code/run.py`

It uses YAML configs from:

- `code/configs/models/*.yaml`
- `code/configs/languages/*.yaml`

### 3.1 Explore available configs

From `code/`:

```bash
python run.py --list-models
python run.py --list-languages
python run.py --help
```

### 3.2 Common commands

Run **training** and/or **prediction**:

```bash
# Train only
python run.py --model byt5_forward --language por --train

# Predict only (requires a trained model already in results/)
python run.py --model byt5_forward --language por --predict

# Train + predict
python run.py --model byt5_forward --language por --train --predict

# Multiple languages (comma-separated)
python run.py --model byt5_forward --language por,eng,ita --train --predict
```

Use a specific checkpoint for prediction (ByT5 / neural baseline only):

```bash
python run.py --model byt5_inverse --language por --predict --checkpoint 3750
```

Bidirectional ByT5 models can predict only the inverse direction:

```bash
python run.py --model byt5_bidirectional --language por --predict --inverse-only
```

### 3.3 Model names used in this repo

These correspond to the YAML files in `code/configs/models/`:

- `byt5_forward`
- `byt5_inverse`
- `byt5_bidirectional`
- `byt5_context` (bidirectional + context, uses UD data)
- `nonneural` (non-neural baseline, prediction only)
- `neural` (neural transducer baseline)
- `llm_llama` (OpenRouter Llama baseline, inverse only)
- `llm_qwen` (OpenRouter Qwen baseline, inverse only)

### 3.4 LLM baselines (OpenRouter)

LLM baselines require an API key in an environment variable.

Create a file `code/.env`:

```env
OPENROUTER_API_KEY=your_key_here
```

Then run (from `code/`):

```bash
python run.py --model llm_llama --language por --predict
python run.py --model llm_qwen  --language por --predict
```

Notes:

- LLM experiments are *not guaranteed to be perfectly deterministic* across time/providers even with `temperature: 0.0`.
- The LLM prompts use a small number of dev-set examples (few-shot) taken from the language config.

### 3.5 Neural baseline (SIGMORPHON transducer)

The neural baseline wrapper calls a bash script in:

- `code/baselines/neural/neural-transducer-master/example/sigmorphon2023-shared-tasks/task0-trm.sh`

On Windows, the simplest way is to run this part in **WSL**.

From `code/`:

```bash
python run.py --model neural --language por --train --predict
```

---

## 4) Outputs: where results go

All results are written under `code/results/`.

Key files:

- `code/results/all_results.csv`: a *central log* of all experiments (rows are appended).
- `code/results/<experiment-dir>/results_<lang>.json`: one JSON per language per experiment directory.

To inspect/clean/backup results:

```bash
cd code
# Inspect what was produced
ls -la results
find results -name 'results_*.json' -print

# Preview the central CSV (if present)
if [ -f results/all_results.csv ]; then head -n 5 results/all_results.csv; fi

# Backup results to a timestamped folder
stamp=$(date +"%Y%m%d_%H%M%S")
cp -r results "results_backup_${stamp}"

# OPTIONAL: clean JSON + move CSV aside for a fresh run
# if [ -f results/all_results.csv ]; then mv results/all_results.csv "results/all_results_old_${stamp}.csv"; fi
# find results -name 'results_*.json' -delete
```

---

## 5) Reproducing the thesis tables

The thesis uses pre-rendered LaTeX table fragments in:

- `thesis/tables/*.tex`

They can be regenerated from a results CSV using:

- `code/generate_thesis_results_tables.py`

### 5.1 Generate tables from the curated snapshot (matches thesis submission)

By default the script reads:

- `code/results/end_results.csv`

Run from the repo root:

```bash
python code/generate_thesis_results_tables.py
```

This writes:

- `thesis/tables/results_inflection.tex`
- `thesis/tables/results_analysis_um_lemma.tex`
- `thesis/tables/results_analysis_um_msd.tex`
- `thesis/tables/results_analysis_ud_lemma.tex`
- `thesis/tables/results_analysis_ud_msd.tex`

### 5.2 Generate tables from a fresh full run

After running experiments from scratch (producing `code/results/all_results.csv`), generate tables from that log:

```bash
python code/generate_thesis_results_tables.py --csv code/results/all_results.csv
```

If multiple rows exist for the same experiment key, the script selects the **most recent** by timestamp.

---

## 6) Building the thesis PDF

The LaTeX project is in `thesis/`.

If you use VS Code, installing **LaTeX Workshop** is the easiest route.

Command line (requires a LaTeX distribution + `latexmk`):

```bash
cd thesis
# Option A: use the Makefile (if available in your environment)
make

# Option B: latexmk directly
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

The output PDF is typically:

- `thesis/build/main.pdf`

---