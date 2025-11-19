#!/usr/bin/env bash
set -euo pipefail

# run_experiments_on_lang.sh
# Usage: ./run_experiments_on_lang.sh <lang>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
NONNEURAL_PY="$SCRIPT_DIR/baseline/nonneural.py"
NEURAL_TRAIN_PY="$SCRIPT_DIR/baseline/neural/neural-transducer-master/src/train.py"
BYT5_SCRIPT="$SCRIPT_DIR/ByT5/byt5_finetune.py"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 <lang>"
	exit 2
fi

LANG="$1"

# Fixed run configuration (no quick/full modes)
MAX_STEPS=20000
BS=400

RESULT_FILE="$RESULTS_DIR/${LANG}_results.txt"
rm -f "$RESULT_FILE"

echo "Run experiments for language: $LANG" | tee -a "$RESULT_FILE"
echo "Started: $(date -u)" | tee -a "$RESULT_FILE"
echo "max_steps=$MAX_STEPS, batch_size=$BS" | tee -a "$RESULT_FILE"
echo "-----------------------------" | tee -a "$RESULT_FILE"

echo "[1] Non-neural baseline" | tee -a "$RESULT_FILE"
echo "(running: python3 $NONNEURAL_PY -p $DATA_DIR)" | tee -a "$RESULT_FILE"
(
	python3 "$NONNEURAL_PY" -p "$DATA_DIR" 2>&1 | sed -n "/^$LANG:/p" | tee -a "$RESULT_FILE"
) || true
echo "(full non-neural output appended)" >> "$RESULT_FILE"

echo "" >> "$RESULT_FILE"
echo "[2] Neural (neural-transducer) baseline" | tee -a "$RESULT_FILE"
# Run the shipped example wrapper script (required). Fail loudly if missing.
NEURAL_TASK_SCRIPT="$SCRIPT_DIR/baseline/neural/neural-transducer-master/example/sigmorphon2023-shared-tasks/task0-trm.sh"
if [ -f "$NEURAL_TASK_SCRIPT" ]; then
	echo "(running neural example script: $NEURAL_TASK_SCRIPT)" | tee -a "$RESULT_FILE"
	(
		cd "$(dirname "$NEURAL_TASK_SCRIPT")" || exit 1
		# Run the example script as-is (no extra CLI args) per request
		bash "$(basename "$NEURAL_TASK_SCRIPT")" 2>&1 | tee -a "$RESULT_FILE"
	) || { echo "Neural baseline failed (see above)" | tee -a "$RESULT_FILE"; exit 1; }
else
	echo "Required neural example script not found at: $NEURAL_TASK_SCRIPT" | tee -a "$RESULT_FILE"
	echo "Please ensure the neural-transducer example script exists." | tee -a "$RESULT_FILE"
	exit 1
fi

echo "" >> "$RESULT_FILE"
echo "[3] ByT5 forward fine-tuning" | tee -a "$RESULT_FILE"
if [ -f "$BYT5_SCRIPT" ]; then
	echo "(running ByT5 forward via CLI: --train --langs $LANG)" | tee -a "$RESULT_FILE"
	(
		python3 "$BYT5_SCRIPT" --train --langs "$LANG" 2>&1 | tee -a "$RESULT_FILE"
	) || echo "ByT5 forward failed" | tee -a "$RESULT_FILE"
else
	echo "ByT5 script not found at $BYT5_SCRIPT" | tee -a "$RESULT_FILE"
fi

echo "" >> "$RESULT_FILE"
echo "[4] ByT5 inverse fine-tuning" | tee -a "$RESULT_FILE"
if [ -f "$BYT5_SCRIPT" ]; then
	(
		python3 "$BYT5_SCRIPT" --train --langs "$LANG" --inverse 2>&1 | tee -a "$RESULT_FILE"
	) || echo "ByT5 inverse failed" | tee -a "$RESULT_FILE"
fi

echo "" >> "$RESULT_FILE"
echo "[5] ByT5 bidirectional fine-tuning" | tee -a "$RESULT_FILE"
if [ -f "$BYT5_SCRIPT" ]; then
	(
		python3 "$BYT5_SCRIPT" --train --langs "$LANG" --bidirectional 2>&1 | tee -a "$RESULT_FILE"
	) || echo "ByT5 bidirectional failed" | tee -a "$RESULT_FILE"
fi

echo "" >> "$RESULT_FILE"
echo "Finished: $(date -u)" | tee -a "$RESULT_FILE"
echo "Results written to: $RESULT_FILE"

exit 0

