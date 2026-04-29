#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -lt 8 ]; then
  echo "Usage: sh run_fedsparsy_distributed_pytorch.sh MODEL DATASET CLIENT_NUM WORKER_NUM ROUND EPOCH DENSITY LR [optional args...]"
  exit 1
fi

MODEL=$1
DATASET=$2
CLIENT_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
DENSITY=$7
LR=$8

shift 8

PROCESS_NUM=$((WORKER_NUM + 1))
hostname > mpi_host_file

sanitize_tag() {
  local raw="$1"
  local cleaned
  cleaned=$(printf "%s" "$raw" | sed -E 's/--//g; s/[[:space:]]+/_/g; s/[^a-zA-Z0-9._=-]+/-/g; s/_+/_/g; s/^_+|_+$//g')
  if [ -z "$cleaned" ]; then
    cleaned="none"
  fi
  printf "%s" "$cleaned"
}

BASE_TAG="model_${MODEL}__data_${DATASET}__cn_${CLIENT_NUM}__wn_${WORKER_NUM}__r_${ROUND}__e_${EPOCH}__density_${DENSITY}__lr_${LR}"
EXTRA_TAG=$(sanitize_tag "$*")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXP_ROOT="./experiment_records"
mkdir -p "$EXP_ROOT"

EXP_NAME="${TIMESTAMP}__${BASE_TAG}__${EXTRA_TAG}"
EXP_NAME=$(printf "%s" "$EXP_NAME" | cut -c1-220)
EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
LOG_FILE="${EXP_DIR}/train.log"
BATCH_LOG_DIR="${EXP_DIR}/batch_metric_logs"

mkdir -p "$EXP_DIR" "$BATCH_LOG_DIR"

cmd=(
  mpirun -np "$PROCESS_NUM" -hostfile ./mpi_host_file python3 ./main_fedsparsy.py
  --gpu_mapping_file "gpu_mapping.yaml"
  --gpu_mapping_key "mapping_default"
  --model "$MODEL"
  --dataset "$DATASET"
  --client_num_in_total "$CLIENT_NUM"
  --client_num_per_round "$WORKER_NUM"
  --comm_round "$ROUND"
  --epochs "$EPOCH"
  --lr "$LR"
  --target_density "$DENSITY"
)

for arg in "$@"; do
  cmd+=("$arg")
done

# Force each experiment's batch metrics to be written under its experiment directory
cmd+=(--batch_metric_log_dir "$BATCH_LOG_DIR")

{
  echo "timestamp=$TIMESTAMP"
  echo "exp_dir=$EXP_DIR"
  echo "process_num=$PROCESS_NUM"
  echo "model=$MODEL"
  echo "dataset=$DATASET"
  echo "client_num_in_total=$CLIENT_NUM"
  echo "client_num_per_round=$WORKER_NUM"
  echo "comm_round=$ROUND"
  echo "epochs=$EPOCH"
  echo "target_density=$DENSITY"
  echo "lr=$LR"
  echo "extra_args=$*"
} > "${EXP_DIR}/meta.txt"

printf '%q ' "${cmd[@]}" > "${EXP_DIR}/launch_cmd.sh"
printf '\n' >> "${EXP_DIR}/launch_cmd.sh"
chmod +x "${EXP_DIR}/launch_cmd.sh"

echo "[FedSparsy] Logs will be saved to: $LOG_FILE"
echo "[FedSparsy] Batch metrics dir: $BATCH_LOG_DIR"

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

echo "[FedSparsy] Finished with exit code: $exit_code"
echo "[FedSparsy] Experiment dir: $EXP_DIR"
exit "$exit_code"
