#!/bin/zsh
# Runs every v5 experiment, one model at a time (Ollama Pro: 3 concurrent requests per account; each model uses 3).
#   zsh scripts/run_v5.sh            (after the gpt-oss:20b Strategy A run with context, if it is still running)
# Progress: results/airports/v5_logs/status.log gets one line per finished model and step.
# Keys come from .env and are never printed.

cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env >/dev/null 2>&1; set +a
LOGS=results/airports/v5_logs
mkdir -p $LOGS
STATUS=$LOGS/status.log
MODELS=(cloud-gpt-oss-20b cloud-gemma4-31b cloud-gpt-oss-120b cloud-mistral-large-3 cloud-deepseek-v4.1-flash)
[[ -n $V5_MODELS ]] && MODELS=(${=V5_MODELS})   # e.g. V5_MODELS="cloud-gpt-oss-20b cloud-gemma4-31b" zsh scripts/run_v5.sh
B_ARGS=(${=V5_B_ARGS})   # extra Strategy B arguments, e.g. V5_B_ARGS="--runs 1" (first run of every task only)

counts() {  # status counts of one model in a results.csv
  venv/bin/python -c "
import csv, collections, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['model'] == sys.argv[2]]
print(len(rows), dict(collections.Counter(r['status'] for r in rows)))" "$1" "$2" 2>/dev/null
}

step() {  # step name, CLI, config, results dir, model, extra arguments
  local name=$1 cli=$2 config=$3 results=$4 model=$5
  local extra=(${@:6})
  echo "$(date '+%F %T') START $name $model ${extra[*]}" >> $STATUS
  venv/bin/$cli run --config $config --models $model $extra >> $LOGS/${name}_${model}.log 2>&1
  venv/bin/$cli run --config $config --models $model --retry-failed $extra >> $LOGS/${name}_${model}.log 2>&1
  echo "$(date '+%F %T') DONE $name $model $(counts $results/results.csv $model)" >> $STATUS
}


if [[ -z $V5_ONLY_B ]]; then   # V5_ONLY_B=1: Strategy B only (Strategy A is complete)
  for m in $MODELS; do
    step A_ctx strategy-a configs/strategy_a_v5_ctx.yaml results/airports/v5_ctx/strategy_a $m
  done
  for m in $MODELS; do
    step A_noctx strategy-a configs/strategy_a_v5_noctx.yaml results/airports/v5_noctx/strategy_a $m
  done
fi
for m in $MODELS; do
  step B strategy-b configs/strategy_b_v5_ctx.yaml results/airports/v5_ctx/strategy_b $m $B_ARGS
done

# Strategy B searches beyond the hourly limit fail (search_partial / retrieval_failed): retry once an hour, up to 12 times.
for i in {1..12}; do
  failed=$(venv/bin/python -c "
import csv
rows = list(csv.DictReader(open('results/airports/v5_ctx/strategy_b/results.csv')))
print(sum(r['status'] in ('api_error', 'timeout', 'retrieval_failed', 'search_partial') for r in rows))")
  [[ $failed == 0 ]] && break
  echo "$(date '+%F %T') B_RETRY round $i: $failed failed runs; waiting for the search limit to reset" >> $STATUS
  sleep 3600
  venv/bin/strategy-b run --config configs/strategy_b_v5_ctx.yaml --models ${(j:,:)MODELS} --retry-failed $B_ARGS >> $LOGS/B_retry.log 2>&1
done
echo "$(date '+%F %T') ALL_DONE" >> $STATUS
