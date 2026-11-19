#!/bin/zsh
# Strategy B search + memory on airports-v5-ctx: replays the synthesis call of every finished search_evidence run
# with synth_memory.v2.txt, one model at a time (no search; Ollama Pro: 3 concurrent requests per account).
#   zsh scripts/run_v5_memory.sh        V5_MODELS="cloud-gpt-oss-20b" zsh scripts/run_v5_memory.sh
# Progress: one line per model in results/airports/v5_logs/status.log (BM = B memory). Keys come from .env.

cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env >/dev/null 2>&1; set +a
LOGS=results/airports/v5_logs
STATUS=$LOGS/status.log
RESULTS=results/airports/v5_ctx/strategy_b_memory
MODELS=(cloud-gpt-oss-20b cloud-gemma4-31b cloud-gpt-oss-120b cloud-mistral-large-3 cloud-deepseek-v4.1-flash)
[[ -n $V5_MODELS ]] && MODELS=(${=V5_MODELS})
REPLAY=(venv/bin/strategy-b replay --config configs/strategy_b_v5_ctx_memory.yaml
        --source-dir results/airports/v5_ctx/strategy_b --source-variant search_evidence)

counts() {  # status counts of one model in a results.csv
  venv/bin/python -c "
import csv, collections, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['model'] == sys.argv[2]]
print(len(rows), dict(collections.Counter(r['status'] for r in rows)))" "$1" "$2" 2>/dev/null
}

for m in $MODELS; do
  echo "$(date '+%F %T') START BM $m" >> $STATUS
  $REPLAY --models $m >> $LOGS/BM_${m}.log 2>&1
  $REPLAY --models $m --retry-failed >> $LOGS/BM_${m}.log 2>&1
  echo "$(date '+%F %T') DONE BM $m $(counts $RESULTS/results.csv $m)" >> $STATUS
done
echo "$(date '+%F %T') BM_ALL_DONE" >> $STATUS
