#!/bin/zsh
# Strategy B on football-v1-ctx-idmix (context on), as scripts/run_v5.sh and run_v5_memory.sh for airports v5: for
# each model, the search_evidence runs (Ollama Pro: 3 concurrent requests per account; each model uses 3), hourly
# retries of the searches that hit the Ollama limits, then the search + memory replay (no search).
# Searches and model calls use OLLAMA_API_KEY; searches are 30 s apart (search.min_interval_s), which kept
# airports v5 free of 429s at ~47 runs/h.
#   zsh scripts/run_football_b.sh
#   FB_MODELS="cloud-gpt-oss-20b" zsh scripts/run_football_b.sh      # some models only
#   FB_SKIP_MEMORY=1 zsh scripts/run_football_b.sh                   # search_evidence only
# Progress: results/football/b_logs/status.log gets one line per finished model and step.
# Context off is not run for now (as for airports, where Strategy B ran on context on only); FB_SETS selects the
# sets once configs/strategy_b_football_idmix_noctx*.yaml exist.
# Keys come from .env and are never printed.

cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env >/dev/null 2>&1; set +a
LOGS=results/football/b_logs
mkdir -p $LOGS
STATUS=$LOGS/status.log
MODELS=(cloud-mistral-large-3 cloud-gpt-oss-20b cloud-gemma4-31b cloud-gpt-oss-120b cloud-deepseek-v4.1-flash)
[[ -n $FB_MODELS ]] && MODELS=(${=FB_MODELS})
SETS=(ctx)
[[ -n $FB_SETS ]] && SETS=(${=FB_SETS})
typeset -A CONFIG DIR
CONFIG=(ctx configs/strategy_b_football_idmix noctx configs/strategy_b_football_idmix_noctx)
DIR=(ctx results/football/v1_ctx_idmix noctx results/football/v1_noctx_idmix)

counts() {  # status counts of one model in a results.csv
  venv/bin/python -c "
import csv, collections, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['model'] == sys.argv[2]]
print(len(rows), dict(collections.Counter(r['status'] for r in rows)))" "$1" "$2" 2>/dev/null
}

failed() {  # runs of one model to retry in a results.csv
  venv/bin/python -c "
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['model'] == sys.argv[2]]
print(sum(r['status'] in ('api_error', 'timeout', 'retrieval_failed', 'search_partial') for r in rows))" "$1" "$2" 2>/dev/null || echo 0
}

# per model: search_evidence, its hourly retries, then its memory replay, so each model is complete before the next
# (mistral-large-3 first: Ollama removes it in a few days)
for s in $SETS; do
  for m in $MODELS; do
    echo "$(date '+%F %T') START B_$s $m" >> $STATUS
    venv/bin/strategy-b run --config $CONFIG[$s].yaml --models $m >> $LOGS/B_${s}_${m}.log 2>&1
    venv/bin/strategy-b run --config $CONFIG[$s].yaml --models $m --retry-failed >> $LOGS/B_${s}_${m}.log 2>&1
    # searches beyond the Ollama limits fail (search_partial / retrieval_failed): retry once an hour, up to 12 times
    for i in {1..12}; do
      n=$(failed $DIR[$s]/strategy_b/results.csv $m)
      [[ $n == 0 ]] && break
      echo "$(date '+%F %T') B_${s}_RETRY $m round $i: $n failed runs; waiting for the search limit to reset" >> $STATUS
      sleep 3600
      venv/bin/strategy-b run --config $CONFIG[$s].yaml --models $m --retry-failed >> $LOGS/B_${s}_${m}.log 2>&1
    done
    echo "$(date '+%F %T') DONE B_$s $m $(counts $DIR[$s]/strategy_b/results.csv $m)" >> $STATUS
    if [[ -z $FB_SKIP_MEMORY ]]; then
      echo "$(date '+%F %T') START BM_$s $m" >> $STATUS
      replay=(venv/bin/strategy-b replay --config $CONFIG[$s]_memory.yaml
              --source-dir $DIR[$s]/strategy_b --source-variant search_evidence --models $m)
      $replay >> $LOGS/BM_${s}_${m}.log 2>&1
      $replay --retry-failed >> $LOGS/BM_${s}_${m}.log 2>&1
      echo "$(date '+%F %T') DONE BM_$s $m $(counts $DIR[$s]/strategy_b_memory/results.csv $m)" >> $STATUS
    fi
  done
done
echo "$(date '+%F %T') ALL_DONE" >> $STATUS
