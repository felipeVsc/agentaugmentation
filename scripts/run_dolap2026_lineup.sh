#!/bin/sh
# Runs the current lineup on the 36 DOLAP 2026 tasks: first the published protocol (published prompts, 5 runs),
# then the new version's Strategy A (Figure 2 prompts, 3 runs). One model at a time (3 concurrent calls per account).
# Then one --retry-failed pass per config. Log: results/airports/dolap2026/logs/status.log
cd "$(dirname "$0")/.."
LOG=results/airports/dolap2026/logs
MODELS="cloud-gemma4-31b cloud-gpt-oss-20b cloud-gpt-oss-120b cloud-deepseek-v4.1-flash"
for cfg in strategy_a_dolap2026 strategy_a_dolap2026_newprompts; do
  for m in $MODELS; do
    echo "$(date '+%F %T') start $cfg $m" >> $LOG/status.log
    venv/bin/strategy-a run --config configs/$cfg.yaml --models $m > $LOG/${cfg}_$m.log 2>&1
    echo "$(date '+%F %T') end   $cfg $m exit=$?" >> $LOG/status.log
  done
  venv/bin/strategy-a run --config configs/$cfg.yaml --models $(echo $MODELS | tr ' ' ',') --retry-failed > $LOG/${cfg}_retry.log 2>&1
  echo "$(date '+%F %T') retry $cfg exit=$?" >> $LOG/status.log
done
echo "$(date '+%F %T') ALL DONE" >> $LOG/status.log
