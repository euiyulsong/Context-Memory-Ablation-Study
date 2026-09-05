python3 memory_ablation.py   --limit 1   --workers 8   --max-summary-updates 2
python3 memory_ablation_3exp.py \
  --limit 50 \
  --workers 8 \
  --max-summary-updates 2 \
  --prompt-limit-words 200 \
  --truncate-tokens 128
python3 memory_squad_chunk_ablation.py \
  --limit 50 \
  --workers 20 \
  --prompt-limit-words 100
