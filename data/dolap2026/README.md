# DOLAP 2026 published run: inputs

Copied from the published project (`~/Documents/code/entity_augmentation`, public repo github.com/felipeVsc/llmaugmentation), unchanged:

| File | What it is |
|---|---|
| `test_results.csv` | The 36 published tasks (node ICAO, H, G, metadata with the no-context input, the 9 prompts, difficulty) |
| `gemini_batch_file.jsonl` | Gemini 2.5 Pro batch requests of the with-context run: the exact with-context inputs |
| `ground_truth.csv` | Property values of every airport (key: `icao`), used by the published evaluation |
| `rank_properties.csv` | pop(p) of Airport properties, in percent |
| `ranks_pagerank.csv` | PageRank score of every airport (key: `icao`) |
| `diff_per_task.csv` | Published difficulty per task (decimal commas) |
| `prompts/*.txt` | The 9 published prompt templates (`[PROPERTIES]` = the H property names) |
| `published_results/*_majority_vote.csv` | Voted answers of the 6 proprietary models; `context*` files are the with-context run |
