# Findings for the paper: Strategy A and B on airports and football

Notes to support writing the paper, from the completed experiments (9 Nov 2026):
- **airports v5:** 243 tasks with context on, 180 with context off.
- **football v1 idmix:** 239 tasks with context on, 233 with context off.
- **Same setup on both:** 5 models, 3 runs. Strategy A uses 9 prompts on both sets. Strategy B (search, and search + memory) runs on the context-on sets.

Every number can be recomputed from the CSV files listed at the end. The charts and short paragraphs behind each point are in the four reports:

- `notebooks/paper_report_football_strategy_a.html`
- `notebooks/paper_report_football_strategy_b.html`
- `notebooks/paper_report_football_a_vs_b.html`
- `notebooks/paper_report_both_datasets.html`

The figures are in `notebooks/figures/paper_reports/<report>/`, as PNG and PDF.

**Scoring.** One voted answer per (model, configuration, task). Each F1 is given with abstentions counted and ignored:
- **F1** = 2c / (2c + 2w + a);
- **F1 ign** = 2c / (2c + w).

Here c, w and a are the counts of correct, wrong and abstained answers. Intervals are 95% bootstrap intervals over tasks. Task-by-task comparisons use an exact McNemar test.

---

## 1. Main results

**Best configuration per model, context on.** In the table:
- **A best** is the model's best prompt by F1, chosen on the same tasks, so it is an optimistic upper bound for A;
- **B mem** is B search + memory.

Each cell is F1 / F1 ign.

| Model | Airports: A best | Airports: B search | Airports: B mem | Football: A best | Football: B search | Football: B mem |
|---|---|---|---|---|---|---|
| gpt-oss:20b | 0.16 / 0.38 | 0.25 / 0.50 | 0.26 / 0.46 | 0.25 / 0.43 | 0.14 / 0.36 | 0.19 / 0.43 |
| gemma4:31b | 0.19 / 0.48 | 0.15 / 0.46 | 0.21 / 0.46 | 0.29 / 0.57 | 0.13 / 0.43 | 0.19 / 0.50 |
| gpt-oss:120b | 0.24 / 0.40 | 0.32 / 0.62 | 0.32 / 0.59 | 0.19 / 0.40 | 0.16 / 0.40 | 0.20 / 0.48 |
| mistral-large-3 | 0.22 / 0.39 | 0.29 / 0.54 | 0.26 / 0.48 | 0.41 / 0.65 | 0.17 / 0.42 | 0.23 / 0.48 |
| deepseek-v4.1-flash | 0.23 / 0.43 | 0.16 / 0.47 | 0.24 / 0.53 | 0.43 / 0.63 | 0.07 / 0.45 | 0.12 / 0.51 |

### Finding 1. Whether web search helps depends on the dataset

The two datasets reverse the conclusion:
- **Airports:** a B variant has the highest F1 for 4 of 5 models, and deepseek ties.
- **Football:** A best has the highest F1 for 4 of 5 models, and only gpt-oss:120b ties (0.19 vs 0.20).

Task-by-task results on football:
- B + memory is significantly worse than A best for gemma4 (−0.10 F1), mistral (−0.18) and deepseek (−0.31).
- It is not different for the two gpt-oss models.

Task-by-task results on airports:
- B + memory beats A best for gpt-oss:20b (+0.10) and gpt-oss:120b (+0.09).

**Why it matters.** The agent, the models and the prompts are identical on both datasets, so the difference comes from the data. Airport facts are published in a few stable reference sources: Wikipedia and OurAirports hold most of the cited evidence. Player statistics depend on season and competition, and are spread over aggregators that disagree.

**Suggested framing.** Retrieval augmentation is not a uniform improvement. Its value depends on how *addressable* and *stable* the target facts are on the web.

### Finding 2. Entity identification is the bottleneck of Strategy B

The share of runs whose search plan names the right node:
- **Airports:** 8–20% of runs.
- **Football:** 10–12% of runs. This happens almost only when the player's name is in the given properties: 92.6% of the runs with a name, against 0.2% without one.

Without the name, the planner searches for generic statistics ("goalkeeper clean sheets"), and 84% of those runs abstain.

Even with the right player, football answers are mostly wrong: 11% correct, 45% wrong and 44% abstained when the name is given. The page is found, but the value read from it belongs to another season or competition.

On airports, the given properties (city, coordinates, codes) are usable search keys. On football they are season statistics, which are not.

### Finding 3. On football, the ranking of models by F1 measures their willingness to answer

With the 9 prompts pooled, football F1 ranges from 0.11 to 0.38, while F1 ign stays within 0.57–0.62:
- three models (gpt-oss:20b, gemma4, gpt-oss:120b) abstain on 76–85% of the tasks;
- mistral and deepseek abstain on 26–29%.

On airports the spread between models is smaller.

**Implication.** Report both F1 and F1 ign. F1 rewards answering and F1 ign rewards accuracy, and the two give different model rankings.

### Finding 4. The model matters more than the prompt for F1; the prompt matters more for accuracy on football

Spread of the mean score, context on:

| | F1 over models | F1 over prompts | F1 ign over models | F1 ign over prompts |
|---|---|---|---|---|
| Airports | 0.07 | 0.01 | 0.19 | 0.11 |
| Football | 0.28 | 0.09 | 0.10 | 0.22 |

In other words:
- **Prompts** mainly change how accurate the answers are.
- **Models** mainly change how many tasks are answered.

Prompt rankings transfer poorly between datasets: Spearman ρ between the airports and football scores of the 45 model × prompt configurations is 0.47 for F1 and 0.01 for F1 ign.
- **Football:** few-shot prompts are clearly best (5-shot mean rank 2.8, 3-shot 3.2) and zero-shot is worst (7.8).
- **Airports:** prompt ranks are flat (4.4–6.2).

Chain of thought gives no consistent gain on either dataset.

### Finding 5. The compact prompt trades precision for answers

For the three cautious models, the compact prompt (unconstrained answer):
- cuts abstention from 79–93% to 16–42%;
- is their best prompt by F1 on football (0.19–0.29);
- gives their lowest F1 ign (0.40–0.57);
- is the most expensive prompt for every model, because it produces the longest output (gpt-oss:20b: 237 output tokens, about 90% of them reasoning);
- is the least stable prompt: all 3 runs agree on only 44–69% of the tasks.

### Finding 6. Memory is a cheap gain for B on football, and a mixed one on airports

Search + memory replays the synthesis call on the same evidence and also allows the model's own knowledge. Compared with search only:

| | Football | Airports |
|---|---|---|
| F1 | +0.04 to +0.07 for every model; significant for gemma4, mistral and deepseek | +0.05 for gemma4 and +0.08 for deepseek; about 0 for the gpt-oss models; −0.03 for mistral |
| F1 ign | rises for all 5 models | falls for 3 models |
| Abstention | falls by 1–14 points | falls by 9–17 points |

What this suggests:
- On airports the evidence is usually better than the model's memory; on football it is not.
- On football, answers that cite no evidence (answers from memory) are only 2–14% of the answers, but they are more precise (0.37–0.67) than the cited ones (0.26–0.31).
- Task by task, memory mostly converts tasks that search left abstained. They become correct and wrong in similar numbers (8–15 abstained → correct and 13–21 abstained → wrong per model).
- Memory reuses the retrieval, so its marginal cost is one synthesis call.

### Finding 7. The two strategies are complementary

On football, only 8–27 tasks per model are right under both A best and B + memory.

A deployable rule, *use B and fall back on A when B abstains*:
- beats A best for gpt-oss:20b (F1 0.27 vs 0.25) and gpt-oss:120b (0.24 vs 0.19);
- ties for gemma4 (0.29);
- loses for mistral and deepseek.

An oracle that picks whichever answer is right reaches 0.27–0.46. That is 0.03–0.09 above A best: the most a confidence-based selector could gain.

When both strategies answer, they give the same value in only 9–19% of the tasks.
- **mistral:** agreement is a useful confidence signal. Answers both strategies give are 64% right, against 48% (A) and 33% (B) when they differ.
- **Smaller models:** agreement is no signal at all.

This is a good discussion point: hybrid A + B systems and self-consistency across strategies.

### Finding 8. Identifiers: the name helps A much more than B

In the 81 high-G football tasks, the player's name and/or team was added with probability 0.38.

| | No name: F1 | No name: F1 ign | Name given: F1 | Name given: F1 ign |
|---|---|---|---|---|
| A, all prompts pooled | 0.17 | 0.44 | 0.40 | 0.70 |
| A best | 0.20 | 0.40 | 0.45 | 0.66 |
| B + memory | 0.13 | 0.37 | 0.22 | 0.38 |

With B, the name mostly lowers abstention: from 75% to 28% with search, and from 67% to 10% with memory. The team alone helps less, and not for every model.

**Interpretation.**
- With the name, the model recognises a known entity from its training data.
- With the name, the search finds the right page, but extracting the stored value is the hard step.

### Finding 9. Context hurts on airports and is neutral on football

The pairs below are those where the context-on task replaces some of the node's own given properties with properties of neighbouring nodes. The other pairs are identical tasks, so they give identical answers.

**Airports, 85 pairs:**
- F1 falls for every model, from −0.05 to −0.16, all significant.
- F1 ign falls significantly for mistral (−0.06) and deepseek (−0.12).

**Football, 33 pairs:**
- Pooled ΔF1 is +0.002 [−0.022, +0.025] and ΔF1 ign is +0.018 [−0.015, +0.051].
- 3 of 45 model × prompt cells are significant, about what chance gives at 95%.

**Interpretation.** A node's own properties are stronger clues than the properties of its neighbours. On airports, an own property like the city or the ICAO prefix nearly identifies the airport, so swapping it for context costs more.

### Finding 10. The difficulty measure transfers poorly

**Airports:**
- The difficulty score works as intended: the easiest tertile scores highest (A best F1 0.28, B + memory 0.34).
- The popularity of the hidden property orders the scores: low → high H stratum goes from 0.05 to 0.31 for A best and from 0.10 to 0.43 for B + memory.
- Search helps most for popular properties.

**Football:**
- The hardest tertile scores *highest* for every approach (A best 0.40, B + memory 0.24).
- The H strata are flat.
- Spearman ρ between pop(p) and F1 is about 0.

**Why.** Football properties cluster at pop ≈ 0.79 (outfield statistics) and ≈ 0.06 (goalkeeper statistics). Within each cluster, what decides the score is whether the value can be guessed:
- easy: *Minutes per start* (truth 61–90, median 88), *Penalties missed* (usually 0), *Date of birth* (often known);
- hard: specific counts like *On-offsides* or *Matches fully played*.

69 of the 81 hardest football tasks are low-G tasks.

**For the paper.** pop(p) × centrality measures how rare the information is *in the graph*, not how hard it is to guess or retrieve. Discuss this as a limitation of the difficulty model, or propose a property-level "guessability" term.

### Finding 11. Wrong numbers are far off, and many of them are a default 0

The median relative error of the wrong numeric answers is about 1.0 for every model on football.

The share of wrong numeric answers that are exactly 0:

| Strategy A | B search | B + memory |
|---|---|---|
| 38% | 6% | 14% |

**Interpretation.**
- B's wrong numbers are real values from the wrong page, season or competition.
- A's are often a default 0 for counts.
- Near misses do not show up as errors, because the scoring tolerance (similarity ≥ 0.75) already accepts them.

### Finding 12. Cost: B is 10–400× more expensive than A and amplifies the input 6–7×

| | Energy per run | Token amplification |
|---|---|---|
| Strategy B | 0.36–1.75 Wh, of which the web searches (0.3 Wh each, 1.2–3.0 per run) are 54–98% | 5.6–7.2× |
| Model's best A prompt | 0.001–0.13 Wh | 1.1–3.9× |

Within Strategy A:
- energy differs about 200× between models (gemma4 lowest, mistral highest);
- energy differs 1.5–6× between prompts of one model.

On airports, the extra energy of B buys a higher F1 for 4 models. On football it buys a lower F1 for 4.

Token amplification **falls** as prompts get longer, because the template counts as user tokens. This makes it a measure of the system's overhead relative to what the user wrote, not of the prompt's length.

### Finding 13. Stability

Averaged over the models, the share of tasks whose 3 runs agree on the outcome:

| A zero-shot | B search | B + memory | A best |
|---|---|---|---|
| 79–83% | 71% | 67% | 60–69% |

A best is the least stable because it is often the compact prompt. The sd of the F1 of single runs stays below 0.04, so voted results are robust.

---

## 2. Discussion points and caveats to address in the paper

1. **Selecting A best on the test tasks is optimistic.** A best is the best of 9 prompts on the same tasks, so the A best vs B comparison favours A. On football A still wins, which strengthens the conclusion. On airports B still beats A best for the gpt-oss models. Also report A zero-shot and A mean.
2. **Scoring by token overlap penalises format, not meaning.**
   - *Position* on football is stored as codes ("GK", "DF,MF"). "Goalkeeper" or "central midfielder" count as wrong.
   - B gets 1 of 71 Position answers right.
   - This deflates B on football slightly. Mention it, or normalise codes in a sensitivity check (not done here, since the scoring was kept fixed).
3. **The numeric tolerance (similarity ≥ 0.75) makes typical-value properties easy.** It also explains why football's "hard" tasks score highest. Consider reporting results without the most guessable properties as a robustness check. For example, without *Minutes per start* the ordering of the tertiles is unchanged.
4. **Football context-on and context-off sets differ in only 33 tasks.** The other 200 tasks are identical and give identical answers through the response cache. Context comparisons must use the 33 changed pairs. Comparing the two sets as wholes looks like "no difference" by construction.
5. **The search + memory variant is a replay.** It uses the evidence of the search run, so search and search + memory differ only in the synthesis instruction. This is a clean ablation of "evidence only" vs "evidence + parametric knowledge".
6. **Searches are counted again in the memory variant's energy.** That is the cost a stand-alone run would have; the marginal cost of memory is one LLM call.
7. **Evidence truncation.** The 3,000-token evidence cap is reached in 54–100% of B runs on football and 14–98% on airports. The synthesis model often sees truncated pages, which may cut the relevant row of a statistics table.
8. **One synthesis_invalid run** (mistral, memory, football) is the only failed run among the 7,170 B runs on football. It is excluded from the vote, as the scoring does for any failed run.
9. **Sources.** Football evidence comes from 1,174 domains and airports evidence from 1,955. The likely dataset sources (fbref.com, ourairports.com, openflights.org) are not blocked: the blocklist only covers the project's own GitHub repository, so the agent behaves as a real user's would. fbref.com is the most frequent football source, but its answers are only 35% precise.

---

## 3. Suggested storyline

1. **Setup:** two property graphs with different kinds of facts: stable reference data (airports) and season statistics (football).
2. **Strategy A:** the model matters more than the prompt for F1. Prompts matter for accuracy, but prompt rankings do not transfer between datasets. Report F1 and F1 ign, because abstention dominates F1 differences.
3. **Strategy B:** search helps on airports and hurts on football. The bottleneck is identifying the entity and then reading the right value. Memory on the same evidence is a cheap, consistent gain on football.
4. **A vs B:** the two strategies are complementary (an oracle adds up to +0.09 F1). Simple fallbacks help the cautious models.
5. **Graph-side factors:** context and identifiers. The name is the strongest single factor for A. Neighbour context is weaker than the node's own properties.
6. **Difficulty:** popularity-based difficulty holds on airports but not on football, which motivates a guessability-aware difficulty model.
7. **Cost:** B costs 10–400× the energy of A and amplifies the input 6–7×. The trade-off pays off only where retrieval works.

---

## 4. Data files (CSV)

| Content | Airports v5 | Football v1 idmix |
|---|---|---|
| Tasks | `tasks/airports/v5_ctx/tasks.csv`, `tasks/airports/v5_noctx/tasks.csv` | `tasks/football/v1_ctx_idmix/tasks.csv` (+ `tasks_readable.csv`), same for `v1_noctx_idmix` |
| Ground truth | `tasks/airports/v5_{ctx,noctx}/ground_truth.csv` | `tasks/football/v1_{ctx,noctx}_idmix/ground_truth.csv` |
| Centrality, entities in the tasks | `tasks/airports/v5_ctx/task_airports_centrality.csv` | `tasks/football/v1_ctx_idmix/task_players_centrality.csv` (also in `v1_noctx_idmix`) |
| Centrality, all nodes | `tasks/airports/v5_ctx/intermediate/centrality.csv` | `tasks/football/v1_ctx_idmix/players_centrality.csv` (also in `v1_noctx_idmix`) |
| Property popularity pop(p) and H stratum (new) | `tasks/airports/v5_{ctx,noctx}/property_popularity.csv` | `tasks/football/v1_{ctx,noctx}_idmix/property_popularity.csv` |
| Raw runs, unscored | `results/airports/v5_{ctx,noctx}/strategy_a/results.csv` and `runs.jsonl`; `v5_ctx/strategy_b{,_memory}/` | `results/football/v1_{ctx,noctx}_idmix/strategy_a/results.csv` and `runs.jsonl`; `v1_ctx_idmix/strategy_b{,_memory}/` |
| Scored runs (one row per run) | `results/airports/v5_ctx/scored/scored_runs.csv`, `v5_noctx/scored/scored_runs.csv` (new) | `results/football/v1_{ctx,noctx}_idmix/scored/scored_runs.csv` |
| Scored voted answers | `results/airports/v5_{ctx,noctx}/scored/scored_voted.csv` | `results/football/v1_{ctx,noctx}_idmix/scored/scored_voted.csv` |
| A vs B per task | `results/airports/v5_ctx/a_vs_b_per_task.csv` | `results/football/v1_ctx_idmix/a_vs_b_per_task.csv` (new; adds B memory and the identifier columns) |
| B search detail per run (new) | `results/airports/v5_ctx/scored/b_search_runs.csv` | `results/football/v1_ctx_idmix/scored/b_search_runs.csv` |
| B evidence domains (new) | `results/airports/v5_ctx/scored/b_evidence_domains.csv` | `results/football/v1_ctx_idmix/scored/b_evidence_domains.csv` |
| Energy and effort per run | `results/energy_effort/runs.csv` (both datasets; football B added) | same file |

**Notes on the files:**
- In the airports `scored_*.csv`, the memory variant has `strategy = B` and `variant = search_memory`. In the football files it has `strategy = B_memory`. The reports normalise this.
- Football Strategy B is in one folder, `results/football/v1_ctx_idmix/strategy_b/`. The 160 deepseek runs made by the second process of the split run were merged into it; `run_manifest_part2.json` there describes that process. Each Strategy B folder (both datasets) has a flat `runs.csv` with one row per run.
