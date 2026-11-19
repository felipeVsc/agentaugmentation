<!--
Draft text for §5 (Experiments) and §6 (Discussion) of "Data Augmentation in Property Graphs using agentic AI",
written from the v5 reports (2026-10-31):
  notebooks/v5_strategy_a_report, v5_paper_figures, v5_strategy_b_report_taskwise(_f1both),
  v5_strategy_b_memory_report(_f1both), v5_strategy_a_vs_b_mixed_taskwise; figures in notebooks/figures/v5_paper/ab_*.
TASK-BY-TASK VERSION: no bootstrap. Differences are reported with the counts of tasks only one condition gets right,
and significance is the exact McNemar test on those counts. The bootstrap version is paper_v5_experiments_discussion.md.
Strategy B has 3 runs for every model. Figure numbers are placeholders.
-->

# 5. Experiments

This section describes the experimental setting (§5.1). It then reports the results of Strategy A, the LLM alone
(§5.2), of Strategy B, the search agent answering from the evidence only (§5.3), and of its variant that may also
use the model's own knowledge (§5.4). It compares the three approaches (§5.5) and their cost in effort and energy
(§5.6).

## 5.1 Experimental setting

**Dataset and tasks.** We use the Airport property graph (§5.1.1): 7,698 *Airport* nodes connected to *City* and
*Country* nodes and to each other by ROUTE_TO, PASSENGERS and TRADE relationships. Tasks are generated with
Algorithm 2:

- 3 airports are drawn per centrality stratum (PageRank quartiles). Hidden properties *H* and given properties *G*
  are drawn per popularity stratum. |H| = 1 and |G| ∈ {1, 3, 5}.
- This gives 27 stratum combinations × 3 airports × 3 sizes of G = **243 tasks** on 21 airports.
- G draws from the node and its context (the city, the country and neighbouring airports) and contains at least one
  property of the airport itself when the airport has one in the required stratum. 27 tasks have a G made only of
  context properties.
- For the comparison *with* and *without context*, we build a matched set of **180 context-off tasks**. Each keeps
  the airport, H, |G| and the G stratum of a context-on task, but replaces every context property by a property of
  the airport from the same popularity stratum. 85 pairs differ by at least one replaced property.
- The hard H stratum (popularity in the first quartile) contains a single property, `commercial_flights` (Eurostat
  statistics, European airports only). It accounts for 81 of the 243 tasks.

**Models.** We evaluate five open-weight models served by Ollama Cloud, with reasoning disabled or set to its lowest
level (Table 4).

| Model | Parameters (total / active) | Reasoning |
|---|---|---|
| gpt-oss:20b | 21B / 3.6B (MoE) | `low` (cannot be disabled) |
| gemma4:31b | 31B (dense) | off |
| gpt-oss:120b | 117B / 5.1B (MoE) | `low` (cannot be disabled) |
| mistral-large-3 | 675B / 41B (MoE) | none |
| deepseek-v4.1-flash | 763B / ≈16B (MoE) | off |

*Table 4: Models evaluated.*

Three further models (nemotron-3-super, glm-5.3-flash and minimax-m3.1-flash) were discarded: their reasoning could
not be limited, and it made their cost, energy and answers incomparable with the others. All models run at
temperature 0.7 with seeds 0–2 (three runs per task) and a 4,096-token output limit that includes reasoning.

**Strategy A (LLM only).** Each task is instantiated in the nine prompts of Figure 2:

- compact;
- zero-shot;
- 1-, 3- and 5-shot;
- each of the last four also with a chain-of-thought instruction.

All prompts ask for CSV output. Their formulation effort ranges from 81 (compact) to 427 tokens (5-shot + CoT). The
few-shot examples are five airports that appear in neither task set. In total, 5 models × 9 prompts × 3 runs ×
(243 + 180) tasks = 57,105 runs.

**Strategy B (search agent).** Each run of Algorithm 4 makes two LLM calls around one search step:

1. **Plan call.** The model states which real-world airport the node most likely is (or *unknown*) and writes at
   most three web search queries.
2. **Search.** Each query goes to Ollama web search, which returns five results. Every page is split into
   500-character chunks, and the three chunks closest to the query are kept (BM25 pre-filter, then `bge-small-en-v1.5`
   embeddings). At most 3,000 evidence tokens are passed on.
3. **Synthesis call.** The model answers from the numbered results and cites the results each value comes from. We
   run two versions of this call:
   - **search** (evidence only): the model may use only the results, and returns *null* when they do not contain the
     value. A value without a valid citation is discarded. This isolates what search adds to the model.
   - **search + memory**: the model may also answer from its own knowledge, marking such values with an empty list
     of sources. This is the fallback of ManuSearch [ref] and of the DOLAP agent.

As in ManuSearch, the number of sub-queries is capped at three and the top five pages are retrieved. The search +
memory variant is run by replaying the synthesis call of every *search* run on the same plan and the same evidence,
with only the synthesis prompt changed. The two variants therefore differ only in whether the model may use its own
knowledge.

Only the authors' repository, which publishes the dataset with its answers, is blocked. Other public copies of the
source data (OpenFlights, OurAirports) remain reachable, as they would be for any user. Retrieval is never shared
between runs, so the three runs of a task search independently. Strategy B was run on the context-on tasks only.

**Scoring.** Each hidden property is classified against the database value as follows.

- **Correct:**
  - text: shared words cover at least 75% of the longer string;
  - numbers: within a relative tolerance of 25%;
  - coordinates: within 10 km (Haversine distance).
- **Wrong:** any other value.
- **Abstained:** no value.

As in the DOLAP study, answers are first aggregated over the runs: the most frequent value for text and the mean for
numbers, with abstentions excluded from the vote. We report two F1 scores:

- **F1**, from precision P = c/(c+w) and recall R = c/(c+w+a). An abstention lowers recall.
- **F1 with abstentions ignored**, 2·acc/(1+acc) with acc = c/(c+w). This is the measure of Table 3 of the DOLAP
  study. A model is judged only on the properties it answers.

We give both scores throughout; the second is noted *ign.* where space is short.

**Comparisons are made task by task, without resampling.** Every task hides one property, so two conditions on the
same tasks give a pair of outcomes per task. We report the difference in each score and the number of tasks that only
one condition answers correctly; a difference is called significant when the exact McNemar test on those counts
(a two-sided binomial test against 50/50) gives p < 0.05. The test concerns which condition gets more properties
right; a condition can raise F1 without it, by turning wrong answers into abstentions, and we report those counts as
well.

**Effort and environmental impact.** Human effort is measured in three ways:

- prompt formulation effort (PFE, the template tokens without placeholders);
- the keystroke-level model (KLM) [6];
- token amplification (generated tokens over tokens written by the user).

Energy per run (the EIS) is the EcoLogits estimate of every LLM call in the run. For deepseek-v4.1-flash, which
EcoLogits does not know, we use a model of similar total size. The energy of web searches is not reported by the
provider and is not included.

## 5.2 Strategy A: the LLM alone

**Main results.** Table 5 gives the best prompt of each model on the context-on tasks.

| Model | Best prompt | F1 | Precision | Abstention | F1 (abst. ignored) |
|---|---|---|---|---|---|
| gpt-oss:20b | 3-shot + CoT | 0.161 | 0.23 | 47% | 0.38 |
| gemma4:31b | zero-shot + CoT | 0.190 | 0.31 | 57% | 0.48 |
| gpt-oss:120b | compact | **0.237** | 0.25 | 12% | 0.40 |
| mistral-large-3 | 5-shot + CoT | 0.219 | 0.24 | 16% | 0.39 |
| deepseek-v4.1-flash | 5-shot + CoT | 0.231 | 0.27 | 26% | 0.43 |

*Table 5: Strategy A, best prompt of each model, context on (243 tasks, answers voted over 3 runs).*

- **All models struggle.** No model exceeds F1 0.24 (0.48 with abstentions ignored), and the three best models are level
  task by task (each pair: 25 against 20 or 16 against 16 properties only one of them gets right).
- **Scored as in the DOLAP study, the best configuration of each model reaches 0.42–0.64** (context on or off,
  whichever is better). This is close to the 0.54–0.80 of the proprietary models of the DOLAP study, with GPT-5
  (0.67) and Gemini 2.5 (0.61–0.66) in the same range as deepseek-v4.1-flash (0.64) and gpt-oss:120b (0.59).
- **A constant guess (the most frequent value of each property over all airports, or its median) reaches F1 0.358
  (0.53 with abstentions ignored, as it never abstains), above every model under both scores.** Its score comes from properties with one dominant value (`runways` = 1, `type` = airport,
  `source` = OurAirports) and from the 25% tolerance, which a median alone satisfies for 30% of the
  `commercial_flights` values.

**Model or prompt?**

- **The model matters more than the prompt.** For a fixed prompt, models span 0.08 F1 on average (0.19 with
  abstentions ignored); for a fixed model, prompts span 0.06 (0.13).
- **Averaged over the nine prompts, F1 grows with model size:** 0.138, 0.160, 0.192, 0.196 and 0.207 from
  gpt-oss:20b to deepseek-v4.1-flash. With abstentions ignored the order changes (0.31, 0.42, 0.50, 0.36 and 0.41):
  gpt-oss:120b answers most precisely, while mistral-large-3 and deepseek-v4.1-flash owe their F1 to answering more
  often, not to being right more often.
- **No prompt feature has a measurable effect.** Pooled over the five models:
  - adding a chain-of-thought instruction changes F1 by +0.003 to +0.007 (+0.013 to +0.019 with abstentions ignored);
  - adding one, three or five examples changes it by +0.001 to +0.007 (−0.018 to −0.001 with abstentions ignored);
  - none of these changes the number of properties answered correctly significantly (e.g. zero-shot + CoT
    against zero-shot: 42 against 35 tasks; 5-shot against zero-shot: 67 against 52).
- **Chain of thought cannot work here.** With reasoning disabled and CSV-only output, the CoT prompts receive the
  same short answers as the others (8–42 output tokens).
- **The one wording that matters is the abstention instruction.** The Figure 2 prompts end with *"if no facts can be
  verified, output only the header row"*; the compact prompt does not. With the compact prompt, every model abstains
  less (gpt-oss:120b: 12% against 60–66%), which is why it is gpt-oss:120b's best prompt. With abstentions ignored, compact is the weakest prompt (0.32
  against 0.39–0.43, mean over the models): it answers more, but less precisely.

**Context.** Contrary to the DOLAP study, the models perform better *without* context.

- **On the 85 matched pairs,** the models get more properties right without context in all 45 model × prompt
  combinations, significantly in 18. F1 with abstentions ignored is lower with context in 36 of 45.
- **Averaged over the nine prompts,** context lowers F1 by 0.04 to 0.13 per model (0.00 to 0.14 with abstentions
  ignored). The largest drop is for deepseek-v4.1-flash (0.207 against 0.336; 0.41 against 0.55 ignored).
- **The comparison is about which properties the model receives, not how many.** The context properties drawn by
  the sampler (a country's consumer price index, the airline of a neighbouring route) do not identify the airport.
  The own properties that replace them (`icao`, `name`, coordinates) do.
- **Context also raises abstention.** For example, deepseek-v4.1-flash zero-shot abstains on 34% of the properties
  with context and 5% without.
- **The DOLAP finding that context helps rests on the Hard-Hard quadrant,** where all six proprietary models were
  above the diagonal (mean +0.11). That quadrant cannot be tested here: our only hard property, `commercial_flights`,
  is also the only hard property of the airport itself, so no context-off task can have it in both H and G. Outside
  that quadrant, neither study shows a sizeable context effect (mean Δ between 0.00 and +0.04).

**Difficulty.**

- **The popularity of the hidden property dominates.**
  - Easy H properties reach F1 0.43. Pooled over models and prompts, easy H scores 0.28 (0.52 with abstentions
    ignored) and hard H 0.04 (0.12).
  - The hard H stratum (`commercial_flights`) stays at 0.02–0.07 with context and 0.09–0.13 without.
  - Tasks whose G holds only context properties are almost never solved (0.00–0.05; 0.00–0.22 with abstentions ignored).
- **Node centrality does not order the difficulty.** Most airports share the minimum PageRank, so the *low* stratum
  is large and mixed.
- **The composite difficulty score is only weakly related to accuracy.** Its easiest quartile has the highest F1 for
  every model, but the rank correlation between difficulty and per-task recall is only −0.05 to −0.13.

**Effort and energy (Strategy A).**

- **Effort does not buy accuracy.** PFE ranges from 81 to 427 tokens and the KLM typing time from 3 to 9 minutes.
  Yet the mean F1 over the five models stays at 0.17–0.18 for every prompt (Figure 4), and at 0.39–0.43 with
  abstentions ignored for every prompt but compact (0.32).
- **Energy per run spans two orders of magnitude:**

  | Model | Wh per run |
  |---|---|
  | gemma4:31b | 0.001 |
  | gpt-oss:20b | 0.009 |
  | gpt-oss:120b | 0.024 |
  | deepseek-v4.1-flash | 0.043 |
  | mistral-large-3 | 0.090 |

  Energy is driven by active parameters and by the hidden reasoning tokens of gpt-oss.
- **The most efficient configuration is gpt-oss:120b with the compact prompt:** the best F1 (0.237; 0.40 with abstentions
  ignored) at 0.033 Wh per run.
  All 57,105 runs together used an estimated 1.89 kWh and cost USD 7.41.

## 5.3 Strategy B: search

**Main results.** Table 6 gives the results of Strategy B (evidence only) on the 243 context-on tasks.

| Model | Runs | F1 | Precision | Abstention | F1 (abst. ignored) | Correct / wrong / abstained |
|---|---|---|---|---|---|---|
| gpt-oss:20b | 3 | 0.249 | 0.34 | 41% | 0.50 | 48 / 95 / 100 |
| gemma4:31b | 3 | 0.152 | 0.29 | 65% | 0.45 | 25 / 60 / 158 |
| gpt-oss:120b | 3 | **0.316** | **0.45** | 47% | **0.62** | 59 / 71 / 113 |
| mistral-large-3 | 3 | 0.294 | 0.37 | 35% | 0.54 | 59 / 100 / 84 |
| deepseek-v4.1-flash | 3 | 0.163 | 0.30 | 63% | 0.47 | 27 / 62 / 154 |

*Table 6: Strategy B (search) on the context-on tasks, voted over 3 runs.*

- **gpt-oss:120b with search reaches F1 0.316,** followed by mistral-large-3 (0.294) and gpt-oss:20b (0.249); with
  abstentions ignored, 0.62, 0.54 and 0.50.
- **Task by task, the three leaders are level** (gpt-oss:120b and mistral-large-3 each get 22 properties right that
  the other misses), and each gets significantly more right than gemma4:31b and deepseek-v4.1-flash (e.g.
  gpt-oss:120b: 40 against 6 and 40 against 8). With abstentions ignored, gpt-oss:120b leads by 0.08–0.17.
- **With abstentions ignored, gpt-oss:120b (0.62) and mistral-large-3 (0.54) beat the constant baseline (0.53)**,
  which no model's best Strategy A prompt does on the context-on tasks (0.38–0.48). The baseline never abstains, so it scores 0.53 under this
  measure.

**Identification is the bottleneck.** The plan call names an airport in only 31–36% of the runs. mistral-large-3 is
the exception at 59%.

- **The plan depends on G.** With no own property of the airport in G, the plan names an airport in 0–20% of the
  runs (mistral-large-3: 42%). With two or more, it names one in most runs (56–100%, with two small cells lower).
  When it writes *unknown*, the queries search for the given values themselves (*"airport CPI 110.62 commercial
  flights"*) and return pages about no airport in particular.
- **When the evidence is about the right airport, the pipeline works.** We approximate this by the airport's
  identifier appearing in the evidence.
  - In those runs (17–24%), F1 per run is 0.44–0.53 (0.68–0.83 with abstentions ignored) and precision 0.52–0.71.
  - In the other runs, F1 per run falls to 0.05–0.19 (0.24–0.50).
  - For deepseek-v4.1-flash, a plan that writes *unknown* leads to F1 0.03 (0.30 with abstentions ignored) and
    90% abstention.

**Voting.** The three runs of Strategy B search independently and find different pages.

- The runs of the two gpt-oss models agree on only 56–62% of the properties, mistral-large-3's on 69%, against 81% for
  deepseek-v4.1-flash and 85% for gemma4:31b, which usually writes a single query.
- Voting therefore raises F1 by +0.06 for the two gpt-oss models, +0.03 for mistral-large-3 and +0.02 for
  deepseek-v4.1-flash. For Strategy A, the
  same models gain +0.003 to +0.040.
- Voting does not raise F1 with abstentions ignored (−0.06 to +0.00). It fills in properties that one run left blank,
  which raises recall but not precision.

**Properties.**

- **Search wins the facts printed on an airport's page:**

  | Property | F1 with B (by model) | F1, abstentions ignored (by model) |
  |---|---|---|
  | `runwaysWidth` | 0.33–0.73 | 0.62–0.87 |
  | `altitude` | 0.36–0.60 | 0.57–0.86 |
  | `timezone` | up to 0.91 | 0.75–1.00 |
  | codes, coordinates, `country` | up to 0.50 | up to 1.00 |

  Strategy A almost never got these right.
- **Search loses values a model can guess without evidence:** `runways`, `type` and `distCity`, and `source` for
  three models. The evidence-only rule makes the agent abstain where Strategy A answers from a prior. Even
  with abstentions ignored, Strategy A's best prompt is ahead on `runways` (0.80 against 0.40, pooled) and
  `distCity` (0.48 against 0.11).
- **`commercial_flights` remains unsolved (F1 ≤ 0.10),** although the few answers given are right more often than
  Strategy A's (0.31 with abstentions ignored, pooled, against 0.12 for the best A prompt). The agent finds the Eurostat tables, the largest source of
  passages (16–26%), but without the airport's name it cannot select the right row.
- **The wrong answers are mostly evidence about another airport:** *Melbourne* for West Wyalong, *RAAF Base Amberley*
  for Northern Peninsula Airport.

**Sources.**

- The answers cite Wikipedia (12–15%), OurAirports (6–14%) and aviation directories most often.
- Public copies of the source data appear in the evidence of 11–19% of the runs. They raise the precision of
  gpt-oss:120b and mistral-large-3 (0.88 and 0.77 in those runs) but not that of the other models.
- Only 1–3% of the answers cite such a copy.

## 5.4 Strategy B: search + memory

**Main results.** Table 7 gives the search + memory variant, on the same plans and the same evidence as Table 6.

| Model | Runs | F1 | Δ F1 vs search | F1 (abst. ignored) | Δ vs search (abst. ignored) | Tasks only memory / only search correct | Abstention | Values from memory | Precision: evidence / memory |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss:20b | 3 | 0.256 | +0.007 | 0.46 | −0.039 | 13 / 7 | 26% | 19% | 0.36 / 0.30 |
| gemma4:31b | 3 | 0.205 | +0.053 | 0.46 | +0.006 | **13 / 0*** | 48% | 22% | 0.36 / 0.17 |
| gpt-oss:120b | 3 | **0.324** | +0.008 | **0.59** | −0.032 | 12 / 7 | 37% | 19% | 0.50 / 0.42 |
| mistral-large-3 | 3 | 0.265 | −0.029 | 0.47 | −0.067 | 4 / 7 | 26% | 8% | 0.35 / 0.09 |
| deepseek-v4.1-flash | 3 | 0.238 | +0.076 | 0.53 | +0.069 | **17 / 1*** | 51% | 12% | 0.41 / 0.06 |

*Table 7: Strategy B (search + memory). Differences on the same tasks; \* = McNemar p < 0.05; "values from memory" are the
values given without a citation, per run.*

- **Memory helps the models that abstain most.** gemma4:31b and deepseek-v4.1-flash, which abstained on 63–65% of
  the properties with search alone, gain significantly (+0.05 and +0.08 F1; 13 and 17 properties gained against 0 and 1 lost). Their abstention falls by 12–17 points.
- **It does not help the others.** The two gpt-oss models are level (+0.01), and mistral-large-3 loses a little
  (−0.03 F1; 4 properties gained, 7 lost, not significant): its answers from memory are right 9% of the time.
- **gpt-oss:120b remains the best configuration (0.324; 0.59 with abstentions ignored)** and gets significantly more
  properties right than gemma4:31b and deepseek-v4.1-flash.
- **With abstentions ignored, memory lowers the score of the models whose memory is unreliable** (mistral-large-3
  0.54 → 0.47; the gpt-oss models −0.03 to −0.04) and raises deepseek-v4.1-flash's (0.47 → 0.53).
  gpt-oss:120b (0.59) and, just, deepseek-v4.1-flash (0.534) are above the constant baseline (0.527).
- **Answers from memory are less often right than answers from evidence, for every model** (0.06–0.42 against
  0.35–0.50). Only gpt-oss:120b's memory is nearly as reliable as its evidence.
- **Memory is reliable only for guessable values.**
  - `type`, `source` and `dst` are always right when answered from memory. These are the values that search alone
    lost; pooled over the models, `type` rises from F1 0.57 to 0.92 and `dst` from 0.60 to 0.82 (0.76 → 0.96 and
    0.95 → 1.00 with abstentions ignored).
  - Half of the answers from memory are `commercial_flights`, and 6% of them are right.
  - Airport-specific facts (coordinates, codes, country, time-zone names) are almost never right from memory (0–8%).
- **On the same evidence, the abstentions that memory turns into answers become mostly wrong ones.** Per model, 20–35
  become wrong and 2–11 become correct. Memory therefore trades precision for coverage: pooled over the models,
  abstention falls from 50% to 38% and precision from 0.36 to 0.34.
- **Memory softens a failed identification.** When the evidence does not contain the airport's identifier, F1 per run
  is 0.10–0.18, against 0.05–0.19 with search alone (0.35–0.51 against 0.24–0.50 with abstentions ignored).
  deepseek-v4.1-flash's runs with an *unknown* plan rise from 0.03 to 0.10 (0.30 to 0.47).

## 5.5 Memory, search and search + memory

Table 8 compares the three approaches on the same tasks and runs, pooled over the five models; Table 9 gives the
paired differences per model.

| Approach | F1 | Precision | Abstention | F1 (abst. ignored) |
|---|---|---|---|---|
| A: zero-shot | 0.174 | 0.25 | 47% | 0.40 |
| A: best prompt | 0.210 | 0.26 | 31% | 0.41 |
| B: search | 0.239 | **0.36** | 50% | **0.53** |
| B: search + memory | **0.259** | 0.34 | 38% | 0.50 |

*Table 8: The three approaches, pooled over the five models, each voted over 3 runs.*

| Model | Runs | Search − A best: F1 / ign. | Tasks only search / only A correct | Search + memory − A best: F1 / ign. | Tasks only s+m / only A correct | Abstention A best → search → search + memory |
|---|---|---|---|---|---|---|
| gpt-oss:20b | 3 | +0.088 / +0.128 | **36 / 18*** | +0.095 / +0.089 | **33 / 9*** | 47% → 41% → 26% |
| gemma4:31b | 3 | −0.037 / −0.024 | 19 / 27 | +0.016 / −0.018 | 26 / 21 | 57% → 65% → 48% |
| gpt-oss:120b | 3 | +0.080 / +0.220 | 29 / 24 | +0.087 / +0.188 | 32 / 22 | 12% → 47% → 37% |
| mistral-large-3 | 3 | +0.074 / +0.154 | 32 / 22 | +0.046 / +0.087 | 27 / 20 | 16% → 35% → 26% |
| deepseek-v4.1-flash | 3 | −0.068 / +0.039 | **13 / 35*** | +0.007 / +0.108 | 21 / 27 | 26% → 63% → 51% |

*Table 9: Strategy B minus Strategy A (best prompt) on the same tasks: F1 / F1 with abstentions ignored, and the tasks only one side gets right. \* = McNemar p < 0.05.*

**Search raises F1 for three of the five models, mostly by avoiding wrong answers.**

- **It raises F1 for both gpt-oss models and mistral-large-3** by +0.07 to +0.09 (+0.13 to +0.22 with abstentions
  ignored).
- **Only gpt-oss:20b gets significantly more properties right than its best A prompt** (36 against 18 tasks).
  gpt-oss:120b (29 against 24) and mistral-large-3 (32 against 22) get more right, not significantly; their gain comes
  mostly from abstaining where the A prompt is wrong (87 and 49 tasks, against 10 and 7 the other way).
- **It leaves gemma4:31b level and hurts deepseek-v4.1-flash,** which gets significantly fewer properties right
  (13 against 35).
- **Against the zero-shot prompt, B's gains are larger:** +0.11 to +0.14 for gpt-oss:20b, gpt-oss:120b and
  mistral-large-3 (+0.10 to +0.25 with abstentions ignored); all three get significantly more properties right than
  zero-shot (38 against 10, 37 against 11, 34 against 14). The zero-shot prompt is the A prompt closest to the agent's prompts: no examples, no chain of
  thought.

**With memory, the agent never gets significantly fewer properties right than the LLM alone.** It closes the gap for gemma4:31b and
deepseek-v4.1-flash, keeps the gains of the gpt-oss models, and gives up part of mistral-large-3's.

**Abstention changes the verdict.**

- **Search makes the answers more precise:** 0.36 against 0.26 for the best A prompt, pooled, and higher for four of
  five models (all but gemma4:31b). With abstentions ignored, search beats the best A prompt for four of five models, most for
  gpt-oss:120b (+0.22) and mistral-large-3 (+0.15), whose answers are right far more often (0.45 against 0.25 and
  0.37 against 0.24).
- **deepseek-v4.1-flash's lower F1 (−0.07) is due to abstention** (88 properties where it abstains and A is wrong),
  not to wrong answers. With abstentions ignored
  it is level with A (+0.04).
- **B's additional abstentions mostly replace wrong answers.** Against A zero-shot, they avoid 2 to 22 wrong answers
  for every correct answer they lose.
- **Whether the agent abstains more than the LLM alone depends on the prompt it is compared with.**
  - The nine A prompts span 2–76% abstention, depending mostly on the abstention instruction.
  - Search falls inside that range for both gpt-oss models, just above it for mistral-large-3, and above it for
    gemma4:31b and deepseek-v4.1-flash.
  - Memory brings every model's abstention back down by 9–17 points.

**Where each approach wins** (pooled over the models):

- **Strategy A** keeps the values a model can estimate without knowing the airport: `runways` (0.62 against
  0.21–0.22; 0.80 against 0.40–0.41 with abstentions ignored) and `distCity` (0.26 against 0.04–0.08; 0.48
  against 0.11–0.18).
- **Search** wins the facts specific to the airport: codes, coordinates, `country`, runway dimensions.
- **Search + memory** wins where both meet: `type`, `dst`, `source`, `timezone`, `altitude`, `name`.
- **No approach solves `commercial_flights`** (0.04–0.10). With abstentions ignored, search is ahead (0.31 and 0.27
  for the two B variants, against 0.11–0.12 for A).
- **By stratum,** search raises F1 most where the hidden property is easy and G is of medium popularity (0.49 against
  0.31 for A zero-shot; 0.71 against 0.55 with abstentions ignored), and memory raises the easy-H row further
  (0.55; 0.74–0.77). The hard-H row stays at or below 0.18 under every approach (0.45 with abstentions ignored).

## 5.6 Effort and environmental impact

**Effort.**

- **Strategy B has a higher formulation effort.** Its two templates total 525 PFE tokens (537 with memory), more
  than any Strategy A prompt (81–427).
- **Its token amplification is 0.17–0.71:** the agent generates fewer tokens than the user writes, because its
  answers are short JSON.

**Energy.** A run of Strategy B sends 3,200–3,900 input tokens in two calls, against about 400 for Strategy A. The
plan call produces about two thirds of B's output tokens.

| | B: search | B: search + memory | A: best prompt |
|---|---|---|---|
| Energy per run | 0.0075–0.77 Wh | 0.0075–0.85 Wh | 0.0007–0.13 Wh |
| Ratio to the same model's mean over the 9 A prompts | 2.5–10× | 2.6–10× | — |
| Energy per correct answer | 0.22–11.0 Wh | 0.15–11.1 Wh | 0.015–2.0 Wh |

- **Memory is almost free.** It adds no search, and 0–11% energy per run. Per correct answer it lowers the energy for
  the models that gain (gemma4:31b 0.22 → 0.15 Wh, deepseek-v4.1-flash 11.0 → 7.0 Wh).
- **Per correct answer, B uses 1.3–16× the energy of the best A prompt,** before counting the searches.

**Cost.**

- **Searches dominate the cost.** At the pay-as-you-go price we observed (≈ USD 0.015 per search), the 861–2,187
  searches per model would cost USD 13–33, against USD 0.26–1.71 for that model's LLM calls.
- **Search limits set the running time.** The undocumented hourly and session limits of the search service
  (≈ 130–170 searches per hour per account) determined B's running time, not model speed. The search + memory
  variant, which makes no search, ran in 23 minutes for all models.

---

# 6. Discussion

**Identifying the entity is the task.**

- Both strategies fail for the same reason: the model does not know which airport the node is.
  - In Strategy A, `name`, `city` and coordinates are almost never recovered, and 65% of the hidden properties are
    answered correctly by none of the five models.
  - In Strategy B, the plan names an airport in a third of the runs (mistral-large-3: 59%). When the evidence
    concerns the right airport, F1 per run is 0.44–0.53 (0.68–0.83 with abstentions ignored); otherwise it is
    0.05–0.19 (0.24–0.50).
- Once the entity is known, retrieving a property printed on its page is easy for every model.
- **This reframes the role of the graph context.** The context properties drawn by Algorithm 2 (statistics of the
  country, attributes of neighbouring airports) rarely identify a node. Replacing them by the node's own properties
  helps every model under F1, and every model but gemma4:31b (level) with abstentions ignored.
- **Future versions of the protocol should therefore distinguish two kinds of G:**
  - properties that identify the entity;
  - properties that only describe its surroundings.
  They should also test context *added to* the node's own properties, not *instead of* them.

**Search is a precision tool; memory restores coverage.**

- **Search makes the answers more precise.** Pooled over the five models, precision rises from 0.26 (best A prompt)
  to 0.36, and F1 with abstentions ignored from 0.41 to 0.53. The extra abstentions mostly avoid errors.
  Task by task, this is where most of the gain lies: only gpt-oss:20b gets significantly more properties right than
  its best A prompt, while gpt-oss:120b and mistral-large-3 mainly stop giving the wrong answers the A prompt gives.
- **Memory on the same evidence recovers coverage.** Abstention falls from 50% to 38% and F1 rises from 0.239 to
  0.259, but precision falls to 0.34 and F1 with abstentions ignored to 0.50. The agent with memory is never
  significantly fewer properties right than the LLM alone.
- **The evidence does the work.** On the same runs, answers from memory are right less often than answers from
  evidence for every model (0.06–0.42 against 0.35–0.50), and memory is reliable only for values with one dominant
  answer (`type`, `source`, `dst`). The gain of the agent over the LLM alone comes from the retrieved pages, not from
  the fallback.
- **Which variant to use depends on the cost of a wrong value.**
  - For augmenting a database, where a wrong value is worse than a missing one, evidence only is the safer policy:
    pooled over the models, it has the highest precision and the highest F1 with abstentions ignored.
  - Where coverage matters, search + memory has the highest F1, and helps most the models that abstain too much
    (gemma4:31b, deepseek-v4.1-flash).
- **The two scores can reverse a conclusion.**
  - Memory raises F1 and lowers F1 with abstentions ignored.
  - deepseek-v4.1-flash with search is below Strategy A in F1 (−0.07) and above it with abstentions ignored (+0.04);
    with search + memory it is level in F1 (+0.01; 21 against 27 properties right) and above it with abstentions ignored
    (+0.11).
  - In Strategy A, F1 grows with model size, but with abstentions ignored gpt-oss:120b is first and the two largest
    models are behind it.
  - Results on LLM-based data augmentation should therefore report precision and abstention, or both scores, never
    one F1 alone.
- **The DOLAP study's Table 3 measure, which ignores abstentions, can be misleading on its own.** In the DOLAP data,
  the best score (Claude Sonnet 4.5, 0.80) rests on three answered properties out of 108, with 97% abstention.
  Conversely, F1 alone hides that search answers `commercial_flights` correctly more often than the LLM alone
  (0.31 against 0.12 with abstentions ignored), because it answers it rarely.

**Prompt engineering matters less than the model and the abstention policy.**

- None of the classic techniques changed F1 measurably once reasoning was disabled and the output constrained.
  With abstentions ignored, chain of thought gains a little (+0.01 to +0.02) and examples lose a little (up to
  −0.02).
- **The wording of the abstention instruction changed abstention by up to 50 points,** and with it the ranking of
  the prompts: compact, which omits it, is gpt-oss:120b's best prompt by F1 and the weakest prompt with abstentions
  ignored (0.32 against 0.39–0.43). In Strategy B, allowing memory plays the same role: it changes abstention by 9–17
  points on the same evidence.
- **For the effort dimension of the protocol, the lowest-effort prompt is as good as any by F1.** The cheapest way to
  more accuracy under both scores is a larger model, or a search step for the models that can use it.

**The models that gain from search are those that plan several queries.**

- The two gpt-oss models and mistral-large-3 write about three queries per run and vary them across runs. Their
  three runs find different pages, and voting combines them (+0.03 to +0.06 F1; voting does not raise F1 with
  abstentions ignored).
- gemma4:31b usually writes one query and gains nothing from search or voting; it gains only once memory is allowed.
- deepseek-v4.1-flash plans like the gpt-oss models but rarely identifies the airport, and then abstains.
- **The single-turn agent leaves identification entirely to one plan call.** An explicit identification step, as in
  multi-agent search systems such as ManuSearch [ref], would target the main source of loss. It would resolve the
  entity first (from codes, coordinates or names in G, possibly over several searches) and search for the property
  second.

**Constant baselines and tolerances.**

- A constant guess beats every model under F1 (0.358). With abstentions ignored (0.53), only gpt-oss:120b (with and
  without memory), mistral-large-3 with search, and, just, deepseek-v4.1-flash with search + memory are above it.
- This reflects both the skewed distributions of some properties and the 25% numeric tolerance, which is lenient
  for properties concentrated around their median.
- **The scoring rule should be tightened before the final tables:** per-property tolerances, and exact match for
  time-zone offsets. Results should also always be reported against such a baseline.

**Energy.**

- Model choice decides energy more than strategy or prompt: two orders of magnitude separate gemma4:31b and
  mistral-large-3 in both strategies.
- Search multiplies the LLM energy per run by 2.5–10 through longer inputs; allowing memory adds almost nothing.
- **The energy of the search itself is unknown.** Agent-level estimates [4] weight tool calls by an assumed factor;
  measuring it would require a self-hosted search engine.

## 6.1 Limitations

- **Scope.**
  - The results cover one graph (airports) and one task size (|H| = 1).
  - Strategy B covers a single agent design (single-turn) and was run with context only.
- **Search + memory is a replay.** It reuses the plan and the evidence of the evidence-only runs. This makes the two
  variants exactly comparable, but a plan written for a memory-enabled agent could differ.
- **"From memory" is the model's own attribution.** A value is counted as from memory when the model cites no
  result; a recalled value that cites an agreeing passage is counted as from evidence.
- **The search engine is a moving target.** Web search results change over time; we store them, so the runs can be
  re-scored but not re-searched. The provider's rate limits are undocumented.
- **The energy figures are estimates.** They come from EcoLogits, not from measurements. deepseek-v4.1-flash uses a
  stand-in model, and searches are not counted.
- **Some public copies of the dataset remain reachable.** They give the agent a small channel to the ground truth;
  blocking them would make the task harder but the setting less realistic.
- **The comparison with the DOLAP study is indirect.** The models differ (proprietary vs open-weight) and so do the
  task sets (36 tasks with |H| = 3 vs 243 with |H| = 1). The hard stratum also differs: `runways`, `iata` and
  `runwaysLength` in the DOLAP study, `commercial_flights` here.
