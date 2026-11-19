# Why the DOLAP 2026 results are higher than ours

*2026-11-10. Analysis notebook: `notebooks/dolap2026_ollama.ipynb` (built by `notebooks/build_dolap2026_notebook.py`),
section 5. Every number below comes from that notebook.*

## Short answer

The F1 values in Table 3 of the DOLAP paper (0.54–0.80) and the F1 values of the current experiments (about 0.1–0.3)
are not the same measure. Four things differ:

| Factor | Effect on the gap | Evidence |
|---|---|---|
| **Scoring**: precision fixed at 1, abstentions ignored, unasked and copied properties counted | **large**: most of the gap | §2–3: the same answers drop from 0.54–0.80 to 0.17–0.43 |
| **Choosing the best of 18 configurations** after seeing the results | medium: +0.05 to +0.13 | §3, step 1 → 2 |
| **Models**: proprietary vs open-weight | real, but only for the GPT-5 models | §4: on the same tasks and scoring, GPT-5 0.43 and GPT-5 Mini 0.38; Claude 0.17–0.22; gemma4:31b 0.19 |
| **Task set**: v5 is harder than the 36 DOLAP tasks | medium: about +0.05 to +0.15 | §5: `commercial_flights` is 1/3 of v5; G names the airport less often |

So the gap is **not only because the DOLAP models are proprietary**. Scored our way, the proprietary models get
F1 0.17–0.43 on the DOLAP tasks, and our lineup gets 0.14–0.34 on v5.

## 1. The two scorings

Each asked (hidden) property of a task is **correct** (c), **wrong** (w, a value that does not match) or **abstained**
(a, no value).

**DOLAP scoring** (the paper's `evaluation.ipynb`):

- **What is compared:** every property the model returned that has a ground-truth value, **whether it was asked or
  not**. This includes values the model copied from its own input.
- **Unanswered properties:** ignored.
- **Tolerance:** token overlap over the *shorter* string ≥ 0.75. A coordinate is compared only when both latitude
  and longitude were answered.
- **Formula:** sklearn's precision/recall/F1 with `y_true` set to all ones and `y_pred` = 1 for a correct answer, 0
  for a wrong one. This gives
  - precision = 1 (whenever at least one answer is correct)
  - recall = accuracy = c / (c + w)
  - F1 = 2·acc / (1 + acc)
- **Table 3:** reports, for each model, the best of the 18 configurations (9 prompts × with/without context).

**Our scoring** (`evaluation.taskset.metrics`, used by every current experiment):

- **What is compared:** only the asked properties.
- **Tolerance:** stricter.
  - Text must overlap in both directions.
  - A list given for a single value is wrong.
  - A lone coordinate is compared within 0.1°.
  - Units are converted.
- **Main metrics:**
  - F1 = 2c / (2c + 2w + a)
  - precision P = c / (c + w)
- **Beside them:** F1 ign = 2c / (2c + w) = 2P / (1 + P). This is the DOLAP formula applied to the asked properties.

## 2. Why the paper says precision is "steadily equal to 1"

Yes, this comes from the sklearn call. It is the same mechanism as our F1 ign.

`y_true` contains only ones (every compared property "should be correct"), so no wrong answer can ever be a false
positive:

| | Predicted 1 (correct) | Predicted 0 (wrong) |
|---|---|---|
| **True 1** (every property) | TP = correct answers | FN = wrong answers |
| **True 0** | never happens | never happens |

As a result:

- FP = 0, so precision = TP / (TP + 0) = 1 as soon as one answer is correct.
- Recall = TP / (TP + FN) = c / (c + w), which is the accuracy over answered properties.
- F1 = 2·1·R / (1 + R) = 2c / (2c + w). This is our **F1 ign**, but computed over every answered property instead of
  only the asked ones.

**Example:** 2 correct and 3 wrong answers.

| | Precision | Recall | F1 |
|---|---|---|---|
| sklearn with `y_true` all ones | 1.0 | 0.4 | 0.571 |
| Ours (a wrong answer counts against the model) | 0.4 | — | F1 ign = 0.571 |

When a wrong answer counts against the model, the published models' precision is far from 1. Over all 18
configurations, on the asked properties:

| Model | Precision | F1 ign | F1 | Error rate | Abstention rate |
|---|---|---|---|---|---|
| GPT-5 | 0.54 | 0.70 | 0.43 | 0.31 | 0.34 |
| GPT-5 Mini | 0.46 | 0.63 | 0.38 | 0.38 | 0.30 |
| Claude Sonnet 4.5 | 0.40 | 0.57 | 0.22 | 0.22 | 0.63 |
| Gemini 2.5 Pro | 0.34 | 0.51 | 0.33 | 0.59 | 0.10 |
| Gemini 2.5 Flash | 0.33 | 0.50 | 0.28 | 0.48 | 0.28 |
| Claude Haiku 4.5 | 0.25 | 0.39 | 0.17 | 0.41 | 0.46 |
| gemma4:31b (ours, 1-run test) | 0.24 | 0.39 | 0.19 | 0.51 | 0.34 |
| Qwen3 4B (ours) | 0.14 | 0.25 | 0.10 | 0.50 | 0.42 |

The paper's statement is therefore an artefact of the formula, not a finding. The new paper should say so explicitly.

## 3. From the Table 3 number to our F1, one step at a time

The steps below use the **same voted answers** (the published ones for the proprietary models). One difference is
removed at a time:

1. **Table 3:** best of 18 configurations, DOLAP F1.
2. **All 18 pooled:** no choice of configuration after seeing the results.
3. **Without copied input:** drop G values the model repeated from its input.
4. **Asked properties only:** drop answers to properties that were not asked.
5. **Strict tolerance:** our tolerance. The value at this step is **F1 ign**.
6. **Abstentions counted:** asked properties without an answer count as misses. The value at this step is **our F1**.

| Model | 1 | 2 | 3 | 4 | 5 (F1 ign) | 6 (F1) | Best configuration (answers behind it) |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | **0.80** | 0.67 | 0.67 | 0.54 | 0.57 | **0.22** | 1-shot CoT + ctx (**3**) |
| GPT-5 | **0.67** | 0.63 | 0.63 | 0.63 | 0.70 | **0.43** | 3-shot CoT (59) |
| Gemini 2.5 Pro | **0.66** | 0.52 | 0.52 | 0.49 | 0.51 | **0.33** | 5-shot CoT + ctx (43; 4 airports) |
| Gemini 2.5 Flash | **0.61** | 0.54 | 0.54 | 0.43 | 0.50 | **0.28** | 5-shot CoT + ctx (52; 4 airports) |
| GPT-5 Mini | **0.60** | 0.51 | 0.51 | 0.50 | 0.63 | **0.38** | 5-shot CoT (49) |
| Claude Haiku 4.5 | **0.55** | 0.44 | 0.43 | 0.36 | 0.39 | **0.17** | 3-shot CoT (29) |
| gemma4:31b (1-run test) | 0.44 | 0.35 | 0.35 | 0.31 | 0.39 | **0.19** | zero-shot CoT + ctx (64) |
| Qwen3 4B | 0.60 | 0.48 | 0.35 | 0.28 | 0.25 | **0.10** | 1-shot + ctx (158) |

What each step does:

- **Best of 18 (step 1 → 2): −0.05 to −0.13.**
  - Sonnet's 0.80 rests on **3 compared answers**.
  - The Gemini rows with context cover only 4 of the 9 airports.
  - Picking the maximum of 18 noisy values over 36 tasks inflates every model, and most of all the models with high
    variance.
- **Copied input (step 2 → 3): about 0 for the proprietary models, −0.13 for Qwen.**
  - Qwen repeats the G values it was given, and the DOLAP scoring counts them as correct.
- **Unasked properties (step 3 → 4): up to −0.13.**
  - Sonnet, Haiku and the two Gemini models add the airport's `name`, `city` and `country` without being asked.
  - These answers are mostly right once the airport is recognised (Sonnet: 71% correct).
  - GPT-5 answers only what is asked, so it is unaffected.
- **Tolerance (step 4 → 5): mostly *up*, by 0.02–0.13.**
  - Our tolerance is stricter on text, but it also compares lone coordinates and numbers with units, which the
    DOLAP rule skipped or compared as text.
  - So the tolerance does not explain the gap.
- **Abstentions (step 5 → 6): the largest step, −0.19 to −0.36.**
  - The DOLAP F1 never sees an unanswered property.
  - Sonnet abstains on 63% of the asked properties, so its F1 drops from 0.57 to 0.22.

**Answer to "is F1 ign still worse than the DOLAP F1?"** Not necessarily.

- On the same answers, F1 ign over all 18 configurations (step 5) is **0.39–0.70** for the proprietary models.
- That is the same range as, or above, their pooled DOLAP F1 (step 2: 0.44–0.67).
- The drop to Table 3's level comes from (a) the best-of-18 choice and (b) the abstentions, which only our F1 counts.

## 4. Same tasks, same scoring: the model effect

Step 6 above compares the models on equal terms: same 36 tasks, all 18 configurations pooled, our scoring.

- **GPT-5 (0.43) and GPT-5 Mini (0.38)** are clearly ahead of every other model.
- **Gemini 2.5 Pro and Flash: 0.28–0.33.**
- **Claude Sonnet and Haiku: 0.17–0.22.** Sonnet is precise (0.40) but abstains on two thirds of the properties.
- **gemma4:31b: 0.19**, in the band of the two Claude models. This is a 1-run test with settings that differ from the
  lineup.
- **Qwen3 4B: 0.10.**

So "proprietary" is not one level. Part of the gap is a real model gap, and it concerns mainly the GPT-5 models.

**Caveat:** the current lineup (gpt-oss:20b, gemma4:31b, gpt-oss:120b, deepseek-v4.1-flash) has **not yet been run on
the 36 DOLAP tasks**. All four Ollama accounts hit their monthly usage limit (HTTP 429 on 2026-11-10). The config is
ready (see §7).

## 5. The task sets differ: DOLAP 36 against airports v5

| | DOLAP 36 | v5 context on | v5 context off |
|---|---|---|---|
| Tasks / airports | 36 / 9 | 243 / 21 | 180 / 20 |
| Airport centrality rank (of 7,698): median, worst | 1,331, 2,618 | 2,565, 3,227 | 2,381, 3,227 |
| Mean \|H\| / \|G\| | 3 / 2.6 | 1 / 3.0 | 1 / 2.8 |
| G names the airport (name, IATA, ICAO or city) | **58%** | **30%** | 56% |
| `commercial_flights` share of the scored H | **2%** | **33%** | **30%** |

The v5 results also used different prompts (Figure 2 prompts with property notes, a repair call, 3 runs) instead of
the published prompts with 5 runs.

**What makes v5 harder:**

1. **`commercial_flights`.** It is a third of the v5 scored properties, and the lineup almost never finds it (recall
   0.03 with context, 0.08 without). Removing it raises the lineup's F1 by 0.05–0.08 on every model.
2. **Whether G identifies the airport.** This is the strongest driver of success in both task sets.
   - Context on, G names the airport in 30% of v5 tasks, against 58% for DOLAP.
   - When G names it, the lineup's F1 on v5 is **0.20–0.34** with context and **0.22–0.41** without.
   - That is the range of the proprietary models on the DOLAP tasks.
   - When G does not name it, the lineup's F1 is 0.10–0.23.
3. **The property mix.** The DOLAP tasks ask more of the properties that follow from recognising the airport:
   - `iata` is 15% of the DOLAP properties and 2% in v5;
   - the proprietary models find `iata`, `altitude`, `latitude`, `city` and `country` 33–48% of the time.
4. **Less obscure airports in DOLAP.** v5 includes airports without routes; DOLAP does not. However, v5 F1 is the same
   across centrality strata (0.17–0.19), so this is not a main factor.

v5 is easier in one respect: the property notes explain `source`, `dst` and `timezone`. Found by the lineup 37–77% of
the time in v5, they are almost never found in DOLAP (0–11%), because those prompts did not explain them.

## 6. Other differences to report

- **The Sonnet row of Table 3** is the 1-shot CoT configuration with context, resting on 3 compared answers. The
  5-shot configuration that its effort column (414 tokens) points to gives 0.72 over 119 answers.
- **The Gemini with-context rows** cover 4 of the 9 airports (the three most central ones and one of medium
  centrality).
- **The GPT-5 Mini with-context batch file** (`gpt_batch_requests.jsonl`, ids `context_gpt_mini…`) contains inputs
  without context. If those were the requests sent, that row is a second run without context.
- **The with-context inputs** of our replication use the published procedure (a slice of 3–8 context entries) with a
  fixed seed. The published slices were not saved.

## 7. What to do

**Paper text:**

- Say that the DOLAP F1 is sklearn's F1 with `y_true` all ones, that precision = 1 follows from it, and that it equals
  2·acc / (1 + acc) over answered properties.
- Report our F1 and precision as the main metrics, with F1 ign beside them for comparability.
- Do not compare Table 3 numbers with ours. When citing DOLAP, use the re-scored values:
  - F1 0.17–0.43
  - precision 0.25–0.54
  - F1 ign 0.39–0.70
  - these are for all configurations pooled on the 36 tasks.

**Run the lineup on the 36 DOLAP tasks** once an Ollama account has credit. This gives the direct model comparison on
the same tasks, with the same prompts and 5 runs.

- Config: `configs/strategy_a_dolap2026.yaml`.
  - The lineup uses the same model settings as v5.
  - It keeps the DOLAP protocol: published prompts, 5 runs, no repair.
  - The key is `OLLAMA_API_KEY_WEBSEARCH`; change `api_key_env` to use another.
- Command, one model at a time:

  ```
  strategy-a run --config configs/strategy_a_dolap2026.yaml --models cloud-gpt-oss-20b
  ```

- Size: 3,240 runs per model, model calls only (no searches). Estimated well under $2 per model.
- Then re-execute the notebook (the build command is in the header of `notebooks/build_dolap2026_notebook.py`).
  Every table and chart picks the models up.

**Possibly, if the comparison should be tighter:** report v5 results also without `commercial_flights`, or by
"G names the airport". This would show how much of the v5 difficulty is that one property.
