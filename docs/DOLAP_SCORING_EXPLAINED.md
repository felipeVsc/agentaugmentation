# How the DOLAP 2026 paper scores the models, explained simply

*How the original paper (Table 3) turns a model's answers into a score. It follows the paper's evaluation notebook,
`evaluation.ipynb`, re-implemented in `src/evaluation/dolap.py` as `score_published` and `published_metrics`.
The re-implementation reproduces 5 of the 6 rows of Table 3 exactly.*

---

## The game

The database has a card for each airport. Each card lists facts: name, city, country, altitude, IATA code, and so on.

For every task we play this game with the model:

1. We **show** the model a few facts from one card. For example: *"altitude is 154, city is Koyuk."*
   These shown facts are called **G** (the "given" facts).
2. We **cover up** some other facts on the same card, and ask the model to guess them. For example: *"What are
   `location`, `source` and `dst`?"* These hidden facts are called **H** (the "hidden" facts).
3. The model writes back a list of facts, `property,value`, one per line.
4. Then we **grade** the answer against the real card.

This file is about step 4, the grading.

---

## Step 1: ask five times, keep the popular answer

The model is asked the **same question 5 times**, because it can answer differently each time.

Then, for each fact, we keep one answer:

- **for words**, the answer that came up **most often**. If 3 of the 5 runs say "Koyuk Airport", we keep
  "Koyuk Airport";
- **for numbers**, the **average**. If the runs say 150, 160, 150, 155, 160, we keep 155.

From here on we only look at this single kept answer, called the **voted answer**.

---

## Step 2: which facts get graded?

This is the most important step, and where the DOLAP grading is unusual.

> **The rule: every fact the model wrote down gets graded, if the real card has that fact.**

It does **not** matter whether we asked for that fact:

- **Facts we asked for (H):** graded.
- **Facts we did not ask for**, but the model volunteered: also graded.
- **Facts we showed the model (G)**, which it copied back: also graded, and of course they are right.
- **Facts the model did not answer: skipped.** They are not counted at all, as if the question had never been asked.
- **Facts the card does not have** (the model says "state = Alaska" but the card has no "state"): skipped.

### A real example

Claude Sonnet 4.5, airport PAKK (Koyuk, Alaska).

- **We showed it:** altitude 154, city Koyuk.
- **We asked for:** `location`, `source`, `dst`.

It answered:

| What it wrote | Real card | Did we ask? | Graded? | Result |
|---|---|---|---|---|
| country = United States | United States | no | **yes** | right ✅ |
| iata = KKA | KKA | no | **yes** | right ✅ |
| name = Koyuk Alfred Adams Airport | same | no | **yes** | right ✅ |
| state = Alaska | (no such fact) | no | no | — |
| location | (no answer) | **yes** | no, skipped | — |
| source | (no answer) | **yes** | no, skipped | — |
| dst | (no answer) | **yes** | no, skipped | — |

It answered **none of the three questions**. The grading still sees **3 graded facts, 3 right: a perfect score**
for this task.

Think of a teacher who asks *"What is 7 × 8?"*. The child answers *"My name is Tom, I am 5, and the sky is blue"*.
The teacher marks three correct answers and forgets the question.

---

## Step 3: what counts as "right"?

The answer does not have to be exactly the same as the card; it only has to be **close enough**. "Close enough" is
measured differently for words, numbers and places.

### Words: count the shared words

1. Make both texts lowercase, remove accents and punctuation, and split them into words.
2. Count the words that appear in both.
3. Divide by the number of words in the **shorter** text.
4. If the result is **0.75 or more**, the answer is right.

**Example 1.**

- **Card:** "Koyuk Alfred Adams Airport" (4 words).
- **Answer:** "Koyuk Airport" (2 words).
- **Shared words:** koyuk and airport, so 2.
- **Score:** 2 ÷ 2, because the shorter text has 2 words. That is **1.0, so right ✅**.

Because it divides by the **shorter** text, a short answer only has to fit *inside* the long one.

**Example 2.**

- **Card:** "Eugene Airport".
- **Answer:** "Airport".
- **Shared words:** airport, so 1.
- **Score:** 1 ÷ 1 = **1.0, so right ✅**, even though "Airport" alone says nothing.

### Numbers: be within 25%

Compute **1 − (difference ÷ the bigger number)**. If it is **0.75 or more**, the answer is right. In other words,
the answer may be up to about **25% off**.

- **Card altitude 154, answer 120.** Difference 34, bigger number 154, so 1 − 34/154 = 0.78: **right ✅**.
- **Card runway length 8,000 ft, answer 7,000.** 1 − 1,000/8,000 = 0.875: **right ✅**.
- **Card 8,000 (feet), answer 2,438 (the same length in metres).** 1 − 5,562/8,000 = 0.30: **wrong ❌**. The answer
  is correct, but in another unit.

### Places: within 10 km

Latitude and longitude are checked **together**. If the model's point is less than **10 km** from the real airport,
both are right.

If the model gave only the latitude, or only the longitude, the grading cannot check it, so **it is skipped** (not
counted).

---

## Step 4: turn right and wrong into a score

Now take all graded facts of a model (over all its tasks), and count:

- **right** answers, and
- **wrong** answers.

Remember: unanswered facts are not here at all; they were skipped in Step 2.

### Accuracy

> **accuracy = right ÷ (right + wrong)**

Out of the facts that were graded, what share was right? For example, 4 right and 6 wrong gives 4 ÷ 10 = **0.4**.

### Precision, recall and F1, the sklearn way

The paper then reports **precision**, **recall** and **F1**, computed with the Python library *sklearn*. It gives
sklearn two lists, one entry per graded fact:

- **`y_pred`** (what happened): 1 if the answer was right, 0 if wrong. Here: `1 1 1 1 0 0 0 0 0 0`.
- **`y_true`** (what "should" happen): **always 1**. Here: `1 1 1 1 1 1 1 1 1 1`.

sklearn then counts three kinds of boxes:

| Box | Meaning | How many here |
|---|---|---|
| **True positive** | `y_true` 1, `y_pred` 1 | 4 (the right answers) |
| **False negative** | `y_true` 1, `y_pred` 0 | 6 (the wrong answers) |
| **False positive** | `y_true` **0**, `y_pred` 1 | **0, always**: `y_true` is never 0 |

From those boxes:

- **precision** = true positives ÷ (true positives + false positives) = 4 ÷ (4 + 0) = **1.0**;
- **recall** = true positives ÷ (true positives + false negatives) = 4 ÷ (4 + 6) = **0.4**, the same as accuracy;
- **F1** = 2 × precision × recall ÷ (precision + recall) = 2 × 1 × 0.4 ÷ 1.4 = **0.57**.

### Why precision is always 1

A "false positive" means *"the model said something, and it was wrong"*. Here, though, every graded fact has
`y_true = 1`, so sklearn can never find a false positive. **A wrong answer is never counted as a false positive; it
is counted as a "miss" (false negative) instead.**

So **precision is 1 for every model, every prompt, every time**. It needs only one right answer, no matter how many
wrong ones there are. This is why the paper says "precision is steadily equal to 1". That comes from how the lists
were built, not from how good the models are.

Since precision is always 1, the F1 is really just accuracy in disguise:

> **DOLAP F1 = 2 × accuracy ÷ (1 + accuracy)**

| Accuracy | DOLAP F1 |
|---|---|
| 0.2 (1 in 5 right) | 0.33 |
| 0.4 | 0.57 |
| 0.5 (half right) | 0.67 |
| 0.67 | 0.80 |

Because of this rescaling, an F1 of 0.80 means **two answers out of three were right**, not four out of five.

---

## Step 5: pick the best result

Each model was run in **18 setups**: 9 different prompts (short, zero-shot, 1/3/5 examples, with or without "think
step by step"), each **with and without** extra context from the graph.

Table 3 shows, for each model, **only its best setup**: the one with the highest F1 out of the 18.

Picking the best of 18 tries is like rolling a die 18 times and reporting your best roll. It makes everyone look
better, and most of all the players whose rolls vary a lot. The extreme case is Claude Sonnet 4.5:

- Its Table 3 score (**F1 0.80**) comes from a setup where only **3 facts** were graded in total. It answered almost
  nothing, and 2 of those 3 were right.
- Over all 18 setups together, the same model gets **0.67**.

---

## The whole recipe on one page

1. Ask each question 5 times; keep the most common word answer or the average number.
2. Grade **every fact the model wrote** that the card has, asked or not, including facts we showed it.
3. **Skip** every fact the model did not answer.
4. A word answer is right if 75% of the shorter text's words are shared; a number if it is within about 25%; a place
   if it is within 10 km (latitude and longitude together).
5. accuracy = right ÷ (right + wrong).
6. sklearn with `y_true` all ones gives precision = 1, recall = accuracy and F1 = 2·accuracy ÷ (1 + accuracy).
7. Report the best of the 18 setups.

---

## What this grading cannot see

| What the model does | What the grading sees |
|---|---|
| Answers **nothing** to a question | nothing; no penalty |
| Answers a question we did **not** ask, correctly | a right answer (a reward) |
| Copies back a fact we **showed** it | a right answer (a reward) |
| Answers **wrongly** | a wrong answer, but precision stays 1 |
| Gives a correct value in another unit | a wrong answer |
| Is lucky in 1 of its 18 setups | that setup is the reported score |

A model that stays quiet on hard questions and repeats easy facts gets a high DOLAP score.

---

## How our grading differs, in one breath

- We grade **only the facts we asked for**.
- A fact the model leaves unanswered is an **abstention**, and it lowers recall and F1.
- A wrong answer is a **false positive**, so precision drops: precision = right ÷ (right + wrong).
- Word answers must overlap **both ways**, and units are converted.
- We report **all setups together** as well as the best one.

Our F1 ign uses the same formula as the DOLAP F1, 2·P ÷ (1 + P), but only over the asked facts. It is there so that
the two can be compared.

Applied to the same published answers, our grading gives the proprietary models F1 0.17–0.43 instead of 0.54–0.80.
The step-by-step breakdown is in `docs/DOLAP_GAP_ANALYSIS.md`.
