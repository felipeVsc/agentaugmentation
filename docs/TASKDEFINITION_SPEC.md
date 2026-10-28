# SPEC: Task Definition

> Component spec for the task construction step of the paper *"Data Augmentation in Property Graphs using agentic AI"* (draft of October 2026, `docs/infosysnewversion.pdf`): Phase 1 of Algorithm 1 (lines 1–7) and Algorithm 2 (*Task Definition Generation*, §3.3 of the draft). The previous draft numbered this algorithm 4; [§12](#12-changes-for-the-october-2026-draft) lists what changed.
>
> This component produces a **fixed task set as CSV files**. Strategy A and Strategy B both read the same files and never generate tasks themselves, so every strategy and model is evaluated on exactly the same tasks.
>
> Where the draft is ambiguous, the v1 choice is marked **[ASSUMPTION]** and listed in [Open questions](#11-open-questions).

---

## 1. Purpose

Given a property graph and a node type `t`, build a set of tasks `(n, H, G)`:

- `n`: a node of type `t`;
- `H`: properties of `n` whose values are hidden and must be predicted;
- `G`: `(property, value)` pairs from `n` and its context that are given to the model.

Tasks are stratified by node centrality and property popularity so that they cover a range of difficulty, and each task gets its difficulty score.

## 2. Scope

**In scope**
- Snapshotting the graph data needed for the task set.
- Computing centrality, context, and property popularity (Phase 1).
- Computing the quartile partitions of nodes and properties, then sampling tasks (Algorithm 2).
- Computing the difficulty of each task.
- Writing the task CSV, the ground truth CSV, and a manifest.

**Out of scope**
- Prompting models, running strategies, or scoring. Those read this component's outputs.
- Building prompt text. The CSV holds structured data; each strategy renders its own prompts from it.

## 3. Pipeline

```
graph (Neo4j) ──► 1. snapshot ──► 2. centrality ──► 3. context ──► 4. popularity
                                                                        │
                                                          5. quartile partitions
                                                                        │
       tasks.csv  ◄── 8. write ◄── 7. difficulty ◄── 6. sampling (Alg. 2)
       ground_truth.csv
       manifest.json
```

Every step writes its output to `<output_dir>/intermediate/`, so a step can be re-run without redoing the previous ones, and the final task set can be traced back to the exact graph state.

## 4. Steps

### 4.1 Snapshot
- Export from Neo4j the nodes of type `t`, all elements reachable within the context definition (§4.3), and their properties. **[ASSUMPTION]** The graph source is Neo4j.
- Save as files (Parquet) with a content hash. All later steps read the snapshot, not the live database, so the task set is reproducible even if the database changes.
- By default the **whole graph** is exported, so that PageRank is computed on the real graph and not on a subgraph. Optional `neo4j.labels` / `neo4j.relationship_types` filters restrict the export.
- Output: `intermediate/snapshot/nodes.parquet` and `edges.parquet`. Properties are stored as JSON with a parallel map of value types, so dates survive the round trip.
- Connection settings come from the `neo4j` config section; the user and password are read **only** from environment variables (`NEO4J_USER`, `NEO4J_PASSWORD`, optionally from a `.env` file) and never written to configs, manifests, or logs.

### 4.2 Centrality
- Compute `cent(e)` for nodes of type `t` and for every element that can contribute a property to a context (neighbor nodes, nodes on `n:1` paths, edges).
- Default measure: PageRank (pluggable).
- `measure: degree` is the draft's alternative: "the probability of the node being source or target of a relationship when drawing randomly". That probability is `deg(n) / 2|E|`; after the per-type normalization below it equals `deg(n) / max deg`, which is what `degree` computes.
- Normalize to `(0, 1]`: divide by the maximum value **per element type**. **[ASSUMPTION]**
- Replace 0 with `epsilon` (config) so that `−log(cent)` stays finite.
- Edge centrality **[ASSUMPTION]**: mean of the normalized centralities of the edge's two endpoints (pluggable).

Output: `intermediate/centrality.parquet` → `element_id, element_kind (node|edge), element_type, raw_centrality, centrality`.

### 4.3 Context
`context(n)` = properties of:
1. `n` itself;
2. direct neighbor nodes;
3. edges between `n` and its direct neighbors;
4. elements on paths starting at `n` where every relationship is `n:1` (e.g. Airport → City → Country).

Rules:
- A relationship type `R` from type `A` to type `B` is `n:1` if every `A` node has at most one outgoing `R` to a `B` node. Detect this from the data and record the result in the manifest.
- `n:1` path length is bounded by `context_max_hops` (config).
- Each context property is recorded with its origin element, path, and value.

Paths: `Airport-[IN]->City` for outgoing relationships, `Airport<-[R]-City` for incoming ones. Direct neighbors are taken in both directions; `n:1` paths follow outgoing relationships only. **[ASSUMPTION]**

Output: `intermediate/context.parquet` → `node_id, prop_key, element_id, element_type, path, value` (plus kind, value type, popularity, centrality, and an `eligible` flag), for the **sampled nodes** only, since only they are needed to draw G.

### 4.4 Property identity
A property is identified by where it comes from, not only its name, so `name` on City and `name` on Country are different properties.

- `prop_key` = `<path>.<property_name>`, where `path` is the relationship path from `n` to the origin element. Examples:
  - `Airport.elevation` (property of `n`)
  - `Airport-[IN]->City.name`
  - `Airport-[IN]->City-[IN]->Country.name`
  - `Airport-[HAS_ROUTE]->.distance` (edge property)
- For direct neighbors reached through a non-`n:1` relationship, several elements can share the same `prop_key`. Each occurrence is kept as a separate candidate with its own `element_id`.

### 4.5 Popularity
- `pop(p)` = number of elements of the origin type with a non-null value for the property ÷ number of elements of that type.
- Computed per `(element_type, property_name)` over the whole snapshot.

Output: `intermediate/popularity.parquet` → `element_type, property_name, non_null, total, popularity`.

### 4.6 Property eligibility
Before sampling, filter candidate properties:
- Properties in `exclude_properties` (config) are never used in H or G (e.g. internal ids, timestamps added by the ETL, embeddings).
- Properties in `never_hide` (config) may appear in G but never in H.
- An H candidate must have a **non-null value on `n`**, otherwise the task can't be scored.
- **[ASSUMPTION]** Only properties with scalar values (string, number, boolean, date) are eligible in v1. Lists and maps are excluded.

## 5. Sampling (Algorithm 2)

### 5.1 Scores and strata
Nodes and properties are stratified on two scales. As in the draft, H and G share **one** property partition (Algorithm 2 draws H from 𝒫ⱼ and G from 𝒫ₖ of the same set 𝒫):

| Factor | Score | Quartiles computed over |
|---|---|---|
| Node | `cent(n)` | `Q0..Q4` over `n ∈ N` |
| Property (H and G) | `pop(p)` | `P0..P4` over the distinct eligible `(element type, property)` pairs of the elements in `N ∪ context(N)` (`N` only with `use_context: false`), each pair counted once |

With `pop(p)` the score is the same for every element of a type, so the draft's set 𝒫 of `(p, m_p)` pairs reduces to one value per `(type, property)`. Without context, the partition is exactly the one over the node type's own properties.

`G_strata_by: score` (not the default) restores the earlier rule: H by `pop(p)` over the node type's properties, and G by its own quartiles `PG0..PG4` of `m_p = pop(p) × cent_p(e)` over every eligible `(element, property)` occurrence. H and G then use two different partitions, which the draft does not do.

**Why H uses `pop(p)` only.** Every H property belongs to `n`, so `cent_p(e) = cent(n)` for all of them. Stratifying H by `pop(p) × cent(n)` would repeat the node factor: a low-centrality node would have almost no properties in the high H stratum, and a high-centrality node almost none in the low one. Those combinations would be skipped, and the node effect and the property effect could not be separated in the analysis. With `pop(p)` alone, the H stratum is independent of the node stratum, and cases such as "a common property of an obscure node" are sampled.

**Deviation from the draft (author's decision).** The draft's score is `m_p = pop(p) × cent_p(e)`. This spec stratifies by `pop(p)` alone, for the reason above, and applies it to H and G alike so that both keep sharing one partition, as in the draft (open question 10). For G drawn from context, `cent_p(e)` would also bring in the neighbor's centrality; that factor is left to the difficulty.

Three strata are used on every scale:

| Stratum | Draft subsets | Node range | H property range | G property range |
|---|---|---|---|---|
| `low` | `Q1`, `𝒫1` | `[Q0, Q1)` | `[P0, P1)` | `[P0, P1)` |
| `mid` | `Q2 ∪ Q3`, `𝒫2 ∪ 𝒫3` | `[Q1, Q3)` | `[P1, P3)` | `[P1, P3)` |
| `high` | `Q4`, `𝒫4` | `[Q3, Q4]` | `[P3, P4]` | `[P3, P4]` |

As in the draft, intervals are closed at the lower bound and open at the upper one, except the last, which is closed at both ends. Each value belongs to exactly one stratum, and a value equal to a quartile boundary goes to the stratum **above** it.

**Ties.** When `Q3 = Q4` (common for H, where many properties have popularity 1.0), `high = [Q3, Q4]` holds the tied values, so no special rule is needed. **[ASSUMPTION]** When `Q0 = Q1` (e.g. more than a quarter of the elements share the minimum PageRank, as nodes without incoming edges do), `low = [Q0, Q1)` would be empty; values equal to `Q0` then go to `low`. The manifest lists the properties in each H stratum and the count per G stratum, so the effect is visible.

This changes only the **sampling** of H and G. The difficulty metric (§6) still uses `pop(p) × cent_p(e)` for every property in H and G.

### 5.2 Procedure
The quartile partitions (§5.1) are computed **before** sampling and passed to it, as in the draft, where Algorithm 2 takes `𝒫1..𝒫4` and `Q1..Q4` as input. `TaskBuilder.build(partitions)` accepts fixed partitions, so the strata can be held constant across graph versions (used by the stability test, §10).

```
input: N, max_n, max_H, max_G, partitions (Q, PH, PG)
for node_stratum in [low, mid, high]:
    NODES ← sample up to max_n nodes with cent(n) in node_stratum
    for n in NODES:
        for size_H in 1..max_H:
            for size_G in 1..max_G:
                for H_stratum in [low, mid, high]:
                    H ← sample size_H properties of n itself, pop(p) in H_stratum
                    for G_stratum in [low, mid, high]:
                        G ← sample size_G properties from properties(n ∪ context(n)) − H,
                            pop(p) in G_stratum (same partition as H)
                        emit (n, H, G)
```

The same H is reused across the three G strata, so for a given node and H the only variable is the given context. This matches the loop nesting of Algorithm 2 (lines 7–14).

### 5.3 Rules
- Sampling is **without replacement** inside H and inside G.
- **Distinct prop_keys in G [ASSUMPTION]:** a prop_key shared by several elements (e.g. `Airport-[HAS_ROUTE]->Airport.name`) is drawn at most once per task (see open question 7).
- **Value leakage:** G excludes any candidate whose value equals an H value after canonicalization (case, whitespace, `2` = `2.0`), or, for string H values of at least `leak_min_substring_len` characters (config, default 4), contains it.
- **No overlap:** G excludes every `prop_key` in H. A `leakage_pairs` list in the dataset config adds known equivalents (e.g. `Airport.iata` ↔ `Airport.code`) so they're never split across H and G.
- **Not enough candidates:** if a stratum has fewer candidates than `size_H` or `size_G`, the combination is **skipped** (not filled with fewer properties, not filled from other strata), and the skip is counted in the report. **[ASSUMPTION]**
- **Duplicates:** tasks with identical `(n, H, G)` are emitted once.
- **Determinism:** one master seed; node and property lists are sorted by id before sampling; a child random generator is derived for each `(node_id, size_H, size_G, H_stratum, G_stratum)` combination. Adding or removing one node does not change the tasks generated for the others.

Expected count before skips and dedup: `27 × max_n × max_H × max_G` (the draft's `Ensure` clause), or `27 × max_n × |sizes_H| × |sizes_G|` with explicit sizes.

**Without context** (`use_context: false`), every G property belongs to `n` and the shared partition is computed over `N` only. (With `G_strata_by: score`, `cent_p(e) = cent(n)` for every G property and the G strata repeat the node strata, as §5.1 explains for H.) A node stratum with fewer than `max_n` nodes yields fewer tasks; the draft's "draw up to `max_n` nodes" allows this.

## 6. Difficulty

```
Difficulty(T) = ∛( |H| / (|H| + |G|)
                  × −log₂(cent(n))
                  × Σ_{p ∈ H ∪ G} −log₂(pop(p) × cent_p(e)) )
```

- Log base 2: this is the base that reproduces Table 1 of the draft.
- `cent_p(e)` is the centrality of the property's origin element (for H, that is `n`).
- The three factors are stored separately, so other formulas can be tested later without regenerating tasks.
- If `cent(n) = 1`, the node factor is 0 and so is the difficulty. This is kept as computed and counted in the report.

## 7. Outputs

All files are written to `<output_dir>/` (e.g. `tasks/airports/v1/`). A task set is immutable once written: any change produces a new version directory.

### 7.1 `tasks.csv` — read by Strategy A and Strategy B

One row per task. H and G are JSON-encoded columns.

| Column | Type | Description |
|---|---|---|
| `task_id` | str | Stable hash of `(dataset, node_id, sorted H prop_keys, sorted G (prop_key, element_id))`. The element id is included because a prop_key can come from several elements |
| `task_set_version` | str | Version of the task set |
| `dataset` | str | Dataset name |
| `node_type` | str | `t` |
| `node_id` | str | Graph id of `n` |
| `node_centrality` | float | `cent(n)` |
| `node_stratum` | str | `low` / `mid` / `high` |
| `H_stratum`, `G_stratum` | str | Strata used for sampling |
| `size_H`, `size_G` | int | |
| `H` | JSON | List of objects: `{prop_key, property_name, element_type, path, popularity, centrality, score}` — **no values** |
| `G` | JSON | List of objects: `{prop_key, property_name, element_type, element_id, path, value, popularity, centrality, score, hops}` |
| `difficulty` | float | §6 |
| `diff_ratio` | float | `|H| / (|H|+|G|)` |
| `diff_node_surprise` | float | `−log₂(cent(n))` |
| `diff_prop_surprise` | float | Sum term |
| `node_centrality_raw` | float | Raw PageRank / degree of `n` |
| `diff_prop_surprise_H`, `diff_prop_surprise_G` | float | The sum term split between H and G |
| `n_G_own`, `n_G_context`, `G_has_own_property`, `G_max_hops` | int / bool | Where G comes from (`METRICS.md` §3.1) |
| `G_candidates` | int | Eligible G properties available for `n` |

Example `H` and `G` values:

```json
H: [{"prop_key": "Airport.elevation", "property_name": "elevation",
     "element_type": "Airport", "path": "Airport"}]

G: [{"prop_key": "Airport.name", "property_name": "name", "element_type": "Airport",
     "element_id": "a_123", "path": "Airport", "value": "Guarulhos International",
     "popularity": 1.0, "centrality": 0.82},
    {"prop_key": "Airport-[IN]->City.name", "property_name": "name", "element_type": "City",
     "element_id": "c_45", "path": "Airport-[IN]->City", "value": "São Paulo",
     "popularity": 1.0, "centrality": 0.91}]
```

### 7.2 `ground_truth.csv` — read only by the evaluation

Kept in a **separate file** so the strategies cannot accidentally read the hidden values.

| Column | Description |
|---|---|
| `task_id` | |
| `prop_key` | One row per property in H |
| `value` | Ground-truth value as stored in the graph |
| `value_type` | `string` / `integer` / `float` / `boolean` / `date` |

### 7.3 `manifest.json`
- Config used, master seed, code version (git commit).
- Snapshot hash and graph statistics (node/edge counts per type).
- Detected `n:1` relationships.
- Quartile boundaries `Q0..Q4` (nodes), `PH0..PH4` (H properties) and `PG0..PG4` (G properties).
- Report: tasks per `(node_stratum, H_stratum, G_stratum, size_H, size_G)`, skipped combinations with reason, duplicates removed, zero-difficulty count.

### 7.4 CSV conventions
- UTF-8, comma separator, RFC 4180 quoting, header row.
- Values in `G` and `ground_truth.csv` keep their original type in JSON (numbers stay numbers).

## 8. Configuration

```yaml
dataset:
  name: airports
  source: neo4j               # neo4j | snapshot (with snapshot_dir)
neo4j:
  uri: bolt://localhost:7687  # overridden by NEO4J_URI
  database: null              # overridden by NEO4J_DATABASE
  id_function: elementId      # elementId (Neo4j 5) | id (Neo4j 4)
  # user / password: only from NEO4J_USER / NEO4J_PASSWORD
  node_type: Airport
  context_max_hops: 3
centrality:
  measure: pagerank
  epsilon: 1.0e-6
sampling:
  max_n: 5
  max_H: 3
  max_G: 3
  seed: 42
  sizes_H: null          # explicit sizes instead of 1..max_H, e.g. [1]
  sizes_G: null          # explicit sizes instead of 1..max_G, e.g. [1, 3, 5]
  use_context: true      # false: G only from n's own properties; G quartiles over N only
  G_strata_by: popularity  # popularity: H and G share one pop(p) partition (default) | score: G by pop(p) × cent_p(e)
properties:
  exclude_properties: [...]
  never_hide: [...]
  leakage_pairs: [[...], [...]]
output_dir: tasks/airports/v1/
```

CLI:
- `tasks check-connection --config configs/airports.yaml`: connects and lists labels and relationship types.
- `tasks snapshot --config ...`: exports the snapshot only.
- `tasks build --config ...`: builds the task set; reuses the snapshot in `intermediate/snapshot/` if present (`--refresh-snapshot` to export again). Refuses to overwrite an existing task set (`--overwrite` for development only).

Implementation: `src/taskdef/`; example config: `configs/airports.yaml`.

## 9. Validation (run automatically before writing)

- Every `task_id` is unique.
- No `prop_key` appears in both H and G of the same task, and no `leakage_pairs` are split.
- Every H property has exactly one row in `ground_truth.csv` with a non-null value.
- No ground-truth value appears in the G of the same task, using the value-leakage rule of §5.3. (A plain string search on the G column would match short values such as `2` inside centralities like `0.82`.)
- Every `size_H`, `size_G` matches the length of H, G.
- Strata in each row match the recorded quartile boundaries.

The build fails if any check fails, and nothing is written to `tasks.csv` / `ground_truth.csv`.

## 10. Testing and acceptance

- **Toy graph tests**: a small hand-built graph with known centralities, contexts, popularities, and expected tasks.
- **Difficulty test**: all four rows of Table 1 of the draft reproduced to two decimals.
- **Determinism test**: two builds with the same config and snapshot produce byte-identical files.
- **Stability test**: removing one node from the snapshot leaves the tasks of other nodes unchanged, given the same quartile partitions (passed to `build`).
- **Strata test**: values on a quartile boundary go to the stratum above; `Q3 = Q4` and `Q0 = Q1` ties as in §5.1.

Acceptance: the airports task set builds, passes §9, and the manifest report shows every stratum populated or explains why it isn't.

## 11. Open questions

1. **Identity of the node.** If the node's name/code isn't in G, the model may not know which entity it's predicting for. Should identifying properties always be given (outside G, not counted in difficulty), or is that part of the difficulty?
2. **Not enough candidates in a stratum:** skip the combination (v1), or draw fewer properties?
3. ~~Algorithm 4 line 15 uses `cent_p(n)`, line 17 uses `cent_p(e)`.~~ Resolved in the October 2026 draft: both H and G use `m_p = pop(p) × cent_p(e)` from the single set 𝒫, with `e` the origin element (for H, `e = n`).
4. ~~Quartiles of G property scores: over all occurrences globally, or per node?~~ Resolved: one partition for H and G over the distinct properties of `N ∪ context(N)` (§5.1).
5. **Centrality normalization** per element type (v1) or over the whole graph? This changes the difficulty values.
6. **Edge centrality** definition.
7. **Multi-valued neighbors:** when a node has several neighbors via the same relationship (e.g. many routes), can G contain properties from more than one of them?
8. **Non-scalar properties** (lists, maps): excluded in v1. Needed for any dataset?
9. **Sparse properties in H:** low-`pop(p)` properties are often null on a given node, so the low H stratum may be empty for many nodes, more often for low-centrality nodes if they have fewer filled properties. The manifest report must show skips per `(node_stratum, H_stratum)` to check this.
10. **Property stratification.** The draft stratifies H and G by `m_p = pop(p) × cent_p(e)`; this spec uses `pop(p)` for both, still as one partition (§5.1). The paper text should say so.
11. **Parameter names.** The draft's Algorithm 1 takes `n_N, n_H, n_G` "replacing `max_n, max_H, max_G`", but Algorithm 2 still uses `max_n, max_H, max_G` and loops `size_H ∈ [1, max_H]`. Until the draft settles this, the config keeps `max_n, max_H, max_G` with the loop semantics. If `n_H, n_G` are meant as fixed sizes rather than maxima, the size loops go and the count becomes `27 × n_N`.
12. **G must contain a property of `n`?** The draft raises (as "TO DISCUSS") that G and H should be non-empty and that G should contain at least one property of `n` itself, because a G made only of context properties (e.g. the city's population and the country's name) is much harder than the difficulty score says. v1 does not enforce it; enforcing it would be a rejection rule in §5.3 (draw G again or skip when no own property is drawn).
13. **Context definition.** The draft is still discussing whether context is a subgraph of the graph type (as in [1]) or a set of `(element, property)` pairs over direct neighbors and `n:1` paths. v1 implements the second (§4.3), which is what both versions of the definition compute.
14. **`|G|` barely changes the difficulty.** `|H|/(|H|+|G|) × Σ_{H∪G} s_p = |H| × mean_{H∪G}(s_p)`, with `s_p = −log₂(pop(p) × cent_p(e))`. Adding G properties lowers the difficulty only if they are easier than the current mean, and leaves it unchanged if they are as hard. Without context every `cent_p(e) = cent(n)`, and on a dataset where most properties have `pop(p) = 1` all `s_p` are equal, so tasks with `|G| = 1, 3, 5` get the same difficulty. Assumption 1 of the draft ("the fewer properties in G, the more difficult") is then not reflected.
15. **Popularity is close to binary on a clean dataset.** When most properties are always filled (`pop = 1`), `P1 = P3 = P4 = 1`: the mid stratum is empty, high holds every complete property, and low holds only the sparse ones, which most nodes lack (H skips). Strata by rank, or by context properties, would be needed to fill the nine `(H, G)` cells.

## 12. Changes for the October 2026 draft

Compared with the previous draft (`docs/paper2.pdf`, Algorithm 4), the new Algorithm 2 changes:

| Change in the draft | Effect here |
|---|---|
| Quartile subsets are half-open, `[l_{i-1}, l_i)`, with the last closed `[l_3, l_4]` (nodes and properties) | §5.1 strata; a boundary value moves to the stratum above. The old `Q3 = Q4` rule is no longer needed; a `Q0 = Q1` rule replaces it |
| Quartile partitions are computed before sampling and passed to Algorithm 2 | §3 step 5, §5.2; `TaskBuilder.partitions()` and `build(partitions)` |
| A single set 𝒫 of `(p, m_p = pop(p) × cent_p(e))` pairs over `N ∪ context(N)` defines the property quartiles | H and G share one partition, by `pop(p)` (deviation, §5.1); open questions 3 and 4 resolved |
| Nodes: "draw **up to** `max_n` nodes" | Already the behavior; now stated in §5.2 |
| Centrality may be replaced by the probability of being an endpoint of a relationship | `measure: degree` (§4.2), already implemented |
| Algorithm 1 parameters renamed `n_N, n_H, n_G`; task written `(n, G, H)` | Not applied (open question 11); column order in `tasks.csv` is unchanged |
| Difficulty formula and Table 1 | Unchanged (§6) |

Not affected: the published DOLAP 2026 task design (`taskdef/published.py`, `tasks import-published`), which reproduces the published paper and has its own strata. Task sets built with the previous rules are not rebuilt; the manifest records the boundaries, and a rebuild goes to a new version directory (§7).
