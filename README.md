# Modeling NBA shot decisions with inverse optimization

**Author:** Colin Nguyen · **Course:** Math 132B (Professor Atzberger), June 2025 · revised October 2026

Shot analytics usually judge outcomes: did the shot go in? This project asks a
different question: given the shots LeBron James, Stephen Curry and James
Harden actually took in 2014–15, what "cost" were they minimising when they
chose them? Each real shot is treated as the chosen option among a few
plausible alternatives the player didn't take, and a neural network learns a
per-player score that tells the real shot apart from the alternatives.

![Attempts and field-goal percentage by distance band, 2014–15](figures/zones.svg)

## What it finds, and how far to trust it

On held-out shots, the network picks out the real shot from its alternatives
far better than chance or a linear model. For LeBron James, the real shot is
ranked first 54% of the time against 29% by chance and 29% for logistic
regression. Curry and Harden show the same pattern.

![Held-out comparison of the majority class, logistic regression and the network](figures/comparison.svg)

| Test set (20% of shots, scored once) | Accuracy | ROC AUC | Real shot ranked first (chance) |
|---|---|---|---|
| **LeBron James** · always "not chosen" | 0.724 (0.717–0.731) | 0.5 | – |
| logistic regression | 0.724 | 0.544 (0.511–0.576) | 0.29 (0.22–0.36) |
| network, notebook's loss | 0.755 | 0.757 | 0.53 |
| network, plain BCE (chosen on validation) | 0.741 | 0.747 (0.703–0.788) | 0.54 (0.47–0.62) · chance 0.29 |
| **Stephen Curry** · always "not chosen" | 0.723 (0.714–0.730) | 0.5 | – |
| logistic regression | 0.723 | 0.555 (0.516–0.592) | 0.32 (0.24–0.39) |
| network, notebook's loss | 0.721 | 0.704 | 0.46 |
| network, plain BCE (chosen on validation) | 0.723 | 0.704 (0.667–0.738) | 0.49 (0.41–0.57) · chance 0.30 |
| **James Harden** · always "not chosen" | 0.725 (0.716–0.732) | 0.5 | – |
| logistic regression | 0.725 | 0.578 (0.545–0.612) | 0.26 (0.19–0.33) |
| network, notebook's loss | 0.759 | 0.784 | 0.62 |
| network, plain BCE (chosen on validation) | 0.773 | 0.794 (0.754–0.832) | 0.64 (0.56–0.72) · chance 0.29 |

Brackets are 95% bootstrap intervals over test shots; every interval is in
`results/results.json`. Three caveats matter more than the numbers:

- **Accuracy is the wrong yardstick.** About 72% of rows are generated
  alternatives, so always answering "not chosen" scores 0.72. Logistic
  regression does exactly that. The network adds at most five points of
  accuracy; its real gain is in ranking (AUC, and the real shot ranked first).
- **The alternatives are synthetic, and the network partly learns the
  generator.** The generator proposes more open shots than players actually
  got: the closest defender was more than 6 ft away for 9–21% of real shots
  but for 26–29% of generated alternatives (`results/artifacts.json`). So
  "a tightly guarded option is more likely the real one" is partly a fact
  about the generator, not only about the player. A second artifact was
  checked and ruled out as the driver: about 10% of alternatives carry a zone
  label that doesn't match their distance, but removing or recomputing the
  zone flags changes validation AUC by at most 0.03.
- **The "KKT" part does not shape the cost.** See below.

![Learned importance of each factor, by player](figures/importance.svg)

Defender distance and shot distance move every player's score most, as in the
course run; Harden's score also leans on the deep-three flag. Read the
defender-distance bar with the caveat above: some of its weight reflects how
the alternatives were generated. The bars average five training seeds, and the
lines show how much they vary; for Harden, the ranges for defender distance
and deep three overlap, so their order is not settled.

## Corrections to the 2025 write-up

**Validation BCE.** The original table had Curry's and Harden's values swapped.
From the saved training history (`results/original_run_2025-06-04.json`, the
run behind the notebook's printed outputs), final-epoch validation BCE was:

| Player | Final validation BCE | Lowest during training | Epochs before early stopping |
|---|---|---|---|
| LeBron James | 0.489 | 0.483 | 61 |
| Stephen Curry | 0.536 | 0.528 | 104 |
| James Harden | 0.509 | 0.499 | 39 |

Those numbers came from a row-level random split, in which near-copies of a
validation shot were in training, so they are optimistic. The held-out table
above replaces them.

**"All models converged to low KKT residuals" was wrong.** At the end of
training, the complementary-slackness term was −213.7 (LeBron), −347.2 (Curry)
and −142.4 (Harden), and still falling. As implemented:

- the term is λ·mean(g), which is signed, so minimising it rewards large
  multipliers instead of penalising violations;
- the constraints (shot clock ≥ 0, defender within 10 ft, shot within 35 ft)
  were applied to standardised features, not to seconds and feet;
- the constraints depend only on the inputs and the multipliers, never on the
  network's weights. Complementary slackness and primal feasibility therefore
  cannot change the learned cost; only the "stationarity" term reaches the
  network, and it pushes the cost to be flat.

A corrected version (|λ·g| and relu(g) in real units, `kkt_fixed`) trains
identically to plain BCE, which confirms the last point. Validation chose
plain BCE. The method is best described as a learned choice model over
generated alternatives, not as recovered KKT conditions.

## How it works

1. **Data.** Every 2014–15 shot by the three players (978, 968 and 1,054
   shots). Features: shot distance, closest defender's distance, shot clock,
   dribbles, the player's field-goal percentage so far in the game, a distance
   band (0–4, 4–14, 14–22, 22+ ft) and the game stage.
2. **Alternatives.** For 800 sampled shots per player, up to three perturbed
   versions of the same moment (different distance, defender spacing, shot
   clock) are generated as options the player didn't take
   (`nbashots/alternatives.py`).
3. **Model.** An MLP (12 → 64 → 32 → 16 → 1, LeakyReLU, batch norm, dropout)
   scores each option; trained with binary cross-entropy and AdamW, with early
   stopping on validation (`nbashots/model.py`).
4. **Evaluation.** Split by shot (60/20/20), so a shot and its alternatives
   never straddle a split. Features are standardised on training rows only.
   The model variant is chosen on validation, and the test set is scored once
   (`scripts/evaluate.py`).

Two feature bugs in the course notebook are fixed in the package: shots were
ordered by the game clock as text (so 9:59 came before 11:00), and the running
FG% was shifted across games.

## Data

The shot logs are **not included**. They are the public Kaggle dataset
[NBA shot logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs)
(dansbecker; 128,069 shots, 2014–15 season, "scraped from NBA's REST API"). Its
licence on Kaggle is listed as **unknown**, so the data and any row-level
derivative stay out of this repository; only aggregates are published.

```bash
# needs a free Kaggle account and the kaggle CLI
kaggle datasets download dansbecker/nba-shot-logs --unzip -p data/
shasum -a 256 data/shot_logs.csv
# e9c7fcd71e6f81ade108841449fe1bccda072ed98fae2b5dc86a3b9f29fe5c7c
```

## Running it

Python 3.12 with [uv](https://docs.astral.sh/uv/); about two minutes on a laptop CPU.

```bash
uv run --python 3.12 --with-requirements requirements.txt python scripts/evaluate.py          # results/results.json
uv run --python 3.12 --with-requirements requirements.txt python scripts/diagnose_artifacts.py # results/artifacts.json
uv run --python 3.12 --with-requirements requirements.txt python scripts/export_demo.py demo.json
python3 scripts/make_figures.py demo.json                                                      # figures/*.svg
```

Set `SHOT_LOGS=/path/to/shot_logs.csv` if the file is not at `data/shot_logs.csv`.

## Layout

| Path | Contents |
|---|---|
| `nbashots/` | Data loading and features, alternative generation, the cost network |
| `scripts/` | Evaluation, artifact check, demo export, figures |
| `results/` | `results.json` (held-out comparison), `artifacts.json`, `original_run_2025-06-04.json` (the course run's saved history) |
| `figures/` | README figures (SVG) |
| `notebooks/course_project_2025.ipynb` | The course notebook as submitted, with a note on its known issues |

## Licence

MIT (`LICENSE`). The shot-log data is not covered and is not included.
