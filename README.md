# Modeling NBA shot decisions with inverse optimization

**Author:** Colin Nguyen · **Course:** Math 132B (Professor Atzberger), June 2025 · revised October 2026

Shot analytics usually judge outcomes: did the shot go in? This project asks a
different question: given the shots LeBron James, Stephen Curry, and James
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

![Held-out comparison of the majority class, logistic regression, and the network](figures/comparison.svg)

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
training, the complementary-slackness term was −213.7 (LeBron), −347.2 (Curry),
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

1. **Data.** Every 2014–15 shot by the three players (978, 968, and 1,054
   shots). Features: shot distance, closest defender's distance, shot clock,
   dribbles, the player's field-goal percentage so far in the game, a distance
   band (0–4, 4–14, 14–22, 22+ ft), and the game stage.
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

## v2: inverse optimization that constrains the cost (pre-registered; stopped at criterion a)

Everything above is the **v1 audit**: the course model, its corrected
numbers and its held-out comparison. Its pipeline (`nbashots/`, `scripts/`)
is unchanged. v2 is a separate attempt at inverse optimization in which the
optimality conditions actually depend on the learned cost. This section was
written and committed before any v2 code or results.

### Forward problem

A player who has the ball chooses **where to shoot from** and **how long to
hold it**, to maximise expected points minus a personal cost:

    maximise over d = (s, t):   EP(s, t; x) - C(s, t)
    subject to   g1 = -s       <= 0     (feet; distance cannot be negative)
                 g2 = s - 35   <= 0     (feet; longer shots are end-of-clock heaves, excluded)
                 g3 = -t       <= 0     (seconds; touch time cannot be negative)
                 g4 = t - k    <= 0     (seconds; cannot hold the ball past the shot clock)

**Decision variables (chosen by the player):**
- `s`, shot distance in feet (`SHOT_DIST`). Where to shoot is the central
  shot-selection choice; the data has distance but no court x/y, so direction
  (corner, wing, top) cannot be modelled.
- `t`, touch time in seconds (`TOUCH_TIME`): how long the player held the ball
  before shooting. It stands in for dribbles (correlation 0.91 in the data)
  and is continuous, which the optimality conditions need. Shot-clock time at
  release is not a separate choice: it equals k - t.

**Context (given, not chosen):** closest defender distance in feet
(`CLOSE_DEF_DIST`), the shot clock at the catch k = `SHOT_CLOCK` +
`TOUCH_TIME` in seconds, the period, and the player. Defender distance is
treated as context even though a player can partly create space; the data
cannot separate the two. Score margin during the game is not in the data
(only the final margin, which is an outcome), so it is not used.

**Shots used:** touch time above 0, shot clock recorded, k at most 24 s, and
shot distance at most 35 ft. This keeps 108,400 of 128,069 shots, and 809,
750 and 843 for LeBron James, Stephen Curry and James Harden. (3,358 shots have
touch time 0 or less, 10,970 have k above 24 s, 5,567 have no shot clock and
249 are beyond 35 ft.)

**Expected points EP(s, t; x)** = P(make | s, t, x) x (2 + P(three | s)).
P(make) is a small smooth network fitted to makes and misses; P(three | s) is
a logistic curve in distance, because the 3-point line is 22 ft in the corners
and 23.75 ft elsewhere and the data has no position. EP is fitted once, checked
on held-out shots, then **frozen**. It is trained on all other players' shots
plus the three players' training shots; their validation and test shots are
held out of it.

**Personal cost** C(d) = b·d + ½ dᵀQd with Q positive semi-definite (Q = LLᵀ),
one (b, Q) per player, d in feet and seconds. This is the quadratic first
version; an input-convex network is added only if the quadratic clearly
underfits (training stationarity residual explains less than 20% of the
variance of ∇EP).

**Why the flat cost is ruled out.** The optimality conditions are
∇EP(dᵢ) - ∇C(dᵢ) = Σⱼ λᵢⱼ ∇gⱼ(dᵢ). EP is fixed and its gradient is not zero
at most observed shots, so C = constant cannot satisfy them unless every shot
sits on a constraint boundary. The cost must explain why players do not simply
take the highest-EP option.

### Loss

For each player, over training shots i, with multipliers λᵢⱼ = softplus(θᵢⱼ) >= 0
(dual feasibility):

    stationarity     r_i = ∇_d EP(d_i) - ∇_d C(d_i) - Σ_j λ_ij ∇g_j(d_i)    (autograd through EP and C)
    loss             mean ||r_i||²  +  ρ · mean Σ_j λ_ij · (-g_j(d_i))

The complementary-slackness penalty λ·(-g) is never negative (λ >= 0, g <= 0)
and is zero only when each multiplier is zero or its constraint is tight.
Primal feasibility holds for every observed shot by the filters above. ρ is
chosen on validation.

### Success criteria (fixed before any results)

Splits: each player's shots, 60/20/20 by shot, seeded. All choices (ρ, the
expected-points model's settings) are made on validation; the test set is
scored once.

**EP check (prerequisite).** On held-out shots, the predicted make probability
is within 0.03 of the observed rate in every decile, the expected calibration
error is at most 0.02, and the log loss beats a logistic regression on
distance alone.

**(a) Synthetic recovery. If this fails, stop and report; do not fit real
players.** Three synthetic players with known quadratic costs (preferring the
rim, the 3-point range and a mid-range pull-up), on the real players' training
contexts, choose d by solving the forward problem exactly (grid search, 0.25
ft by 0.1 s), plus Gaussian noise (sd 1.0 ft and 0.25 s), clipped to the
feasible set. A cost counts as **recovered** if both hold for all three:
- relative parameter error ||θ̂ - θ*|| / ||θ*|| <= 0.25, with θ = (b, Q);
- on held-out synthetic contexts, decisions solved with θ̂ versus θ* differ by
  a median of at most 1.0 ft and at most 0.25 s.

**(b) Held-out decision prediction on real shots.** For each test shot, solve
the forward problem with the player's learned cost and compare with the actual
(s, t). Baselines: (1) maximise EP alone (C = 0); (2) the player's mean
training decision. Metric: mean absolute error in feet and in seconds. The
model **beats** a baseline on a variable if its MAE is lower and the 95%
bootstrap interval of the paired difference is entirely below zero. v2
**works on real data** if it beats both baselines on shot distance for at
least two of the three players. Touch time is reported the same way.

**(c) Ablation.** Refit with the stationarity term removed (everything else
unchanged) and compare the learned (b, Q) and validation MAE. The stationarity
term **changes what is learned** if the parameters differ by more than 25%
(relative) and the validation MAE differs. Scored on validation, not test.

### Result: stopped at criterion (a)

The expected-points model passed its check on 11,080 held-out shots
(calibration error 0.011, worst decile gap 0.024, log loss 0.654 against
0.673 for distance alone; `results/v2_ep.json`).

**Synthetic recovery failed, so v2 was not fitted to the real players** and
criteria (b) and (c) were not run (`results/v2_synthetic.json`):

| Synthetic player | Parameter error (limit 0.25) | Decision gap, ft / s (limits 1.0 / 0.25) | Recovered |
|---|---|---|---|
| prefers the rim | 0.49 | 0.75 / 0.10 | no |
| prefers 3-point range | 1.01 | 24.75 / 0.70 | no |
| mid-range pull-up | 0.65 | 9.25 / 0.90 | no |
| *no-noise diagnostic:* rim | 0.07 | 0.00 / 0.00 | yes |
| *no-noise:* 3-point range | 0.71 | 17.75 / 0.20 | no |
| *no-noise:* mid-range | 0.44 | 0.00 / 0.00 | no (parameters) |

Why, from the diagnostics:

1. **The cost is not identified by these decisions.** The context the data
   offers (defender distance, shot clock, period) moves the optimal decision
   very little: under the 3-point cost the optimal distance varies by 0.23 ft
   (sd) across real contexts. Stationarity then pins down only one
   combination of b and Q, a ridge of costs that fit equally well. Even
   without noise, the fit reached a residual of 3·10⁻⁶ with the wrong
   curvature.
2. **Stationarity is a local condition, and EP is not concave in distance.**
   EP peaks at the rim, dips through mid-range and jumps at the arc, so a
   cost can make every observed shot a stationary point while the global
   optimum lies elsewhere. Solving with the recovered 3-point cost moves
   shots 17.75 ft away from the true cost's choices.
3. **Noise larger than the signal.** With decision noise of 1 ft and 0.25 s,
   larger than the spread the costs produce, stationarity evaluated at noisy
   (non-optimal) points biases the fit for all three.

Possible next steps, not run: fit by the likelihood of the observed decision
under a choice model over the whole feasible grid, P(d | x) ∝ exp((EP − C)/T),
which uses the global objective rather than local conditions and models noise
directly (a flat cost is still ruled out, since choices would then follow EP
alone); and fewer cost parameters, or context that moves decisions more.
`scripts/v2_real.py` is written but was not run.

## v3a: a choice model over the feasible set (pre-registered; stopped at criterion a)

The last attempt. v1 and v2 above are kept as they are. This section was
written and committed before any v3 code or results. If criterion (a) or the
identifiability check fails, v3 stops and is reported as a negative result.

### Model

Same forward problem, data filters, splits (60/20/20 by shot, seeds as in v2),
grid (0.25 ft by 0.1 s, with t <= k) and frozen expected-points model as v2.
Instead of optimality conditions, each observed decision is treated as a draw
from a choice distribution over the feasible grid:

    P(d | x) ∝ exp( (EP(d; x) - C(d)) / T )
    C(d) = β_mid·[4 < s <= 14] + β_long·[14 < s <= 22] + β_three·[s > 22] + α·t

**Five parameters per player:** three distance-band costs relative to the rim
band (a rim cost cannot be identified: adding a constant to every option
changes nothing), a touch-time cost α in points per second, and the
temperature T in points (how closely choices follow EP − C). The bands are the
notebook's. One cost per band rather than a curve keeps each number readable
("worth x points to this player"). Fitting is by maximum likelihood in the
equivalent linear form u = a·EP − b·bands − c·t with a = 1/T, β = b/T, α = c/T.
That log-likelihood is concave, so there is one maximum if the information is
well conditioned. Each observed (s, t) is rounded to its grid cell.

### Success criteria (fixed before any results)

**(a) Synthetic recovery, at v2's noise.** Three synthetic players with the
same characters as v2, in v3's parameters (T in points, β in points, α in
points per second), fixed now:

| Synthetic player | T | β_mid | β_long | β_three | α |
|---|---|---|---|---|---|
| prefers the rim | 0.15 | 0.15 | 0.25 | 0.30 | 0.05 |
| prefers 3-point range | 0.15 | 0.05 | 0.05 | −0.20 | 0.05 |
| mid-range pull-up | 0.15 | −0.15 | −0.15 | 0.10 | 0.02 |

Each uses one real player's training contexts (as in v2). A decision is drawn
from the true choice distribution, then v2's noise is added (Gaussian, sd 1.0 ft
and 0.25 s), clipped to the feasible set and rounded to the grid. A player
counts as **recovered** if:
- the relative error of θ = (T, β_mid, β_long, β_three, α) is at most 0.25;
- the band shares predicted on held-out synthetic contexts with the recovered
  and true parameters differ by a total variation distance of at most 0.05.

**(a′) Identifiability.** From the observed Fisher information at the
maximum, in the linear form (a, b_mid, b_long, b_three, c), for each synthetic
player and then each real player:
- the information matrix is positive definite;
- no pair of parameters has |correlation| above 0.95 (a ridge shows up as a
  correlation near ±1);
- every parameter's standard error is at most 25% of its estimate, or at most
  0.25 in absolute terms (in the linear form's units) when the estimate is
  smaller than 1.

A profile likelihood for each parameter of each synthetic player is also
reported. If (a) or (a′) fails on synthetic data, stop.

**(b) Held-out real shots, test set scored once.** Two baselines:
- **EP alone:** the same choice model with C = 0, so only T is fitted.
- **The player's empirical distribution:** the product of the player's
  training histograms of s (0.25 ft cells) and t (0.1 s cells), each with
  add-one smoothing, restricted to t <= k and renormalised for each shot.

Measures: mean log-likelihood per shot (nats on the shared grid), and the
mean absolute error of predicted shot distance (the model's expected s
against the actual s). v3 is **better** than a baseline for a player if the
95% paired bootstrap interval (1,000 resamples) of the log-likelihood
difference is entirely above zero. v3 **works on real data** if it beats both
baselines for at least two of the three players.

**(c) More than restating frequencies.** The empirical distribution predicts
the same shot distance whatever the context; v3 can shift with context
through EP and the shot-clock limit. On the test shots, for two context
splits, closest defender at or below versus above the player's training
median, and shot clock at the catch at or below 10 s versus above, compare the
observed difference in mean shot distance with the model's predicted
difference. v3 **explains a shift** if the observed difference's 95% bootstrap
interval excludes zero and the predicted difference has the same sign and
lies within that interval. v3 adds something beyond frequencies only if it
explains at least one shift for at least two players. A split with no
observed shift (interval includes zero) is reported as "nothing to explain".

### v3a result: stopped at criteria (a) and (a′)

All three synthetic players failed both recovery and the identifiability
check, so v3 was not fitted to the real players, and (b) and (c) were not
run (`results/v3_synthetic.json`; `scripts/v3_real.py` is written but has
never been run).

| Synthetic player | θ error (limit 0.25) | Band-share gap (limit 0.05) | Largest correlation | Failing standard error |
|---|---|---|---|---|
| prefers the rim | 5.57 | 0.04 | 0.74 | long-two cost, unbounded |
| prefers 3-point range | 0.38 | 0.03 | 0.87 | long-two cost |
| mid-range pull-up | 0.59 | 0.08 | 0.68 | long-two cost |

What the diagnostics show:

1. **No ridge this time.** Unlike v2, the parameters are not traded off along
   a ridge: the largest correlation is 0.87, and every profile-likelihood
   interval is bounded except one. The choice model fixed the problem it was
   meant to fix.
2. **The long-two band cost could not be pinned down, because the synthetic
   players almost never shot from there:** 0.2%, 0.6% and 5% of their shots
   fell in 14–22 ft. This is a weakness of the synthetic players chosen at
   pre-registration, not of the data. The real players take 14–21% of their
   shots from that band, but the rule was to stop.
3. **Noise at band edges biases the costs.** v2's noise (1 ft) moves shots
   across the 4, 14 and 22 ft edges. That pulls band costs toward zero and
   makes choices look more random: T came out 0.165 and 0.173 against a true
   0.15. Even where the decisions were well predicted (band-share gaps of 0.03
   to 0.04), the parameters were off by 38% or more.

## v3b: the same model, retested with a redesigned synthetic test (pre-registered; stopped at criterion a)

**Why the v3a test was redesigned.** v3a failed its synthetic test, and its
results above stay in the record. The diagnostics showed two flaws in the
test, not in the model:
1. **The synthetic players barely used one band.** They took 0.2%, 0.6% and
   5% of their shots from 14–22 ft, so that band's cost could not be estimated
   from any method. The real players take 12–19% of their training shots there.
2. **The noise was double-counted.** Extra distance noise (1 ft, 0.25 s) was
   layered on a model whose temperature T already describes randomness in
   choices. The test was then of a misspecified model, and the noise moved
   shots across band edges, biasing the band costs toward zero.

The model, its five parameters, the fitting, the thresholds and the real-data
criteria (b) and (c) are unchanged from v3a. Only the synthetic test changes.
This section was written and committed before any v3b run.

### Redesigned synthetic test

**(a) Well-specified recovery (gates everything after it).**
- **Three synthetic players**, one per real player. Each uses that player's
  training contexts and training sample size (485, 450 and 505 shots), and
  draws decisions from the model itself, with no added noise.
- **True parameters:** T = 0.15 points and α = 0.05 points per second for all
  three, as in v3a. The three band costs are solved for so that the
  synthetic player's predicted band shares on those training contexts match
  the real player's training-split shares, each within ±0.01:

  | Matched to | 0–4 ft | 4–14 ft | 14–22 ft | 22+ ft |
  |---|---|---|---|---|
  | LeBron James | 0.171 | 0.307 | 0.192 | 0.330 |
  | Stephen Curry | 0.122 | 0.196 | 0.124 | 0.558 |
  | James Harden | 0.168 | 0.248 | 0.156 | 0.428 |

  Only training shots are used for this; no validation or test shot informs it.
- **Five seeds per synthetic player** (seeds 0–4 for the draws): 15 datasets,
  all reported.
- **Thresholds unchanged from v3a**, applied to every dataset:
  - relative θ error ≤ 0.25;
  - band-share gap (total variation, on that player's validation contexts) ≤ 0.05;
  - identifiability: information positive definite, no |correlation| above
    0.95, and each standard error within 25% of its estimate (or 0.25 absolute
    when the estimate is below 1, in the linear form).
- **v3b(a) passes only if all 15 datasets pass all three checks.** Profile
  likelihoods are reported for the first seed of each player.

**(b) Robustness (reported, does not gate).** The same 15 datasets with v2's
added noise (Gaussian, sd 1.0 ft and 0.25 s, clipped to the feasible set),
refitted and scored on the same checks. This shows how much
misspecification the estimates tolerate.

**Then, only if (a) passes:** fit the three real players and run v3a's
registered criteria (b) and (c) unchanged (`scripts/v3_real.py`). If either
fails, no claim is made about the players' preferences.

### v3b result: 13 of 15 synthetic datasets passed; the gate required all 15

The real players were not fitted, and no claim is made about their
preferences (`results/v3b_synthetic.json`; `scripts/v3_real.py` has still
never been run). Every synthetic player's band shares matched its target
within 0.001.

| Check (15 datasets: 3 players × 5 seeds) | Well-specified (gating) | With v2's noise (not gating) |
|---|---|---|
| θ error ≤ 0.25 | 15 of 15 (largest 0.096) | 15 of 15 (largest 0.171) |
| Identifiability (positive definite, \|corr\| ≤ 0.95, SEs) | 15 of 15 (largest \|corr\| 0.935) | – |
| Band-share gap ≤ 0.05 | 13 of 15 (misses: 0.056 and 0.052) | 8 of 15 (largest 0.068) |
| **All checks** | **13 of 15: fails the gate** | 8 of 15 |

Profile likelihoods (first seed) are bounded for every parameter of every
synthetic player.

What this shows:
- **The model is identifiable and its parameters are recovered.** All 15
  well-specified datasets recovered θ within 10% with no ridge, and v2's
  extra noise only raised the error to at most 17%. The redesign removed
  both of v3a's flaws.
- **The band-share check failed narrowly.** At 450–505 shots per player,
  parameter errors of 6–8% shift the predicted band shares by about five
  percentage points, the size of the 0.05 limit set in v3a. Two of 15
  datasets landed just over it. The limit was kept as pre-registered.
- **Under the rule, v3 stops here.** Whether to read 13 of 15 as close
  enough is a judgment the pre-registration ruled out, so v3b is reported as
  a failed gate.

## Power check of the v3b gate (pre-registered)

v3b stays a failed gate. This step asks only whether the gate could have been
passed by an ideal estimator at these sample sizes. It changes nothing in the
method and was written and committed before running.

**How the v3b band-share gap is defined.** For a synthetic player, the band
shares are the model's predicted probabilities of the four distance bands
(0–4, 4–14, 14–22, 22+ ft), averaged over that player's validation contexts.
The gap is the total variation distance, ½ Σ |share(ŵ) − share(w*)|, between
the shares under the fitted parameters ŵ and under the true parameters w*.
It is measured against the true model's shares, not against sampled shares,
so a sampled-share benchmark does not apply.

**Benchmark (independent of our fitting method).** An ideal (efficient)
estimator has errors distributed as ŵ ~ N(w*, I(w*)⁻¹), where I(w*) is the
Fisher information at the true parameters over the player's training contexts
(485, 450 and 505 shots). For this choice model the information does not
depend on the observed choices: I(w*) = Σᵢ Cov_{P(·|xᵢ,w*)}(features), computed
exactly from w* and the contexts, with no data drawn and no fit run.
- **Primary:** exact Monte Carlo. 2,000 draws of ŵ per synthetic player
  (seed 2026). For each draw, compute the θ error and the band-share gap
  exactly as v3b does, through the model's real (non-linear) band shares.
- **Cross-check (reported, does not decide):** the delta method, which
  linearises the band shares around w*. That gives a Gaussian distribution
  for the share errors, with 100,000 draws.
- **Identifiability:** the identifiability part of the gate is evaluated at
  w* with I(w*) (standard errors, correlations, positive definiteness). It is
  either passed or failed, with no randomness.

**Per-dataset pass probability.** For each synthetic player, p = P(θ error ≤
0.25 and gap ≤ 0.05) × [identifiability at w* passes]. The v3b gate required
all 15 datasets (5 per player) to pass, so the ideal estimator passes the gate
with probability P = Π over players of p⁵. Monte Carlo intervals are reported.

**Decision rule.**
- If P ≥ 0.80, the limit was fair: v3b stands, and the project stops for
  good.
- If P < 0.80, the gate was underpowered, and Step 2 (v3c) is pre-registered
  and run.

The decision uses the primary point estimate.

**If Step 2 runs, its band-share rule is fixed here, before seeing numbers.**
- **Limit:** for each synthetic player, the 99th percentile of the ideal
  estimator's gap from the primary Monte Carlo, rounded up to the next 0.005.
- **Seeds:** at least 20 fresh seeds per player (none of v3b's 0–4).
- **Pass rule:** v3c passes if, for every player, at least 18 of 20 datasets
  pass all checks.
- **Power check before any v3c fit:** the ideal estimator's probability of
  meeting that rule is computed and reported before the fits, and must be at
  least 0.80.

### Power-check result: the v3b gate was underpowered

An ideal estimator (Cramér–Rao errors at the true parameters, no fitting)
passes one v3b dataset with probability 0.92 (LeBron James), 0.94 (Stephen
Curry) and 0.94 (James Harden). Its θ error never exceeded 0.25 in 6,000
draws; every failure came from the band-share limit. Requiring all 15
datasets to pass, the **ideal estimator passes the v3b gate with probability
0.36** (Monte Carlo 95% interval 0.30–0.42; the linearised cross-check gives
0.35). That is below 0.80, so the gate was underpowered and Step 2 runs
(`results/v3b_power.json`).

For scale: the ideal estimator's median gap is about 0.03 and its 95th
percentile just over 0.05, the v3b limit itself. v3b's 13 of 15 is what an
ideal estimator would typically produce. v3b remains a failed gate as
registered.

## v3c: the v3b test with a powered band-share limit (pre-registered; synthetic gate passed, real-data criteria failed)

The band-share limit was revised **after v3b missed narrowly**, because the
power check showed the old limit sat at the ideal estimator's 95th percentile,
making the 15-of-15 gate a 36% shot even for a perfect method. The revision
follows the rule committed in the power-check note before its numbers were
seen. This section was written and committed before any v3c fit.

- **Same method and code path, no tuning:** `nbashots/v3.py`, and the band
  solve and checks from `scripts/v3b_synthetic.py`. The same three synthetic
  players, with the same true parameters, contexts and sample sizes.
- **Unchanged checks:** parameter-error limit 0.25, and the same
  identifiability check.
- **Band-share limit per player**, from the rule (99th percentile of the
  ideal estimator's gap, rounded up to 0.005; `results/v3c_limits.json`):
  LeBron James **0.070**, Stephen Curry **0.065**, James Harden **0.070**.
- **Fresh seeds:** 10–29 for every player, 20 per player, 60 datasets, all
  reported. None of v3b's seeds 0–4 are reused.
- **Pass rule:** v3c passes if, for every player, at least 18 of 20 datasets
  pass all checks. The ideal estimator meets this with probability 0.999.
- **If v3c passes:** fit the three real players and run the real-data criteria
  exactly as pre-registered in v3a: log-likelihood against EP alone and
  against the empirical distribution; decision prediction; the context-shift
  test (`scripts/v3_real.py`, unchanged). If either fails, no claim is made
  about preferences.
- **If v3c fails,** the project ends. There is no further defect in the test to
  point at.

### v3c result: the synthetic gate passed; both real-data criteria failed

**Synthetic gate (`results/v3c_synthetic.json`): passed.** All 60 datasets
passed (20 of 20 per player):
- parameter errors at most 0.115;
- identifiability passed in every dataset;
- band-share gaps at most 0.064.

Eight of the 60 gaps are above v3b's old 0.05 limit, about the rate the power
check predicts for an ideal estimator.

**Real players (`results/v3_real.json`, test set scored once): both
pre-registered criteria failed, so no claim is made about the players'
preferences.**

| Test set | LeBron James | Stephen Curry | James Harden |
|---|---|---|---|
| Log-likelihood per shot: v3 / EP alone / empirical | −9.86 / −9.96 / −9.57 | −9.42 / −9.86 / −9.18 | −9.69 / −9.86 / −9.10 |
| v3 − EP alone (95% interval) | +0.04 to +0.17 (better) | +0.29 to +0.58 (better) | +0.12 to +0.23 (better) |
| v3 − empirical (95% interval) | −0.42 to −0.16 (worse) | −0.42 to −0.07 (worse) | −0.75 to −0.44 (worse) |
| Expected-distance error, ft: v3 / empirical | 8.52 / 8.24 | 8.49 / 8.36 | 8.90 / 8.63 |
| Defender farther than median: observed vs v3 shift in distance | +9.0 ft (6.5 to 11.2) vs −0.6 | +7.9 ft (4.9 to 10.7) vs −1.3 | +9.1 ft (6.6 to 11.4) vs −1.1 |
| More than 10 s on the clock at the catch: observed vs v3 | −7.1 ft (−10.7 to −3.6) vs +0.3 | no shift to explain | −6.4 ft (−9.6 to −3.2) vs 0.0 |

What this means:
- **(b) failed.** v3 beats expected points alone for every player, but loses to
  each player's own shot distribution, on log-likelihood and on distance error.
- **(c) failed.** Where the data shows a shift with context, v3 gets it wrong.
  Players shoot from 8–9 ft farther when the defender is farther away; v3
  predicts slightly shorter shots. This is the limit stated from v2 on,
  showing up in the data: defender distance is partly a result of where the
  shot is taken (a three-pointer has the defender farther away), so treating
  it as given context runs the cause the wrong way.
- **The synthetic test was easier than reality.** The real fits put the
  temperature at 0.6–1.4 points (Curry 0.58, Harden 0.67, LeBron 1.44),
  against 0.15 for the synthetic players. Real choices follow expected points
  far more loosely, and the real fits fail the identifiability check for
  LeBron James and Stephen Curry (it passes for James Harden).

## The attempts, in order

| | Question | Outcome |
|---|---|---|
| **v1** (course, June 2025; audited October 2026) | Can a network tell a player's real shot from generated alternatives? | Yes on held-out shots: real shot ranked first 49–64% vs 26–32% for logistic regression, chance about 29%. Partly learns the generator; the "KKT" terms never touched the cost. |
| **v2** | Can KKT conditions with a cost inside recover a player's cost? | No. Failed pre-registered synthetic recovery: a ridge of costs fits equally well, and stationarity is only local. |
| **v3a** | Can a five-parameter choice model recover it? | Stopped at a flawed test: synthetic players who almost never used one band, plus double-counted noise. |
| **v3b** | The same model, redesigned test | Parameters recovered in 15 of 15 datasets, but 2 missed the band-share limit narrowly; the gate required all 15. |
| **Power check** | Could an ideal estimator have passed v3b's gate? | Only 36% of the time: the limit sat at the ideal estimator's 95th percentile. The gate was underpowered. |
| **v3c** | The same model, band-share limit set by the power check (revised after v3b's narrow miss, by a rule fixed beforehand), fresh seeds | Synthetic gate passed (60 of 60). On real players, both criteria failed: worse than each player's own shot distribution, and wrong about how shots shift with context. |

No claim about any player's preferences or cost is made from v2, v3a, v3b, or v3c.

## Limitations and next steps

- The shot logs carry no stated licence, so the data and any row-level derivative stay out of
  the repository; a reader has to download them from Kaggle to re-run anything.
- The alternatives are synthetic, and the network partly learns the generator (see the caveats
  under "What it finds"). The v1 result is a ranking over generated options, not a recovered
  cost.
- The "KKT" terms of the course model never shaped the cost; v2, v3a, v3b, and v3c each
  stopped at a pre-registered criterion, so no claim about any player's cost is made.
- No further attempt is registered here. The record stops at v3c.

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

## Run it

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
