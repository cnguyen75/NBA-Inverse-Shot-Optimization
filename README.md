# Modeling NBA Player Shot Decisions Using Inverse Optimization and KKT Conditions

**Author:** Colin Nguyen  
**Course:** Math 132B — Professor Atzberger  
**Date:** June 10, 2025  

---

## Overview
This project models how NBA players make **shot selection decisions** using *inverse optimization* and *Karush–Kuhn–Tucker (KKT)* conditions.  
Rather than predicting shot outcomes, this framework aims to **infer latent cost functions** that drive each player’s decision to shoot, given contextual constraints such as defender distance, shot clock, and shot distance.

The goal is to determine whether player behavior aligns with rational cost minimization and to interpret how players trade off between competing factors like pressure, distance, and rhythm.

---

## Motivation
Traditional analytics evaluate what happens **after** a player takes a shot — e.g., field goal percentage or efficiency — but overlook **why** the shot was taken in the first place.

By modeling players’ decisions as **cost-minimizing actions**, this project provides insight into how elite players like **LeBron James**, **Stephen Curry**, and **James Harden** balance spatial awareness, risk, and timing during games.

---

## ⚙️ Problem Formulation
We define each player’s decision process as:

\[
x_i \in \arg \min_x c_\theta(x)
\]

Where:
- \( x_i \) = context vector for the i-th observed shot (e.g., defender distance, shot clock, shot distance, FG%).
- \( c_\theta(x) \) = latent cost function parameterized by neural network weights θ.

The model enforces **KKT optimality**:
- **Primal feasibility:** \( g_j(x^*) \le 0 \)
- **Dual feasibility:** \( \lambda_j \ge 0 \)
- **Complementary slackness:** \( \lambda_j g_j(x^*) = 0 \)
- **Stationarity:** \( \nabla_x L(x^*) = 0 \)

---

## Methods

### Data
- Player shot data from the 2014–2015 NBA season  
- Features include shot distance, defender distance, shot clock, player FG%, and contextual attributes  
- Each observation includes one actual shot (`IS_CHOSEN = 1`) and several alternatives (`IS_CHOSEN = 0`)

### Model
A **neural network cost function** approximates each player’s latent decision-making behavior:
- Input: 12 features  
- 3 hidden layers (Leaky ReLU activations)  
- Batch Normalization after each layer  
- Dropout: 0.3 (first layer), 0.2 (subsequent layers)  
- Optimizer: **AdamW**  
- Early stopping based on validation BCE loss  

The composite loss function:

\[
L_{total} = L_{BCE} + \lambda (V_{stationarity} + V_{comp.slackness} + V_{primal.feasibility})
\]

### Constraints
Soft gameplay constraints:
- \( g_1(x) = -\text{SHOT\_CLOCK} \le 0 \)
- \( g_2(x) = \text{CLOSE\_DEF\_DIST} - 10 \le 0 \)
- \( g_3(x) = \text{SHOT\_DIST} - 35 \le 0 \)

---

## Results

| Player | Validation BCE ↓ | Key Features | Behavioral Insight |
|---------|------------------|---------------|--------------------|
| **LeBron James** | 0.48 | Defender Distance, Shot Clock | Prefers uncontested, efficient midrange shots; values spacing and time |
| **Stephen Curry** | 0.50 | Defender Distance, Shot Distance, FG% | Balances rhythm and range; shoots when in rhythm or open |
| **James Harden** | 0.53 | Defender Distance, Shot Distance, Deep Three | Prioritizes space for step-backs and late-clock isolation shots |

All models converged to low BCE and KKT residuals, indicating rational decision patterns under learned cost functions.

Feature importance (average gradient magnitude) and heatmap visualizations confirmed intuitive tendencies — for instance, defender proximity consistently ranked as the most influential factor.

---

## Interpretation
- **LeBron James:** Rational, efficient shooter prioritizing uncontested midrange attempts.  
- **Stephen Curry:** Adaptive balance of rhythm and distance — rational yet fluid decision-maker.  
- **James Harden:** Space-maximizing, risk-tolerant decision style aligned with iso-heavy play.

---

## Conclusion
This study demonstrates how **inverse optimization with KKT conditions** can uncover the underlying logic of player decision-making.  
By treating shot selection as a cost-minimization process, we learn player-specific cost functions that explain *why* shots are taken, not just *what* shots are made.

The framework bridges **optimization theory** and **deep learning**, opening paths for future applications in:
- Player scouting and behavioral modeling  
- Coaching decision support tools  
- Simulated gameplay and strategy evaluation  

---

## Technologies Used
- **Python**
- **PyTorch**
- **NumPy**, **Pandas**
- **Matplotlib / Seaborn** (visualization)
- **AdamW Optimizer**, **Batch Normalization**, **Dropout**
- **Inverse Optimization**, **KKT Loss Formulation**

---

## Future Work
- Extend analysis to role players and defensive behavior  
- Incorporate spatial-temporal tracking data (SportVU)  
- Explore interpretability via SHAP values or saliency analysis  

---
