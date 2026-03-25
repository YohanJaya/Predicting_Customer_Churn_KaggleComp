Here’s a clean, professional README for your **Optuna Pruning: MedianPruner vs HyperbandPruner** note. I’ve formatted it for GitHub/Notion style so it’s easy to read:

---

# Optuna Pruning: MedianPruner vs HyperbandPruner

This document explains **pruning in Optuna**, its benefits, and when to use `MedianPruner` vs `HyperbandPruner`.

---

## What is Pruning?

**Pruning** = automatically **stop underperforming trials early** during hyperparameter optimization.

Instead of running every trial to the maximum number of iterations (wasting time and risking overfitting), Optuna evaluates **intermediate scores** (e.g., validation AUC after some steps) and terminates trials that are unlikely to succeed.

### Benefits

* Saves significant training time (often **2–5× faster**)
* Reduces overfitting of aggressive/complex hyperparameters
* Focuses compute on promising regions → better generalization
* Particularly effective for boosting models (CatBoost, LightGBM, XGBoost), where AUC quickly improves then plateaus

---

## 1. MedianPruner

**Simple, effective default pruner.**

### How it works

* At each reporting step, it compares the trial’s **best-so-far score** to the **median score** of all completed trials at the same step
* If the trial is worse than the median → it is **pruned (stopped early)**

### Strengths

* Easy to understand
* Stable, robust baseline for most tasks

### When to use

* General-purpose
* Moderate search spaces
* Reliable pruning without being too aggressive

### Key Parameters

```python
n_startup_trials=5      # No pruning until first 5 trials finish
n_warmup_steps=5        # No pruning until a trial reports 5 steps
interval_steps=1        # Check pruning every step (aggressive)
```

---

## 2. HyperbandPruner

**Advanced pruner, often faster and better for boosting models.**

### How it works

* Implements the **Hyperband algorithm** (multi-fidelity bandit approach)
* Runs many **cheap/short trials** first
* Allocates more resources only to **promising trials**
* Aggressively prunes the rest using **Successive Halving**

### Strengths

* Finds better parameters faster
* Effective with **high max iterations (1000+)**
* Outperforms MedianPruner for wide search spaces

### When to use

* Long trainings, wide hyperparameter search space
* Need **maximum efficiency**
* MedianPruner prunes too conservatively

### Key Parameters

```python
min_resource=10       # Minimum iterations for cheap trials (1–50 typical)
max_resource=2500     # Maximum iterations ('auto' detects max reported step)
reduction_factor=3    # Keep top 1/3 of trials at each bracket (higher = more aggressive)
```

---

## Quick Decision Guide

| Situation                                     | Recommended Pruner | Reason                                                      |
| --------------------------------------------- | ------------------ | ----------------------------------------------------------- |
| First experiments, simplicity & stability     | MedianPruner       | Reliable, easy to tune parameters                           |
| High iterations (1000+), want speed & quality | HyperbandPruner    | Efficient resource allocation, often +0.005–0.015 better LB |
| Very few trials (<20–30)                      | MedianPruner       | Hyperband needs more trials to perform well                 |

---

### Tips

* Start with **30 trials using MedianPruner**, then run **30–50 with HyperbandPruner**
* Compare **best AUC** and final leaderboard performance
* Frequent **early pruning messages in logs** → normal & good! It indicates bad hyperparameter regions are avoided

---
