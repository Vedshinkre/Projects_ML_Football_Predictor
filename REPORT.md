# Technical Report: Architecting a Premier League Predictive Engine

---

## 1. Introduction & Philosophy

Predicting the outcome of English Premier League (EPL) matches is notoriously difficult. Football is a low-scoring, high-variance sport where a single anomalous event (a red card, a penalty, a deflection) can completely dictate the outcome.

The goal of this project was to build a robust, pure-mathematics machine learning engine capable of finding the signal through the noise. Rather than throwing hundreds of random statistics into a black-box model, this project utilised a "Feature Staircase" methodology—an ablation-style study where data sets were modularly layered to quantify the exact predictive value of different football concepts:

- Form

- Physics  

- Tactics

- Finances

---

## 2. The "Feature Staircase" Methodology

To ensure the model was not overfitting to irrelevant noise, the configuration (`config.py`) was designed modularly. Separate models were trained on progressively complex feature sets, tracking how each new layer impacted cross-validated accuracy.

---

### Base Tiers: Temporal & Short-Term Form

#### Level 1 (Static)

The absolute baseline. Uses only environmental data:

- `Venue_code`

- `Opp_code`

- `Hour`

- `Day_code`

Predictably, this hovered around ~55% accuracy, barely outperforming a naive "always predict home win" baseline.

---

#### Level 2 (Rolling)

Introduced 3-game rolling averages for offensive and defensive output:

- `GF_rolling`

- `GA_rolling`

- `Sh_rolling`

- `SoT_rolling`

---

#### Level 3 (Physical)

Expanded short-term form to include aggression and set-piece metrics:

- `Corn_rolling`

- `Fouls_rolling`

- `Cards_rolling`

---

#### Level 4 (Pure Matchup)

The final "basic" tier. Maps the Home Team's form directly against the Away Team's recent form to create a localised snapshot of the matchup.

---

### Advanced Tiers: Analytics & Context

#### Level 5 (Professional)

Introduced advanced sabermetrics:

- Dynamic Elo

- Historical Head-to-Head Win Rate (`My_H2H_Win_Rate`)

- Expected Goal Differential Proxy (`xGD_Proxy`)

This pushed the model past the ~65% accuracy threshold.

---

#### Level 6 (Ultimate)

Integrated entity resolution by joining EA Sports FIFA datasets, adding:

- `Ov_Diff`

- `Worth_Diff`

- `Age_Diff`

This emerged as the optimal production model.

---

#### Level 7 (Context)

Added seasonal macro-context like:

- `PPG_Diff` (Points Per Game Differential)

While precise, early-season volatility introduced slight overfitting. This demonstrated that underlying structural metrics (Elo, FIFA ratings) are more stable predictors than current league table standings.

---

## 3. Core Feature Engineering

### A. The Dynamic Elo System

Standard form metrics (like "Last 5 Games") fail to capture long-term pedigree. To address this, a custom Elo rating system was engineered from scratch.

- Every team starts with a base rating of 1500.

- After every match, points are exchanged based on outcome and expected outcome.

The expected win probability is calculated using the logistic curve:

`We = 1 / (1 + 10^((R_opp - R_home)/400))`

Elo ratings are updated using a multiplier:

`K = 20`

This ensures:

- A relegation team beating Manchester City results in a massive rating swing.

- Manchester City beating a relegation team results in minimal change

---

### B. Expected Goals (xGD) Proxy

Because historical datasets often lack official Expected Goals data prior to 2017, the pipeline calculates a mathematical proxy for tactical dominance.

By normalising the differential in Shots on Target (SoT) across recent matches, the model evaluates underlying performance rather than actual goals scored, which are highly susceptible to luck and variance.

---

### C. Log-Scaled Financial Gaps

In the modern Premier League, financial disparity is a strong predictor of success.

By extracting total squad valuations from FIFA datasets, the model calculates the financial gap between two squads.

Because these gaps are exponential (e.g., €1B vs. €50M squads), values are transformed using:

```

np.log1p()

```

This normalises the distribution and prevents the random forest from overweighting extreme outliers.

---

## 4. Preventing Data Leakage (Temporal Integrity)

The most common mistake in sports prediction models is future data leakage—accidentally allowing the model to see the result of the match it is trying to predict.

This pipeline strictly enforces chronological integrity through two mechanisms:

---

### The `shift(1)` Rule

During feature engineering, all dynamic calculations (Elo updates, rolling averages, head-to-head histories) are shifted down by one row for that specific team.

This ensures the data vector fed to the model represents the exact mathematical state of the team on the morning of the match.

---

### TimeSeriesSplit Cross-Validation

Standard K-fold cross-validation randomly shuffles data.

If a model trains on a match from 2022, it cannot accurately test on a match from 2018—it has already "learned" the future.

The hyperparameter tuner uses:

```

TimeSeriesSplit

```

This enforces an expanding window that only validates against chronological future data.

---

## 5. Conclusion & Deployment

By modularising the feature space and isolating the exact predictive value of advanced football metrics, this project successfully built a True Fundamental Model.

The Level 6 (Ultimate) feature set, paired with a heavily regularised random forest:

- `max_depth = 5`

- `min_samples_split = 20`

proved to be the optimal architecture, achieving 66.5% accuracy on unseen future data.

The model is serialised and deployed via the accompanying Streamlit web application, allowing for real-time, in-memory inference on upcoming Premier League fixtures.
