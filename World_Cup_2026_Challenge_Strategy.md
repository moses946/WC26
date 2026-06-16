# World Cup 2026 Goal Prediction Challenge: Deep-Dive Strategy

## Executive Summary

The Fjelstul World Cup Database fundamentally changes the nature of this competition. While the prediction targets are team-level (`total_goals` and `stage_reached`), the source database contains 27 interconnected tables covering tournaments, matches, teams, players, squads, managers, goals, substitutions, referees, and tournament structure.

The likely winning solution will not come from model complexity. It will come from building a rich historical representation of each national team while avoiding leakage and overfitting.

---

# 1. Competition Framing

## Targets

For each qualified 2026 team:

1. Total goals scored during the tournament.
2. Furthest stage reached.

## Constraints

The competition is closed-data.

Allowed:
- Fjelstul World Cup Database
- Features derived from provided tables

Not allowed:
- FIFA rankings
- Elo ratings
- Betting odds
- Scraped football statistics
- External football databases

---

# 2. Dataset Structure Analysis

## Key Observation

Although the database contains millions of values, the prediction unit is:

```text
one row = one team in one World Cup
```

Examples:

```text
Brazil 2018
Argentina 2014
Germany 2006
France 2022
```

The effective training set is only a few hundred team-tournament observations.

Implications:

- Small-data regime
- High overfitting risk
- Feature engineering dominates model selection
- Validation design becomes critical

---

# 3. Available Information Sources

## Core Tournament Tables

- tournaments
- qualified_teams
- team_appearances
- matches
- tournament_standings
- group_standings

## Team-Level Information

- teams
- host_countries
- confederations

## Squad Information

- squads
- players
- player_appearances

## Manager Information

- managers
- manager_appointments
- manager_appearances

## Match Event Information

- goals
- substitutions
- bookings

## Tournament Structure Information

- tournament_stages
- format-related tables

---

# 4. Feature Engineering Roadmap

## Layer 1: Historical Strength Features

Build historical summaries:

- goals per match
- goals conceded per match
- goal difference per match
- win rate
- draw rate
- loss rate
- average stage reached
- best stage reached
- knockout appearance frequency

---

## Layer 2: Recency Features

Recent tournaments should matter more.

Generate:

- last World Cup metrics
- last 2 World Cups
- last 3 World Cups
- exponentially weighted averages

Example:

2022 = 1.0

2018 = 0.7

2014 = 0.5

2010 = 0.3

---

## Layer 3: Team Trajectory Features

Capture improvement or decline.

Examples:

- stage trend slope
- goals trend slope
- goal difference trend slope
- qualification frequency trend

These features transform team history into a time-series representation.

---

## Layer 4: Era-Normalized Features

Historical tournaments differ substantially.

Normalize metrics:

```text
team_goals_per_match / tournament_goals_per_match
```

```text
team_goal_difference / tournament_average_goal_difference
```

Use percentile-based rankings where possible.

---

## Layer 5: Tournament Experience Features

Examples:

- World Cup appearances
- years since first appearance
- years since last appearance
- quarterfinal appearances
- semifinal appearances
- final appearances
- championships

---

## Layer 6: Host Effects

Generate:

- host nation flag
- previous host performance
- host confederation effects
- historical host advantage metrics

Important for:

- United States
- Canada
- Mexico

---

## Layer 7: Squad Stability Features

Potential high-value feature family.

Examples:

- returning players
- average prior World Cups per player
- veteran player counts
- squad continuity

Most competitors will likely ignore these.

---

## Layer 8: Manager Features

Potential hidden signal.

Examples:

- manager continuity
- manager World Cup experience
- previous knockout appearances
- previous championships
- domestic vs foreign manager

---

## Layer 9: Knockout DNA Features

Separate group-stage and knockout performance.

Examples:

- knockout win rate
- knockout goal difference
- penalty shootout success rate
- extra-time performance

---

# 5. Validation Strategy

## Do Not Use Random K-Fold

Random splits leak historical information.

This produces overly optimistic validation scores.

---

## Recommended Validation

Leave-One-World-Cup-Out

Example:

Train:
1930–2018

Validate:
2022

Then:

Train:
1930–2014

Validate:
2018

Repeat across tournaments.

This closely matches the real prediction task.

---

# 6. Stage Prediction Strategy

## Standard Approach

Predict:

```text
group
round_of_32
round_of_16
quarterfinal
semifinal
runner_up
champion
```

as a multiclass target.

### Problem

Classes are ordinal and hierarchical.

---

## Preferred Approach

Predict progression probabilities.

Binary targets:

- reach_round_of_32
- reach_round_of_16
- reach_quarterfinal
- reach_semifinal
- reach_final
- win_final

Advantages:

- easier optimization
- hierarchical structure
- calibrated probabilities
- more robust on small datasets

---

# 7. Goals Prediction Strategy

Instead of directly predicting total goals:

## Step 1

Predict:

```text
goals_per_match
```

## Step 2

Predict tournament progression.

## Step 3

Estimate matches played from predicted stage.

## Step 4

Compute:

```text
predicted_goals =
predicted_goals_per_match *
predicted_matches_played
```

This enforces consistency between both targets.

---

# 8. Modeling Stack

## Baselines

- Ridge Regression
- ElasticNet
- Logistic Regression

---

## Main Models

### Regression

- CatBoost Regressor
- LightGBM Regressor
- Poisson Regression

### Classification

- CatBoost Classifier
- LightGBM Classifier
- Ordinal models

---

## Ensemble Layer

Blend:

- linear models
- boosting models
- count models

Only after validation is stable.

---

# 9. Overfitting Prevention

## Feature Controls

- remove redundant features
- monitor feature importance stability
- eliminate leakage candidates

## Model Controls

- shallow trees
- early stopping
- conservative learning rates
- regularization

## Validation Controls

- tournament-aware CV
- no random splits

---

# 10. Experimental Workflow

## Phase 1

Reverse-engineer:

- Train.csv
- Test.csv

Map every column.

## Phase 2

Build a leakage-safe feature store.

## Phase 3

Train baseline models.

## Phase 4

Add feature groups incrementally.

## Phase 5

Evaluate using tournament-aware CV.

## Phase 6

Build ensemble.

---

# 11. Expected Winning Recipe

Approximate contribution:

- 70% feature engineering
- 20% ensembling
- 10% hyperparameter tuning

Most likely successful ingredients:

1. Historical team representation
2. Recency weighting
3. Era normalization
4. Hierarchical stage prediction
5. CatBoost / LightGBM ensemble
6. Strict leakage prevention

---

# 12. Immediate Next Steps

1. Download Train.csv and Test.csv.
2. Inspect schema and keys.
3. Identify target definitions.
4. Build team-tournament feature store.
5. Create tournament-aware validation.
6. Establish CatBoost baseline.
7. Add advanced feature families.
8. Benchmark every feature group using out-of-fold performance.

The strongest solutions will likely treat each national team as a longitudinal time series evolving across World Cups rather than as independent tournament entries.
