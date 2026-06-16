#!/usr/bin/env python3
"""
World Cup 2026 Prediction Pipeline
===================================
Predicts total_goals and stage_reached for 48 qualified teams.

Uses the Fjelstul World Cup Database (27 tables) with:
- 9 feature engineering layers
- CatBoost / LightGBM / Poisson / Ridge models
- Leave-One-World-Cup-Out validation
- 48-team format post-processing (rank-and-fill)
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("DATA POINTS/data")
TRAIN_PATH = Path("DATA POINTS/Train.csv")
TEST_PATH = Path("DATA POINTS/Test.csv")
SAMPLE_SUB_PATH = Path("DATA POINTS/SampleSubmission.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Stage ordinal encoding — maps Train.csv stage_reached values to ordinal
STAGE_TO_ORDINAL = {
    "group stage": 0,
    "second group stage": 1,   # historical (1974-1982), treat like R16 analogue
    "final round": 1,          # 1950 format
    "round of 16": 2,
    "quarter-finals": 3,
    "semi-finals": 4,
    "third-place match": 4,    # SF losers play 3rd-place match
    "final": 5,                # encompasses both runner-up and champion
}

# Submission stage labels (2026 format)
STAGE_LABELS_2026 = {0: "group", 1: "roundof32", 2: "roundof16", 3: "qf", 4: "sf", 5: "runnerup", 6: "champion"}

# 2026 match-count table
MATCHES_2026 = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 8}

# 2026 stage slot counts for rank-and-fill post-processing
STAGE_SLOTS_2026 = {
    6: 1,   # champion
    5: 1,   # runner-up
    4: 2,   # semifinalists (losers)
    3: 4,   # quarterfinalists (losers)
    2: 8,   # R16 losers
    1: 16,  # R32 losers
    0: 16,  # group stage exits
}

# =============================================================================
# COUNTRY NAME NORMALIZATION
# =============================================================================
# Train.csv / database names -> Test.csv canonical names
TRAIN_TO_TEST_NAME = {
    "Turkey": "Turkiye",
    "Czech Republic": "Czechia",
    "Czechoslovakia": "Czechia",
    "Zaire": "DR Congo",
    "Ivory Coast": "Cote d'Ivoire",
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Serbia and Montenegro": "Serbia",
    "Yugoslavia": "Serbia",
    "Soviet Union": "Russia",
    "Dutch East Indies": "Indonesia",
    "Chinese Taipei": "Chinese Taipei",
    "Korea Republic": "South Korea",
}

# Teams in Test.csv that have NO World Cup history at all (true debut teams)
DEBUT_TEAMS_2026 = {"Cabo Verde", "Curacao", "Jordan", "Uzbekistan"}

# 2026 host countries
HOST_COUNTRIES_2026 = {"United States", "Mexico", "Canada"}
HOST_CONFEDERATION_2026 = "Confederation of North, Central American and Caribbean Association Football"


def normalize_name(name):
    """Normalize historical country name to Test.csv canonical form."""
    return TRAIN_TO_TEST_NAME.get(name, name)


# =============================================================================
# PHASE 1: DATA LOADING
# =============================================================================

def load_data():
    """Load all data sources."""
    print("=" * 70)
    print("PHASE 1: DATA LOADING & EXPLORATION")
    print("=" * 70)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print(f"Train: {train.shape[0]} rows × {train.shape[1]} cols")
    print(f"Test:  {test.shape[0]} teams to predict")

    # Load Fjelstul database
    db = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        db[f.stem] = pd.read_csv(f)
    print(f"Database: {len(db)} tables loaded")

    # --- Normalize country names in Train ---
    train["canonical_country"] = train["country"].apply(normalize_name)

    # --- Encode stage ordinally ---
    train["stage_ordinal"] = train["stage_reached"].map(STAGE_TO_ORDINAL)

    # For "final" entries, determine champion vs runner-up from tournament winners
    tournaments = db["tournaments"]
    tournament_winners = {}
    for _, t in tournaments.iterrows():
        tid = t["tournament_id"]
        winner = normalize_name(t.get("winner", ""))
        tournament_winners[tid] = winner

    # Teams that reached "final" AND won = champion (6), else runner-up (5)
    for idx, row in train[train["stage_reached"] == "final"].iterrows():
        winner = tournament_winners.get(row["tournament_id"], "")
        if normalize_name(row["country"]) == winner or row["country"] == winner:
            train.loc[idx, "stage_ordinal"] = 6  # champion
        else:
            train.loc[idx, "stage_ordinal"] = 5  # runner-up

    # --- Profile targets ---
    print(f"\n--- Target: total_goals ---")
    print(f"  Mean: {train['total_goals'].mean():.1f}, Median: {train['total_goals'].median():.0f}, "
          f"Std: {train['total_goals'].std():.1f}, Range: [{train['total_goals'].min()}, {train['total_goals'].max()}]")

    print(f"\n--- Target: stage_ordinal ---")
    stage_dist = train["stage_ordinal"].value_counts().sort_index()
    for val, count in stage_dist.items():
        label = {0: "group", 1: "2nd group/R16-era", 2: "R16", 3: "QF", 4: "SF/3rd", 5: "runner-up", 6: "champion"}
        print(f"  {label.get(val, val)}: {count}")

    # --- Identify debut teams ---
    train_countries = set(train["canonical_country"].unique())
    test_countries = set(test["country"].values)
    debut_in_test = test_countries - train_countries
    print(f"\n2026 debut teams (no WC history): {debut_in_test}")

    return train, test, sample_sub, db, tournament_winners


# =============================================================================
# PHASE 2: FEATURE ENGINEERING
# =============================================================================

def build_features_for_rows(rows_df, full_history_df, db, current_year_col="year"):
    """
    Build feature vectors for each row in rows_df.
    For each row, uses only data from tournaments STRICTLY BEFORE that row's year.
    """
    features_list = []

    # Pre-compute some DB lookups
    host_teams = set()
    hosts = db["host_countries"]
    for _, h in hosts.iterrows():
        tid = h["tournament_id"]
        hname = normalize_name(h["team_name"])
        try:
            yr = int(tid.split("-")[1]) if "-" in str(tid) else None
        except (ValueError, IndexError):
            yr = None
        if yr:
            host_teams.add((yr, hname))

    # Host confederation per year
    host_confed_per_year = {}
    for _, h in hosts.iterrows():
        tid = h["tournament_id"]
        try:
            yr = int(tid.split("-")[1])
        except (ValueError, IndexError):
            continue
        hname = normalize_name(h["team_name"])
        # Find confederation from full history
        team_confed = full_history_df[full_history_df["canonical_country"] == hname]
        if len(team_confed) > 0:
            confed = team_confed.iloc[-1]["confederation_name"]
            host_confed_per_year.setdefault(yr, set()).add(confed)

    # Squad data for Layer 7
    squads = db.get("squads", pd.DataFrame())
    if not squads.empty:
        squads["canonical_country"] = squads["team_name"].apply(normalize_name)
        squads["year"] = squads["tournament_id"].str.extract(r"(\d{4})").astype(float)
        # Filter to men's tournaments only
        squads = squads[squads["tournament_id"].str.contains("Men's|^WC-\\d{4}$", regex=True, na=False) |
                        ~squads["tournament_name"].str.contains("Women's", na=False)]

    # Manager data for Layer 8
    mgr_appts = db.get("manager_appointments", pd.DataFrame())
    if not mgr_appts.empty:
        mgr_appts["canonical_country"] = mgr_appts["team_name"].apply(normalize_name)
        mgr_appts["year"] = mgr_appts["tournament_id"].str.extract(r"(\d{4})").astype(float)
        mgr_appts = mgr_appts[~mgr_appts["tournament_name"].str.contains("Women's", na=False)]

    # Match data for Layer 9
    matches = db.get("matches", pd.DataFrame())
    if not matches.empty:
        matches["home_canonical"] = matches["home_team_name"].apply(normalize_name)
        matches["away_canonical"] = matches["away_team_name"].apply(normalize_name)
        matches["year"] = matches["tournament_id"].str.extract(r"(\d{4})").astype(float)
        matches = matches[~matches["tournament_name"].str.contains("Women's", na=False)]

    # Penalty data
    penalties = db.get("penalty_kicks", pd.DataFrame())
    if not penalties.empty:
        penalties["canonical_country"] = penalties["team_name"].apply(normalize_name)
        penalties["year"] = penalties["tournament_id"].str.extract(r"(\d{4})").astype(float)
        penalties = penalties[~penalties["tournament_name"].str.contains("Women's", na=False)]

    # Group standings for goal detail
    group_standings = db.get("group_standings", pd.DataFrame())
    if not group_standings.empty:
        group_standings["canonical_country"] = group_standings["team_name"].apply(normalize_name)
        group_standings["year"] = group_standings["tournament_id"].str.extract(r"(\d{4})").astype(float)
        group_standings = group_standings[~group_standings["tournament_name"].str.contains("Women's", na=False)]

    for idx, row in rows_df.iterrows():
        country = row["canonical_country"]
        year = row[current_year_col]
        confed = row.get("confederation_name", "Unknown")

        # Historical data for this team before current year
        history = full_history_df[
            (full_history_df["canonical_country"] == country) &
            (full_history_df["year"] < year)
        ].sort_values("year")

        f = {}
        f["tournament_year"] = year

        # ===================== LAYER 1: Historical Strength =====================
        n = len(history)
        f["hist_appearances"] = n

        if n > 0:
            f["hist_avg_goals"] = history["total_goals"].mean()
            f["hist_total_goals"] = history["total_goals"].sum()
            gpm = history["total_goals"] / history["matches_played"]
            f["hist_avg_gpm"] = gpm.mean()
            f["hist_avg_matches"] = history["matches_played"].mean()
            f["hist_avg_stage"] = history["stage_ordinal"].mean()
            f["hist_max_stage"] = history["stage_ordinal"].max()
            f["hist_min_stage"] = history["stage_ordinal"].min()
            f["hist_std_stage"] = history["stage_ordinal"].std() if n > 1 else 0
            f["hist_knockout_rate"] = (history["stage_ordinal"] >= 2).mean()
            f["hist_qf_rate"] = (history["stage_ordinal"] >= 3).mean()
            f["hist_sf_rate"] = (history["stage_ordinal"] >= 4).mean()
            f["hist_final_rate"] = (history["stage_ordinal"] >= 5).mean()
            f["hist_champion_rate"] = (history["stage_ordinal"] >= 6).mean()
        else:
            for col in ["hist_avg_goals", "hist_total_goals", "hist_avg_gpm",
                         "hist_avg_matches", "hist_avg_stage", "hist_max_stage",
                         "hist_min_stage", "hist_std_stage", "hist_knockout_rate",
                         "hist_qf_rate", "hist_sf_rate", "hist_final_rate",
                         "hist_champion_rate"]:
                f[col] = np.nan

        # ===================== LAYER 2: Recency-Weighted =====================
        if n > 0:
            years_ago = year - history["year"]
            weights = (0.3 ** (years_ago / 4))  # Aggressive recency decay
            weights_norm = weights / weights.sum()
            gpm = history["total_goals"] / history["matches_played"]

            f["recent_wt_goals"] = (history["total_goals"] * weights_norm).sum()
            f["recent_wt_gpm"] = (gpm * weights_norm).sum()
            f["recent_wt_stage"] = (history["stage_ordinal"] * weights_norm).sum()

            for k in [1, 2, 3]:
                last_k = history.tail(k)
                f[f"last{k}_avg_goals"] = last_k["total_goals"].mean()
                f[f"last{k}_avg_gpm"] = (last_k["total_goals"] / last_k["matches_played"]).mean()
                f[f"last{k}_avg_stage"] = last_k["stage_ordinal"].mean()
                f[f"last{k}_max_stage"] = last_k["stage_ordinal"].max()
        else:
            for col in ["recent_wt_goals", "recent_wt_gpm", "recent_wt_stage"]:
                f[col] = np.nan
            for k in [1, 2, 3]:
                for s in ["avg_goals", "avg_gpm", "avg_stage", "max_stage"]:
                    f[f"last{k}_{s}"] = np.nan

        # ===================== LAYER 3: Trajectory =====================
        if n >= 3:
            x = np.arange(n)
            gpm_vals = (history["total_goals"] / history["matches_played"]).values
            stage_vals = history["stage_ordinal"].values
            f["trend_stage_slope"] = np.polyfit(x, stage_vals, 1)[0]
            f["trend_gpm_slope"] = np.polyfit(x, gpm_vals, 1)[0]
            mid = n // 2
            f["trend_stage_recent_vs_early"] = stage_vals[mid:].mean() - stage_vals[:mid].mean()
        else:
            f["trend_stage_slope"] = 0.0
            f["trend_gpm_slope"] = 0.0
            f["trend_stage_recent_vs_early"] = 0.0

        # ===================== LAYER 4: Era-Normalized =====================
        # Normalize by the PREVIOUS tournament's averages
        prev_tournaments = full_history_df[full_history_df["year"] < year]
        if len(prev_tournaments) > 0:
            last_tournament_year = prev_tournaments["year"].max()
            last_tourney = full_history_df[full_history_df["year"] == last_tournament_year]
            avg_goals_tourney = last_tourney["total_goals"].mean()
            avg_gpm_tourney = (last_tourney["total_goals"] / last_tourney["matches_played"]).mean()
            avg_stage_tourney = last_tourney["stage_ordinal"].mean()

            if n > 0:
                last_team = history.iloc[-1]
                team_gpm = last_team["total_goals"] / last_team["matches_played"]
                f["era_gpm_ratio"] = team_gpm / avg_gpm_tourney if avg_gpm_tourney > 0 else 1.0
                f["era_goals_ratio"] = last_team["total_goals"] / avg_goals_tourney if avg_goals_tourney > 0 else 1.0
                f["era_stage_diff"] = last_team["stage_ordinal"] - avg_stage_tourney
            else:
                f["era_gpm_ratio"] = 1.0
                f["era_goals_ratio"] = 1.0
                f["era_stage_diff"] = 0.0
        else:
            f["era_gpm_ratio"] = 1.0
            f["era_goals_ratio"] = 1.0
            f["era_stage_diff"] = 0.0

        # ===================== LAYER 5: Tournament Experience =====================
        f["total_wc_appearances"] = n
        f["is_debut"] = 1 if n == 0 else 0
        f["modern_era_debut"] = 1 if n == 0 and year >= 2010 else 0
        if n > 0:
            f["years_since_last_wc"] = year - history["year"].max()
            f["years_since_first_wc"] = year - history["year"].min()
            span = (year - history["year"].min()) / 4 + 1
            f["wc_frequency"] = n / span
            f["qf_appearances"] = (history["stage_ordinal"] >= 3).sum()
            f["sf_appearances"] = (history["stage_ordinal"] >= 4).sum()
            f["final_appearances"] = (history["stage_ordinal"] >= 5).sum()
            f["titles"] = (history["stage_ordinal"] >= 6).sum()
        else:
            f["years_since_last_wc"] = 99
            f["years_since_first_wc"] = 0
            f["wc_frequency"] = 0.0
            f["qf_appearances"] = 0
            f["sf_appearances"] = 0
            f["final_appearances"] = 0
            f["titles"] = 0

        # ===================== LAYER 10: Regression-to-Mean =====================
        # Champions and finalists historically regress. Only 2 teams ever
        # won consecutive WCs (Italy 34-38, Brazil 58-62). Signal this.
        if n > 0:
            last_stage = history.iloc[-1]["stage_ordinal"]
            f["was_champion_last_wc"] = 1 if last_stage == 6 else 0
            f["was_finalist_last_wc"] = 1 if last_stage >= 5 else 0
            f["was_semifinalist_last_wc"] = 1 if last_stage >= 4 else 0

            # Stage drop: how much teams typically regress after their peak
            if n >= 2:
                f["last_stage_drop"] = history.iloc[-1]["stage_ordinal"] - history.iloc[-2]["stage_ordinal"]
            else:
                f["last_stage_drop"] = 0

            # Historical regression signal: avg stage change after reaching final
            # (do previous finalists tend to do worse next time?)
            finalist_years = history[history["stage_ordinal"] >= 5]["year"].values
            post_final_drops = []
            for fy in finalist_years:
                next_app = history[history["year"] > fy]
                if len(next_app) > 0:
                    drop = history[history["year"] == fy]["stage_ordinal"].values[0] - next_app.iloc[0]["stage_ordinal"]
                    post_final_drops.append(drop)
            f["avg_post_final_regression"] = np.mean(post_final_drops) if post_final_drops else 0

            # Peak vs last: how far from best-ever performance
            f["peak_vs_last"] = history["stage_ordinal"].max() - last_stage
        else:
            f["was_champion_last_wc"] = 0
            f["was_finalist_last_wc"] = 0
            f["was_semifinalist_last_wc"] = 0
            f["last_stage_drop"] = 0
            f["avg_post_final_regression"] = 0
            f["peak_vs_last"] = 0

        # ===================== LAYER 6: Host Effect =====================
        f["is_host"] = 1 if (year, country) in host_teams else 0
        year_host_confeds = host_confed_per_year.get(year, set())
        f["same_confed_as_host"] = 1 if confed in year_host_confeds else 0

        # ===================== LAYER 7: Squad Depth =====================
        if not squads.empty and "player_id" in squads.columns:
            current_squad = squads[
                (squads["canonical_country"] == country) & (squads["year"] == year)
            ]
            prev_squad = squads[
                (squads["canonical_country"] == country) & (squads["year"] < year)
            ]
            cur_players = set(current_squad["player_id"].values)
            prev_players = set(prev_squad["player_id"].values) if len(prev_squad) > 0 else set()

            f["squad_size"] = len(cur_players)
            f["returning_players"] = len(cur_players & prev_players)
            f["returning_ratio"] = f["returning_players"] / f["squad_size"] if f["squad_size"] > 0 else 0

            if len(cur_players) > 0 and len(prev_squad) > 0:
                exp = prev_squad[prev_squad["player_id"].isin(cur_players)].groupby("player_id")["year"].nunique()
                f["avg_player_wc_exp"] = exp.mean() if len(exp) > 0 else 0
                f["max_player_wc_exp"] = exp.max() if len(exp) > 0 else 0
                f["veteran_count"] = int((exp >= 2).sum()) if len(exp) > 0 else 0
            else:
                f["avg_player_wc_exp"] = 0
                f["max_player_wc_exp"] = 0
                f["veteran_count"] = 0
        else:
            for col in ["squad_size", "returning_players", "returning_ratio",
                         "avg_player_wc_exp", "max_player_wc_exp", "veteran_count"]:
                f[col] = np.nan

        # ===================== LAYER 8: Manager =====================
        if not mgr_appts.empty and "manager_id" in mgr_appts.columns:
            cur_mgr = mgr_appts[
                (mgr_appts["canonical_country"] == country) & (mgr_appts["year"] == year)
            ]
            if len(cur_mgr) > 0:
                mgr_id = cur_mgr.iloc[0]["manager_id"]
                mgr_hist = mgr_appts[
                    (mgr_appts["manager_id"] == mgr_id) & (mgr_appts["year"] < year)
                ]
                f["manager_wc_exp"] = len(mgr_hist)
                f["manager_same_team"] = len(
                    mgr_hist[mgr_hist["canonical_country"] == country]
                )
                # Foreign manager?
                mgr_country = cur_mgr.iloc[0].get("country_name", "")
                f["manager_foreign"] = 0 if normalize_name(mgr_country) == country else 1
            else:
                f["manager_wc_exp"] = 0
                f["manager_same_team"] = 0
                f["manager_foreign"] = np.nan
        else:
            f["manager_wc_exp"] = np.nan
            f["manager_same_team"] = np.nan
            f["manager_foreign"] = np.nan

        # ===================== LAYER 9: Knockout DNA =====================
        if not matches.empty and "stage_name" in matches.columns:
            ko_home = matches[
                (matches["year"] < year) &
                (~matches["stage_name"].str.contains("group", case=False, na=False)) &
                (matches["home_canonical"] == country)
            ]
            ko_away = matches[
                (matches["year"] < year) &
                (~matches["stage_name"].str.contains("group", case=False, na=False)) &
                (matches["away_canonical"] == country)
            ]

            ko_gf = ko_home["home_team_score"].sum() + ko_away["away_team_score"].sum()
            ko_ga = ko_home["away_team_score"].sum() + ko_away["home_team_score"].sum()
            ko_n = len(ko_home) + len(ko_away)

            ko_wins = ((ko_home["home_team_win"] == True) | (ko_home["home_team_win"] == 1)).sum()
            ko_wins += ((ko_away["away_team_win"] == True) | (ko_away["away_team_win"] == 1)).sum()

            f["ko_matches"] = ko_n
            f["ko_win_rate"] = ko_wins / ko_n if ko_n > 0 else np.nan
            f["ko_gd_per_match"] = (ko_gf - ko_ga) / ko_n if ko_n > 0 else np.nan
        else:
            f["ko_matches"] = 0
            f["ko_win_rate"] = np.nan
            f["ko_gd_per_match"] = np.nan

        # Penalty history
        if not penalties.empty:
            team_pens = penalties[
                (penalties["canonical_country"] == country) & (penalties["year"] < year)
            ]
            f["penalty_kicks_total"] = len(team_pens)
            f["penalty_conversion_rate"] = team_pens["converted"].mean() if len(team_pens) > 0 else np.nan
        else:
            f["penalty_kicks_total"] = 0
            f["penalty_conversion_rate"] = np.nan

        # ===================== LAYER 4b: Group Stage Detail =====================
        if not group_standings.empty:
            team_gs = group_standings[
                (group_standings["canonical_country"] == country) &
                (group_standings["year"] < year)
            ]
            if len(team_gs) > 0:
                f["hist_gs_avg_points"] = team_gs["points"].mean()
                f["hist_gs_avg_gf"] = team_gs["goals_for"].mean()
                f["hist_gs_avg_ga"] = team_gs["goals_against"].mean()
                f["hist_gs_avg_gd"] = team_gs["goal_difference"].mean()
                f["hist_gs_win_rate"] = (team_gs["wins"] / team_gs["played"]).mean()
                f["hist_gs_advance_rate"] = team_gs["advanced"].mean() if "advanced" in team_gs.columns else np.nan
            else:
                for col in ["hist_gs_avg_points", "hist_gs_avg_gf", "hist_gs_avg_ga",
                             "hist_gs_avg_gd", "hist_gs_win_rate", "hist_gs_advance_rate"]:
                    f[col] = np.nan
        else:
            for col in ["hist_gs_avg_points", "hist_gs_avg_gf", "hist_gs_avg_ga",
                         "hist_gs_avg_gd", "hist_gs_win_rate", "hist_gs_advance_rate"]:
                f[col] = np.nan

        # ===================== LAYER 11: Confederation Strength =====================
        # How strong is this team's confederation historically?
        # Helps differentiate borderline R32 vs group-exit teams
        confed_history = full_history_df[
            (full_history_df["year"] < year) &
            (full_history_df["confederation_name"] == confed)
        ]
        if len(confed_history) > 0:
            f["confed_avg_stage"] = confed_history["stage_ordinal"].mean()
            f["confed_avg_gpm"] = (confed_history["total_goals"] / confed_history["matches_played"]).mean()
            f["confed_knockout_rate"] = (confed_history["stage_ordinal"] >= 2).mean()
            # Recent confederation strength (last 3 tournaments)
            recent_years = sorted(confed_history["year"].unique())[-3:]
            recent_confed = confed_history[confed_history["year"].isin(recent_years)]
            f["confed_recent_avg_stage"] = recent_confed["stage_ordinal"].mean()
            f["confed_recent_knockout_rate"] = (recent_confed["stage_ordinal"] >= 2).mean()
        else:
            f["confed_avg_stage"] = np.nan
            f["confed_avg_gpm"] = np.nan
            f["confed_knockout_rate"] = np.nan
            f["confed_recent_avg_stage"] = np.nan
            f["confed_recent_knockout_rate"] = np.nan

        # ===================== Tournament team count =====================
        tourney_info = db["tournaments"]
        t_info = tourney_info[tourney_info["year"] == year]
        f["tournament_team_count"] = t_info["count_teams"].values[0] if len(t_info) > 0 else 32

        # ===================== Confederation encoding =====================
        f["confederation_name"] = confed
        f["region_name"] = row.get("region_name", "Unknown")

        features_list.append(f)

    return pd.DataFrame(features_list, index=rows_df.index)


def build_2026_test_rows(test, train, db):
    """
    Build pseudo-rows for the 48 test teams that can be fed into
    build_features_for_rows with year=2026.
    """
    print("\n  Building 2026 test rows...")

    test_rows = []
    debut_baselines = compute_debut_baselines(train)

    for _, row in test.iterrows():
        country = row["country"]

        # Try to find this team in training data
        team_hist = train[train["canonical_country"] == country]

        if len(team_hist) > 0:
            # Use last known confederation info
            last = team_hist.iloc[-1]
            test_rows.append({
                "canonical_country": country,
                "year": 2026,
                "confederation_name": last["confederation_name"],
                "region_name": last["region_name"],
                "team_id": last["team_id"],
                "team_code": last["team_code"],
                "ID": row["ID"],
            })
        else:
            # Debut team — try to find confederation from teams database
            teams_db = db["teams"]
            teams_db_norm = teams_db.copy()
            teams_db_norm["canonical"] = teams_db_norm["team_name"].apply(normalize_name)
            match = teams_db_norm[teams_db_norm["canonical"] == country]

            if len(match) > 0:
                confed = match.iloc[0].get("confederation_name", "Unknown")
                region = match.iloc[0].get("region_name", "Unknown")
                team_id = match.iloc[0].get("team_id", "Unknown")
                team_code = match.iloc[0].get("team_code", "Unknown")
            else:
                # Hard-coded for known debut teams
                debut_confed = {
                    "Cabo Verde": ("Confederation of African Football", "Africa"),
                    "Curacao": ("Confederation of North, Central American and Caribbean Association Football", "Caribbean"),
                    "Jordan": ("Asian Football Confederation", "Middle East"),
                    "Uzbekistan": ("Asian Football Confederation", "Central Asia"),
                }
                confed, region = debut_confed.get(country, ("Unknown", "Unknown"))
                team_id = "Unknown"
                team_code = row["ID"].split("_")[1]

            test_rows.append({
                "canonical_country": country,
                "year": 2026,
                "confederation_name": confed,
                "region_name": region,
                "team_id": team_id,
                "team_code": team_code,
                "ID": row["ID"],
            })

    test_df = pd.DataFrame(test_rows)
    print(f"  Built {len(test_df)} test rows")
    debut_count = test_df["canonical_country"].isin(DEBUT_TEAMS_2026).sum()
    print(f"  Debut teams: {debut_count}")

    return test_df, debut_baselines


def compute_debut_baselines(train):
    """Analyze historical debut teams to establish baselines."""
    print("\n  Analyzing historical debut team performances...")

    first_app = train.groupby("canonical_country")["year"].min().reset_index()
    first_app.columns = ["canonical_country", "debut_year"]

    debut_perf = train.merge(first_app, on="canonical_country")
    debut_perf = debut_perf[debut_perf["year"] == debut_perf["debut_year"]]

    print(f"  Found {len(debut_perf)} historical debut performances")
    print(f"  Debut avg goals: {debut_perf['total_goals'].mean():.1f}")
    print(f"  Debut avg gpm:   {(debut_perf['total_goals']/debut_perf['matches_played']).mean():.2f}")
    print(f"  Debut avg stage: {debut_perf['stage_ordinal'].mean():.1f}")
    print(f"  Debut stage distribution:")
    for stage, count in debut_perf["stage_reached"].value_counts().items():
        print(f"    {stage}: {count}")

    return {
        "avg_goals": debut_perf["total_goals"].mean(),
        "avg_gpm": (debut_perf["total_goals"] / debut_perf["matches_played"]).mean(),
        "avg_stage": debut_perf["stage_ordinal"].mean(),
    }


# =============================================================================
# PHASE 4: MODELING & VALIDATION
# =============================================================================

def run_pipeline(train, test, db, tournament_winners):
    """Full modeling pipeline: feature engineering, validation, prediction."""
    from sklearn.linear_model import Ridge, LogisticRegression, PoissonRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
    from sklearn.impute import SimpleImputer
    from catboost import CatBoostRegressor, CatBoostClassifier
    from lightgbm import LGBMRegressor, LGBMClassifier
    import lightgbm as lgb

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)

    feature_df = build_features_for_rows(train, train, db)
    print(f"\n  Feature matrix: {feature_df.shape[1]} columns × {feature_df.shape[0]} rows")

    # Separate categorical and numeric columns
    cat_cols = ["confederation_name", "region_name"]
    feature_cols = [c for c in feature_df.columns if c not in cat_cols]

    # Encode categoricals
    le_confed = LabelEncoder()
    le_region = LabelEncoder()

    all_confeds = list(feature_df["confederation_name"].unique()) + ["Unknown"]
    all_regions = list(feature_df["region_name"].unique()) + ["Unknown", "Caribbean", "Central Asia"]
    le_confed.fit(all_confeds)
    le_region.fit(all_regions)

    feature_df["confed_enc"] = le_confed.transform(feature_df["confederation_name"])
    feature_df["region_enc"] = le_region.transform(feature_df["region_name"])

    feature_cols = [c for c in feature_df.columns if c not in cat_cols]
    print(f"  Final feature count: {len(feature_cols)}")

    # Prepare arrays
    X = feature_df[feature_cols].values.astype(float)
    y_goals = train["total_goals"].values.astype(float)
    y_stage = train["stage_ordinal"].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    if HAS_WANDB:
        wandb.init(
            project="world-cup-2026",
            name="prediction-pipeline",
            config={
                "features_count": len(feature_cols),
                "dataset_rows": len(train),
                "goals_models": ["ridge", "poisson", "catboost", "lgbm"],
                "stage_models": ["catboost", "lgbm", "logistic_regression"],
                "catboost_params": {
                    "iterations": 500,
                    "depth": 4,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 5
                },
                "lgbm_params": {
                    "n_estimators": 500,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "reg_lambda": 5
                }
            }
        )

    # =========================================================================
    # VALIDATION (Leave-One-World-Cup-Out)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: VALIDATION (Leave-One-World-Cup-Out)")
    print("=" * 70)

    years = sorted(train["year"].unique())
    val_years = [y for y in years if y >= 2006]  # validate on modern tournaments

    cv_goals = []
    cv_stage = []

    for val_year in val_years:
        mask_tr = train["year"] != val_year
        mask_val = train["year"] == val_year

        X_tr, X_val = X[mask_tr], X[mask_val]
        y_g_tr, y_g_val = y_goals[mask_tr], y_goals[mask_val]
        y_s_tr, y_s_val = y_stage[mask_tr], y_stage[mask_val]

        # --- Goals models ---
        g_preds = {}

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_g_tr)
        g_preds["ridge"] = ridge.predict(X_val)

        try:
            poisson = PoissonRegressor(alpha=1.0, max_iter=1000)
            poisson.fit(X_tr, y_g_tr)
            g_preds["poisson"] = poisson.predict(X_val)
        except Exception:
            pass

        cb_r = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05,
                                  l2_leaf_reg=5, verbose=0, random_seed=42,
                                  early_stopping_rounds=50)
        cb_r.fit(X_tr, y_g_tr, eval_set=(X_val, y_g_val), verbose=0)
        g_preds["catboost"] = cb_r.predict(X_val)

        lgbm_r = LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                reg_lambda=5, verbose=-1, random_state=42, n_jobs=1)
        lgbm_r.fit(X_tr, y_g_tr, eval_set=[(X_val, y_g_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        g_preds["lgbm"] = lgbm_r.predict(X_val)

        ens_goals = np.mean(list(g_preds.values()), axis=0)
        ens_goals = np.maximum(ens_goals, 0).round()
        rmse = np.sqrt(mean_squared_error(y_g_val, ens_goals))
        mae = mean_absolute_error(y_g_val, ens_goals)
        cv_goals.append({"year": val_year, "rmse": rmse, "mae": mae})

        # --- Stage models ---
        s_probas = []

        cb_c = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
                                   l2_leaf_reg=5, verbose=0, random_seed=42,
                                   early_stopping_rounds=50, loss_function="MultiClass",
                                   auto_class_weights="Balanced")
        cb_c.fit(X_tr, y_s_tr, eval_set=(X_val, y_s_val), verbose=0)
        s_probas.append(cb_c.predict_proba(X_val))

        lgbm_c = LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 reg_lambda=5, verbose=-1, random_state=42, n_jobs=1,
                                 class_weight="balanced")
        lgbm_c.fit(X_tr, y_s_tr, eval_set=[(X_val, y_s_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        s_probas.append(lgbm_c.predict_proba(X_val))

        try:
            lr = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                     class_weight="balanced")
            lr.fit(X_tr, y_s_tr)
            s_probas.append(lr.predict_proba(X_val))
        except Exception:
            pass

        # Average probabilities
        n_classes = max(p.shape[1] for p in s_probas)
        aligned = []
        for p in s_probas:
            if p.shape[1] < n_classes:
                padded = np.zeros((p.shape[0], n_classes))
                padded[:, :p.shape[1]] = p
                aligned.append(padded)
            else:
                aligned.append(p)
        avg_p = np.mean(aligned, axis=0)
        ens_stage = np.argmax(avg_p, axis=1)

        acc = accuracy_score(y_s_val, ens_stage)
        mae_s = mean_absolute_error(y_s_val, ens_stage)
        cv_stage.append({"year": val_year, "accuracy": acc, "mae": mae_s})

        print(f"  {val_year}: Goals RMSE={rmse:.2f} MAE={mae:.2f} | Stage Acc={acc:.2f} MAE={mae_s:.2f}")

        if HAS_WANDB:
            wandb.log({
                "val_year": val_year,
                "fold_goals_rmse": rmse,
                "fold_goals_mae": mae,
                "fold_stage_acc": acc,
                "fold_stage_mae": mae_s,
            })

    cv_g = pd.DataFrame(cv_goals)
    cv_s = pd.DataFrame(cv_stage)
    mean_rmse = cv_g['rmse'].mean()
    std_rmse = cv_g['rmse'].std()
    mean_mae = cv_g['mae'].mean()
    mean_acc = cv_s['accuracy'].mean()
    mean_mae_s = cv_s['mae'].mean()

    print(f"\n  --- CV Summary (avg over {len(val_years)} tournaments) ---")
    print(f"  Goals RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
    print(f"  Goals MAE:  {mean_mae:.2f}")
    print(f"  Stage Acc:  {mean_acc:.2f}")
    print(f"  Stage MAE:  {mean_mae_s:.2f}")

    if HAS_WANDB:
        wandb.log({
            "cv_goals_rmse_mean": mean_rmse,
            "cv_goals_rmse_std": std_rmse,
            "cv_goals_mae_mean": mean_mae,
            "cv_stage_acc_mean": mean_acc,
            "cv_stage_mae_mean": mean_mae_s,
        })

    # =========================================================================
    # FINAL MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: FINAL PREDICTION")
    print("=" * 70)

    # Build test features
    test_rows_df, debut_baselines = build_2026_test_rows(test, train, db)
    test_feature_df = build_features_for_rows(test_rows_df, train, db)

    # Encode categoricals for test
    test_feature_df["confed_enc"] = le_confed.transform(
        test_feature_df["confederation_name"].apply(
            lambda x: x if x in le_confed.classes_ else "Unknown"
        )
    )
    test_feature_df["region_enc"] = le_region.transform(
        test_feature_df["region_name"].apply(
            lambda x: x if x in le_region.classes_ else "Unknown"
        )
    )

    X_test = test_feature_df[feature_cols].values.astype(float)
    X_test = imputer.transform(X_test)

    # Train final models on ALL data
    final_g_preds = {}
    final_s_probas = []

    # Ridge
    ridge_f = Ridge(alpha=10.0)
    ridge_f.fit(X, y_goals)
    final_g_preds["ridge"] = ridge_f.predict(X_test)

    # Poisson
    try:
        poisson_f = PoissonRegressor(alpha=1.0, max_iter=1000)
        poisson_f.fit(X, y_goals)
        final_g_preds["poisson"] = poisson_f.predict(X_test)
    except Exception:
        pass

    # CatBoost Regressor
    cb_r_f = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                l2_leaf_reg=5, verbose=0, random_seed=42)
    cb_r_f.fit(X, y_goals)
    final_g_preds["catboost"] = cb_r_f.predict(X_test)

    # LightGBM Regressor
    lgbm_r_f = LGBMRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                              reg_lambda=5, verbose=-1, random_state=42, n_jobs=1)
    lgbm_r_f.fit(X, y_goals)
    final_g_preds["lgbm"] = lgbm_r_f.predict(X_test)

    # CatBoost Classifier
    cb_c_f = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                                 l2_leaf_reg=5, verbose=0, random_seed=42,
                                 loss_function="MultiClass", auto_class_weights="Balanced")
    cb_c_f.fit(X, y_stage)
    final_s_probas.append(cb_c_f.predict_proba(X_test))

    # LightGBM Classifier
    lgbm_c_f = LGBMClassifier(n_estimators=600, max_depth=4, learning_rate=0.05,
                               reg_lambda=5, verbose=-1, random_state=42, n_jobs=1,
                               class_weight="balanced")
    lgbm_c_f.fit(X, y_stage)
    final_s_probas.append(lgbm_c_f.predict_proba(X_test))

    # Logistic Regression
    try:
        lr_f = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                   class_weight="balanced")
        lr_f.fit(X, y_stage)
        final_s_probas.append(lr_f.predict_proba(X_test))
    except Exception:
        pass

    # --- Feature importance ---
    importances = cb_r_f.get_feature_importance()
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print("\n  Top 20 Features (CatBoost Goals):")
    print(imp_df.head(20).to_string(index=False))

    if HAS_WANDB:
        try:
            wandb.log({"feature_importances": wandb.Table(dataframe=imp_df)})
        except Exception as e:
            print(f"  W&B warning: Failed to log feature importances: {e}")

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================
    print("\n" + "-" * 50)
    print("POST-PROCESSING")
    print("-" * 50)

    # Goals ensemble
    raw_goals = np.mean(list(final_g_preds.values()), axis=0)
    raw_goals = np.maximum(raw_goals, 0)

    # Stage ensemble: average probabilities
    n_test = len(test)
    n_classes = max(p.shape[1] for p in final_s_probas)
    aligned = []
    for p in final_s_probas:
        if p.shape[1] < n_classes:
            padded = np.zeros((n_test, n_classes))
            padded[:, :p.shape[1]] = p
            aligned.append(padded)
        else:
            aligned.append(p)
    avg_proba = np.mean(aligned, axis=0)

    # Compute team strength score (expected ordinal stage)
    strength = avg_proba @ np.arange(n_classes)

    # Rank-and-fill: assign stages based on strength ranking
    ranked_idx = np.argsort(-strength)  # strongest first
    predicted_stages = np.zeros(n_test, dtype=int)

    slot_pos = 0
    for stage_val in sorted(STAGE_SLOTS_2026.keys(), reverse=True):  # 6,5,4,3,2,1,0
        n_slots = STAGE_SLOTS_2026[stage_val]
        for _ in range(n_slots):
            if slot_pos < n_test:
                predicted_stages[ranked_idx[slot_pos]] = stage_val
                slot_pos += 1

    # Adjust goals based on predicted stage + 2026 format
    # The raw model was trained on historical data where avg matches per team
    # was lower. With 48 teams and an extra knockout round, teams play more.
    # Compute actual historical average matches per stage to get proper GPM.
    hist_avg_matches_by_stage = {}
    for s_val in range(7):
        stage_rows = train[train["stage_ordinal"] == s_val]
        if len(stage_rows) > 0:
            hist_avg_matches_by_stage[s_val] = stage_rows["matches_played"].mean()
    overall_hist_avg_matches = train["matches_played"].mean()
    print(f"  Historical avg matches per team: {overall_hist_avg_matches:.1f}")
    print(f"  Historical avg matches by stage: {hist_avg_matches_by_stage}")

    predicted_goals = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        stage = predicted_stages[i]
        matches_2026 = MATCHES_2026[stage]

        # Estimate GPM from raw model output
        # The raw model predicts total_goals trained on historical data
        # Convert to GPM using historical avg matches, then re-scale to 2026 matches
        raw_gpm = raw_goals[i] / overall_hist_avg_matches

        # Apply a small boost for quality dilution: strong teams score more
        # against weaker opponents in the expanded 48-team field
        dilution_boost = 1.0
        if stage >= 3:  # QF+ teams are strong, benefit from weaker group opponents
            dilution_boost = 1.08
        elif stage >= 1:  # R32/R16 teams get a smaller boost
            dilution_boost = 1.04

        adjusted = raw_gpm * matches_2026 * dilution_boost
        predicted_goals[i] = max(1 if stage == 0 else 2, round(adjusted))

    print(f"  Goals range: {predicted_goals.min()} - {predicted_goals.max()}")
    print(f"  Avg goals/team: {predicted_goals.mean():.1f}")

    # =========================================================================
    # BUILD SUBMISSION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUBMISSION")
    print("=" * 70)

    submission = pd.DataFrame({
        "ID": test["ID"].values,
        "total_goals": predicted_goals,
        "Target": [STAGE_LABELS_2026[s] for s in predicted_stages],
    })

    # Display predictions ranked by strength
    display = submission.copy()
    display["country"] = test["country"].values
    display["strength"] = strength.round(3)
    display["raw_goals"] = raw_goals.round(1)
    display = display.sort_values("strength", ascending=False)

    print("\nPredictions (ranked by model strength):")
    print(display[["country", "Target", "total_goals", "strength", "raw_goals"]].to_string(index=False))

    # Save
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to {output_path}")

    # --- Sanity Checks ---
    print("\n--- Sanity Checks ---")
    stage_counts = submission["Target"].value_counts()
    expected = {"group": 16, "roundof32": 16, "roundof16": 8, "qf": 4, "sf": 2, "runnerup": 1, "champion": 1}
    all_pass = True
    for stage, exp_count in expected.items():
        actual = stage_counts.get(stage, 0)
        status = "✅" if actual == exp_count else "❌"
        if actual != exp_count:
            all_pass = False
        print(f"  {status} {stage}: expected {exp_count}, got {actual}")

    print(f"\n  Total goals: {submission['total_goals'].sum()}")
    print(f"  Avg goals/team: {submission['total_goals'].mean():.1f}")
    print(f"  Min goals: {submission['total_goals'].min()}")
    print(f"  Max goals: {submission['total_goals'].max()}")

    if all_pass:
        print("\n🎉 All sanity checks passed!")

    if HAS_WANDB:
        try:
            # Save the submission file as a versioned artifact
            artifact = wandb.Artifact("world_cup_submission", type="submission")
            artifact.add_file(str(output_path))
            wandb.log_artifact(artifact)
            print("  W&B: Logged submission artifact successfully.")
        except Exception as e:
            print(f"  W&B warning: Failed to log artifact: {e}")
        finally:
            wandb.finish()

    return submission


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    train, test, sample_sub, db, tournament_winners = load_data()
    submission = run_pipeline(train, test, db, tournament_winners)
