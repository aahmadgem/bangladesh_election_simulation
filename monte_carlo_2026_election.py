import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD EXCEL FILE
# ============================================================
path = r" "   #insert file path
xl = pd.ExcelFile(path)

AL_COL = "Bangladesh Awami League"

# ============================================================
# 2. DEFINE RUNTIME NORMALIZATION FUNCTION
#    (groups all non-core parties into "Other")
# ============================================================
CORE_PARTIES = {
    "Bangladesh Awami League",
    "Bangladesh Nationalist Party",
    "Jamat-E-Islami Bangladesh",
    "Jatiya Party",
}

def normalize_parties_for_model(df):
    out = pd.DataFrame()
    out["Constituency_Number"] = df["Constituency_Number"]
    out["Constituency_Name"]   = df["Constituency_Name"]

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    vote_cols = [c for c in numeric_cols if c not in ["Constituency_Number","Margin_Votes"]]

    # Core parties
    for p in CORE_PARTIES:
        out[p] = df[p] if p in df.columns else 0

    # Everything else → Other
    other_cols = [c for c in vote_cols if c not in CORE_PARTIES]
    out["Other"] = df[other_cols].sum(axis=1) if other_cols else 0

    # Margin (if present)
    if "Margin_Votes" in df.columns:
        out["Margin_Votes"] = df["Margin_Votes"]

    return out

# ============================================================
# 3. LOAD SHEETS & NORMALIZE
# ============================================================
df91_raw = xl.parse("1991")
df96_raw = xl.parse("1996")
df01_raw = xl.parse("2001")
df08_raw = xl.parse("2008")
df18_raw = xl.parse("2018")  # already clean wide

df91 = normalize_parties_for_model(df91_raw)
df96 = normalize_parties_for_model(df96_raw)
df01 = normalize_parties_for_model(df01_raw)
df08 = normalize_parties_for_model(df08_raw)
df18 = df18_raw.copy()

print("✔ Runtime normalization applied to 1991–2008")
print("✔ 2018 sheet kept unchanged\n")

# ============================================================
# 4. NATIONAL TOTAL FUNCTION
# ============================================================
def national_totals(df):
    cols = [
        "Bangladesh Awami League",
        "Bangladesh Nationalist Party",
        "Jamat-E-Islami Bangladesh",
        "Jatiya Party",
        "Other",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].sum()

nat91 = national_totals(df91)
nat96 = national_totals(df96)
nat01 = national_totals(df01)
nat08 = national_totals(df08)
nat18 = national_totals(df18)

print("National totals 1991:\n", nat91)
print("\nNational totals 2018:\n", nat18, "\n")

# ============================================================
# 5. BUILD HISTORICAL PRIOR USING YOUR RULES
# ============================================================
# weights (you may adjust)
W_1991 = 0.05
W_1996 = 0.10
W_2001 = 0.20
W_2008 = 0.25
W_2018 = 0.40

# 1991 / 1996 / 2008 standard scaling
adj91 = nat91 * W_1991
adj96 = nat96 * W_1996
adj08 = nat08 * W_2008

# 2001 special rule (BNP–JeI 80:20)
adj01 = nat01.copy()
BNP_2001 = nat01["Bangladesh Nationalist Party"]
JeI_2001 = nat01["Jamat-E-Islami Bangladesh"]

BNP_2001_adj = BNP_2001 + 0.80 * JeI_2001
JeI_2001_adj = 0.20 * JeI_2001

adj01["Bangladesh Nationalist Party"] = BNP_2001_adj
adj01["Jamat-E-Islami Bangladesh"]   = JeI_2001_adj
adj01 = adj01 * W_2001

# 2018: AL had 70% pre-filled → only 30% treated as "true" preference
adj18 = nat18.copy()
adj18["Bangladesh Awami League"] = nat18["Bangladesh Awami League"] * 0.30
adj18 = adj18 * W_2018

# Combine all adjusted elections
combined = adj91 + adj96 + adj01 + adj08 + adj18

parties_full = [
    "Bangladesh Awami League",
    "Bangladesh Nationalist Party",
    "Jamat-E-Islami Bangladesh",
    "Jatiya Party",
    "Other",
]

hist_prior = (combined[parties_full] / combined[parties_full].sum()).to_dict()

print("\nHistorical prior vote shares:\n", hist_prior, "\n")

# ============================================================
# 6. POLL PRIOR WITH UNDECIDED (SANEM Round-2)
#    - 40% decided & revealed
#    - 60% undecided → 50/50 BNP & JeI
# ============================================================
decided_BNP = 0.413
decided_JeI = 0.303
decided_AL  = 0.188
decided_JP  = 0.009
decided_others = 1 - (decided_BNP + decided_JeI + decided_AL + decided_JP)

f_decided   = 0.40
f_undecided = 0.60

poll_prior = {
    "Bangladesh Nationalist Party":
        f_decided * decided_BNP + 0.5 * f_undecided,
    "Jamat-E-Islami Bangladesh":
        f_decided * decided_JeI + 0.5 * f_undecided,
    "Bangladesh Awami League": f_decided * decided_AL,
    "Jatiya Party":            f_decided * decided_JP,
    "Other":                   f_decided * decided_others,
}

s = sum(poll_prior.values())
poll_prior = {k: v / s for k, v in poll_prior.items()}

print("Poll prior with undecided rule:\n", poll_prior, "\n")

# ============================================================
# 7. BLEND HISTORY + POLL → NATIONAL 2026 PRIOR BEFORE AL BAN
# ============================================================
w_hist = 0.2
w_poll = 0.8

PARTIES = [
    "Bangladesh Nationalist Party",
    "Jamat-E-Islami Bangladesh",
    "Jatiya Party",
    "Other",
]

baseline_2026 = {}
for p in ["Bangladesh Awami League"] + PARTIES:
    baseline_2026[p] = (
        w_hist * hist_prior.get(p, 0.0)
        + w_poll * poll_prior.get(p, 0.0)
    )

print("Baseline 2026 BEFORE AL ban:\n", baseline_2026, "\n")

# ============================================================
# 8. APPLY AL BAN (60% → JP, 40% non-voting)
# ============================================================
AL_to_JP_frac = 0.60
AL_nonvote_frac = 0.40

AL_share = baseline_2026["Bangladesh Awami League"]
nonvote_from_AL = AL_nonvote_frac * AL_share

nat_no_AL = {
    "Bangladesh Nationalist Party": baseline_2026["Bangladesh Nationalist Party"],
    "Jamat-E-Islami Bangladesh":   baseline_2026["Jamat-E-Islami Bangladesh"],
    "Jatiya Party":                baseline_2026["Jatiya Party"] + AL_to_JP_frac * AL_share,
    "Other":                       baseline_2026["Other"],
}

s_voting = sum(nat_no_AL.values())
nat_2026_final = {k: v / s_voting for k, v in nat_no_AL.items()}

print("Final 2026 national vote shares with AL banned:\n")
for p, v in nat_2026_final.items():
    print(f"{p:32s}: {v*100:5.2f}%")

print("\nNon-voting from AL ban:", nonvote_from_AL * 100, "%\n")

# ============================================================
# 9. CONSTITUENCY-LEVEL BASELINE FROM 2008 (WITH AL BAN)
#    + JeI + Others coalition
# ============================================================
df_const = df08.copy()  # 2008 as base

# Apply AL ban at constituency level (same 60/40 rule)
df_const["JP_votes_2026_base"] = (
    df_const["Jatiya Party"] + AL_to_JP_frac * df_const[AL_COL]
)
df_const["Nonvote_from_AL"] = AL_nonvote_frac * df_const[AL_COL]

# BNP / JeI / Other votes from 2008
df_const["BNP_votes_2008"]   = df_const["Bangladesh Nationalist Party"]
df_const["JeI_votes_2008"]   = df_const["Jamat-E-Islami Bangladesh"]
df_const["Other_votes_2008"] = df_const["Other"]

# Coalition votes: JeI + Others
df_const["COAL_votes_2008"] = (
    df_const["JeI_votes_2008"] + df_const["Other_votes_2008"]
)

# Total *voting* denominator (BNP + JP + JeI+Other)
df_const["Total_votes_2026_base"] = (
    df_const["BNP_votes_2008"]
    + df_const["JP_votes_2026_base"]
    + df_const["COAL_votes_2008"]
)

# Baseline shares per constituency (post-AL-ban, coalition)
df_const["share_base_BNP"]  = (
    df_const["BNP_votes_2008"] / df_const["Total_votes_2026_base"]
).fillna(0.0)
df_const["share_base_JP"]   = (
    df_const["JP_votes_2026_base"] / df_const["Total_votes_2026_base"]
).fillna(0.0)
df_const["share_base_COAL"] = (
    df_const["COAL_votes_2008"] / df_const["Total_votes_2026_base"]
).fillna(0.0)

# ============================================================
# 10. NATIONAL BASELINE VS DESIRED 2026 SHARES (COALITION)
# ============================================================
# Build coalition-national shares: JeI+Other as one party
nat_2026_final_coal = {
    "BNP":  nat_2026_final["Bangladesh Nationalist Party"],
    "JP":   nat_2026_final["Jatiya Party"],
    "COAL": nat_2026_final["Jamat-E-Islami Bangladesh"] + nat_2026_final["Other"],
}

nat_baseline_counts_coal = {
    "BNP":  (df_const["share_base_BNP"]  * df_const["Total_votes_2026_base"]).sum(),
    "JP":   (df_const["share_base_JP"]   * df_const["Total_votes_2026_base"]).sum(),
    "COAL": (df_const["share_base_COAL"] * df_const["Total_votes_2026_base"]).sum(),
}
nat_baseline_total_coal = sum(nat_baseline_counts_coal.values())

nat_baseline_shares_coal = {
    k: nat_baseline_counts_coal[k] / nat_baseline_total_coal
    for k in nat_baseline_counts_coal
}

print("National baseline shares (2008 + AL ban, JeI+Other coalition):")
print(nat_baseline_shares_coal, "\n")

# Swing factor: desired_share / baseline_share
swing_factor_coal = {
    k: nat_2026_final_coal[k] / nat_baseline_shares_coal[k]
    for k in nat_2026_final_coal
}
print("Swing factors by party (BNP, JP, COAL):", swing_factor_coal, "\n")

# ============================================================
# 11. EXPECTED 2026 SHARE PER CONSTITUENCY (AFTER NATIONAL SWING)
# ============================================================
df_const["share_exp_BNP"] = (
    df_const["share_base_BNP"] * swing_factor_coal["BNP"]
)
df_const["share_exp_JP"] = (
    df_const["share_base_JP"] * swing_factor_coal["JP"]
)
df_const["share_exp_COAL"] = (
    df_const["share_base_COAL"] * swing_factor_coal["COAL"]
)

share_cols = ["share_exp_BNP", "share_exp_JP", "share_exp_COAL"]
df_const["share_sum"] = df_const[share_cols].sum(axis=1)
df_const["share_sum"] = df_const["share_sum"].replace(0, np.nan)

for col in share_cols:
    df_const[col] = df_const[col] / df_const["share_sum"]

# Where everything was NaN (no votes at all), assign equal shares
mask_all_nan = df_const[share_cols].isna().all(axis=1)
for col in share_cols:
    df_const.loc[mask_all_nan, col] = 1.0 / len(share_cols)

df_const[share_cols] = df_const[share_cols].fillna(0.0)

print("Example expected shares for first 5 constituencies (2008 base):\n")
print(df_const[["Constituency_Number", "Constituency_Name"] + share_cols].head(), "\n")

# ============================================================
# 12. MONTE CARLO SEAT FORECAST (538-STYLE)
#      Parties: BNP, JP, JeI+Other coalition
# ============================================================
N_SIMS = 5000
K_CONC = 200.0
EPS = 1e-6

PARTY_LIST = ["BNP", "JP", "COAL"]
party_index = {p: i for i, p in enumerate(PARTY_LIST)}

exp_shares = df_const[share_cols].values
n_const = len(df_const)
n_parties = len(PARTY_LIST)

seat_counts = np.zeros((N_SIMS, n_parties), dtype=int)
rng = np.random.default_rng(seed=42)

for s in range(N_SIMS):
    sim_seats = np.zeros(n_parties, dtype=int)

    for i in range(n_const):
        alpha_i = exp_shares[i, :] * K_CONC
        alpha_i = np.clip(alpha_i, EPS, None)  # ensure alpha > 0
        draws_i = rng.dirichlet(alpha_i)
        winner = np.argmax(draws_i)
        sim_seats[winner] += 1

    seat_counts[s, :] = sim_seats

# ============================================================
# 13. SUMMARIZE SEAT DISTRIBUTIONS
# ============================================================
def summarize(idx, name):
    seats = seat_counts[:, idx]
    mean = seats.mean()
    p05 = np.percentile(seats, 5)
    p50 = np.percentile(seats, 50)
    p95 = np.percentile(seats, 95)
    print(f"{name:15s} | mean={mean:6.1f}  P5={p05:4.0f}  P50={p50:4.0f}  P95={p95:4.0f}")

print("\nSeat distribution summaries (BNP vs JP vs JeI+Other coalition):\n")
summarize(party_index["BNP"],  "BNP")
summarize(party_index["JP"],   "Jatiya Party")
summarize(party_index["COAL"], "JeI+Other")

# Majority threshold (300 seats → 151 is majority)
majority = 151
BNP_seats  = seat_counts[:, party_index["BNP"]]
JP_seats   = seat_counts[:, party_index["JP"]]
COAL_seats = seat_counts[:, party_index["COAL"]]

prob_BNP_majority  = np.mean(BNP_seats  >= majority)
prob_JP_majority   = np.mean(JP_seats   >= majority)
prob_COAL_majority = np.mean(COAL_seats >= majority)

print("\n=== Majority Probabilities (Each bloc separately) ===")
print(f"Probability BNP ≥ {majority}:    {prob_BNP_majority*100:5.2f}%")
print(f"Probability JP ≥ {majority}:     {prob_JP_majority*100:5.2f}%")
print(f"Probability JeI+Other ≥ {majority}: {prob_COAL_majority*100:5.2f}%")

print("\nDone. 2008 is the geographic base; JeI+Other run as a coalition in 2026.")
