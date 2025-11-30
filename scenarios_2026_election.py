import numpy as np
import pandas as pd

# ============================================================
#  BASE CLASS
# ============================================================
class BangladeshElectionBase:
    def __init__(self, path):
        self.xl = pd.ExcelFile(path)

        # Load raw sheets
        self.df91_raw = self.xl.parse("1991")
        self.df96_raw = self.xl.parse("1996")
        self.df01_raw = self.xl.parse("2001")
        self.df08_raw = self.xl.parse("2008")
        self.df18_raw = self.xl.parse("2018")

        # Normalize
        self.df91 = self.normalize(self.df91_raw)
        self.df96 = self.normalize(self.df96_raw)
        self.df01 = self.normalize(self.df01_raw)
        self.df08 = self.normalize(self.df08_raw)
        self.df18 = self.df18_raw.copy()

        # National totals
        self.nat91 = self.national_totals(self.df91)
        self.nat96 = self.national_totals(self.df96)
        self.nat01 = self.national_totals(self.df01)
        self.nat08 = self.national_totals(self.df08)
        self.nat18 = self.national_totals(self.df18)

        print("✔ Base model loaded successfully.\n")

    CORE = {
        "Bangladesh Awami League",
        "Bangladesh Nationalist Party",
        "Jamat-E-Islami Bangladesh",
        "Jatiya Party",
    }

    def normalize(self, df):
        out = pd.DataFrame()
        out["Constituency_Number"] = df["Constituency_Number"]
        out["Constituency_Name"]   = df["Constituency_Name"]

        numeric_cols = df.select_dtypes(include=["float","int"]).columns

        # core parties
        for p in self.CORE:
            out[p] = df[p] if p in df.columns else 0

        # all other numeric → Other
        other_cols = [c for c in numeric_cols if c not in self.CORE]
        out["Other"] = df[other_cols].sum(axis=1)

        if "Margin_Votes" in df.columns:       # FIX 1
            out["Margin_Votes"] = df["Margin_Votes"]

        return out

    def national_totals(self, df):
        cols = list(self.CORE) + ["Other"]
        for c in cols:
            if c not in df:
                df[c] = 0
        return df[cols].sum()


# ============================================================
#  RUN SCENARIO FUNCTION
# ============================================================
def run_scenario(
    BASE,
    hist_weights=(0.05,0.10,0.20,0.25,0.40),
    JeI_BNP_split_2001=(0.20,0.80),
    AL_prefill_2018=0.30,
    poll_decided_weight=0.40,
    poll_undecided_split=(0.50,0.50),
    AL_to_JP_2026=0.60,
    AL_nonvote_2026=0.40,
    w_hist=0.20,
    w_poll=0.80,
    N_SIMS=5000,
    K_CONC=200.0,
    EPS=1e-6,
    coalition=True   # JeI + Others alliance
):

    # Unpack parameters
    W91,W96,W01,W08,W18 = hist_weights
    JeI_keep, BNP_gain  = JeI_BNP_split_2001
    split_BNP, split_JeI = poll_undecided_split

    # ---------------------------------------------------------
    # 1. HISTORICAL PRIOR
    # ---------------------------------------------------------
    nat91 = BASE.nat91 * W91
    nat96 = BASE.nat96 * W96

    # 2001 BNP–JeI split
    nat01 = BASE.nat01.copy()
    BNP_raw = nat01["Bangladesh Nationalist Party"]
    JeI_raw = nat01["Jamat-E-Islami Bangladesh"]

    nat01["Bangladesh Nationalist Party"] = BNP_raw + BNP_gain * JeI_raw
    nat01["Jamat-E-Islami Bangladesh"]    = JeI_keep * JeI_raw
    nat01 = nat01 * W01

    nat08 = BASE.nat08 * W08

    nat18 = BASE.nat18.copy()
    nat18["Bangladesh Awami League"] = AL_prefill_2018 * nat18["Bangladesh Awami League"]
    nat18 = nat18 * W18

    combined = nat91 + nat96 + nat01 + nat08 + nat18

    parties = [
        "Bangladesh Awami League",
        "Bangladesh Nationalist Party",
        "Jamat-E-Islami Bangladesh",
        "Jatiya Party",
        "Other"
    ]

    hist_prior = (combined[parties] / combined[parties].sum()).to_dict()

    # ---------------------------------------------------------
    # 2. POLLING PRIOR
    # ---------------------------------------------------------
    f_dec = poll_decided_weight
    f_und = 1 - f_dec

    decided_BNP = 0.413
    decided_JeI = 0.303
    decided_AL  = 0.188
    decided_JP  = 0.009
    decided_other = 1 - (decided_BNP+decided_JeI+decided_AL+decided_JP)

    poll_prior = {
        "Bangladesh Nationalist Party":
            f_dec*decided_BNP + split_BNP * f_und,

        "Jamat-E-Islami Bangladesh":
            f_dec*decided_JeI + split_JeI * f_und,

        "Bangladesh Awami League": f_dec*decided_AL,
        "Jatiya Party":            f_dec*decided_JP,
        "Other":                   f_dec*decided_other,
    }

    S = sum(poll_prior.values())
    poll_prior = {k:v/S for k,v in poll_prior.items()}

    # ---------------------------------------------------------
    # 3. NATIONAL PRIOR BEFORE AL BAN
    # ---------------------------------------------------------
    baseline_2026 = {
        p: w_hist*hist_prior[p] + w_poll*poll_prior[p]
        for p in parties
    }

    # ---------------------------------------------------------
    # 4. APPLY AL BAN
    # ---------------------------------------------------------
    AL_share = baseline_2026["Bangladesh Awami League"]

    nat_no_AL = {
        "Bangladesh Nationalist Party": baseline_2026["Bangladesh Nationalist Party"],
        "Jamat-E-Islami Bangladesh":   baseline_2026["Jamat-E-Islami Bangladesh"],
        "Jatiya Party":                baseline_2026["Jatiya Party"] + AL_to_JP_2026 * AL_share,
        "Other":                       baseline_2026["Other"],
    }

    S2 = sum(nat_no_AL.values())
    nat_2026_final = {k:v/S2 for k,v in nat_no_AL.items()}

    # ---------------------------------------------------------
    # 5. CONSTITUENCY BASELINE FROM 2008
    # ---------------------------------------------------------
    df_const = BASE.df08.copy()

    df_const["JP_votes"] = df_const["Jatiya Party"] + AL_to_JP_2026 * df_const["Bangladesh Awami League"]
    df_const["BNP_votes"] = df_const["Bangladesh Nationalist Party"]
    df_const["JeI_votes"] = df_const["Jamat-E-Islami Bangladesh"]
    df_const["Other_votes"] = df_const["Other"]

    df_const["Total_votes"] = (
        df_const["BNP_votes"] + df_const["JeI_votes"] +
        df_const["JP_votes"] + df_const["Other_votes"]
    )

    PARTIES = ["Bangladesh Nationalist Party","Jamat-E-Islami Bangladesh","Jatiya Party","Other"]

    # baseline constituency shares
    for p, col in [
        ("Bangladesh Nationalist Party","BNP_votes"),
        ("Jamat-E-Islami Bangladesh","JeI_votes"),
        ("Jatiya Party","JP_votes"),
        ("Other","Other_votes"),
    ]:
        df_const[f"share_base_{p}"] = (df_const[col] / df_const["Total_votes"]).fillna(0)

    # ---------------------------------------------------------
    # 6. SWING FACTORS
    # ---------------------------------------------------------
    nat_counts = {
        p:(df_const[f"share_base_{p}"] * df_const["Total_votes"]).sum()
        for p in PARTIES
    }
    S3 = sum(nat_counts.values())
    nat_shares = {p:nat_counts[p]/S3 for p in PARTIES}

    swing = {p: nat_2026_final[p] / nat_shares[p] for p in PARTIES}

    # ---------------------------------------------------------
    # 7. EXPECTED 2026 CONSTITUENCY SHARE
    # ---------------------------------------------------------
    for p in PARTIES:
        df_const[f"share_exp_{p}"] = df_const[f"share_base_{p}"] * swing[p]

    share_cols = [f"share_exp_{p}" for p in PARTIES]
    df_const["sum"] = df_const[share_cols].sum(axis=1).replace(0,np.nan)

    for p in PARTIES:
        df_const[f"share_exp_{p}"] /= df_const["sum"]

    df_const[share_cols] = df_const[share_cols].fillna(1/len(PARTIES))
    exp = df_const[share_cols].values

    # ---------------------------------------------------------
    # 8. MONTE CARLO
    # ---------------------------------------------------------
    n_const = len(df_const)
    nP = len(PARTIES)
    idx = {p:i for i,p in enumerate(PARTIES)}

    seat_counts = np.zeros((N_SIMS,nP),int)
    rng = np.random.default_rng(42)

    for s in range(N_SIMS):
        seats = np.zeros(nP,int)
        for i in range(n_const):
            alpha = np.clip(exp[i]*K_CONC, EPS, None)
            draw = rng.dirichlet(alpha)
            seats[np.argmax(draw)] += 1
        seat_counts[s] = seats

    # coalition
    if coalition:
        JeI_i = idx["Jamat-E-Islami Bangladesh"]
        O_i   = idx["Other"]
        coalition_seats = seat_counts[:,JeI_i] + seat_counts[:,O_i]
    else:
        coalition_seats = None

    return {
        "national_vote_shares": nat_2026_final,
        "seat_counts": seat_counts,
        "seat_means": seat_counts.mean(axis=0),
        "party_index": idx,
        "coalition_seats": coalition_seats,
    }
