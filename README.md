This repository contains a complete, modular election-forecasting framework for simulating the 2026 Bangladesh Parliamentary Election using a Monte Carlo model.



This project simulates the 2026 general election by combining:

1. Historical Election Data (1991–2018)

Each election contributes weighted information. Special adjustments are applied:

1991, 1996, 2008 → normal, down-weighted because of generational turnover

2001 → BNP–JeI vote alignment rule (e.g., 80/20 split)

2018 → Awami League “prefilled vote inflation” correction (user-defined %)

2. National Polling (InnoVision & IRI  2025)

Decided voters use direct polling.
Undecided voters are distributed between BNP and JeI based on a user-controlled split (e.g., 60/40).

3. AL Ban Scenario (2026)

The forecast assumes Awami League cannot contest:

X% of AL votes → Jatiya Party

Y% of AL votes → non-voting

(Parameters configurable)

4. Constituency-Level Swing Model

Baseline vote shares come from a chosen historical base election (default: 2008).
A national uniform swing is applied to match the desired 2026 vote totals.

5. Monte Carlo Simulation

Each constituency’s vote shares are drawn from a Dirichlet distribution:

Reflects uncertainty

Preserves the vote share proportions

Generates a seat winner per simulation

Running thousands of simulations generates:

Seat histograms

Probabilities of different outcomes

Coalition seat counts (e.g., JeI + Others alliance)

1. Load the base model
from monte_carlo_2026_election import BangladeshElectionBase
from scenarios_2026_election import run_scenario

BASE = BangladeshElectionBase("BD_Election_Final.xlsx")

2. Run a scenario
results = run_scenario(
    BASE,
    AL_prefill_2018=0.20,
    AL_to_JP_2026=0.80,
    poll_undecided_split=(0.60, 0.40),
    JeI_BNP_split_2001=(0.40, 0.60),
    coalition=True
)

3. View results
from results_2026_election import print_results

print_results(results)

You can modify:

| Parameter              | Meaning                                 |
| ---------------------- | --------------------------------------- |
| `AL_prefill_2018`      | % of AL votes in 2018 considered “real” |
| `JeI_BNP_split_2001`   | JeI vote retention vs BNP gain (2001)   |
| `poll_undecided_split` | BNP / JeI share of undecided voters     |
| `AL_to_JP_2026`        | AL voters moving to Jatiya Party        |
| `AL_nonvote_2026`      | AL voters dropping out                  |
| `w_hist`, `w_poll`     | Historical vs polling blend             |
| `N_SIMS`               | Monte Carlo simulation count            |
| `coalition=True`       | Whether JeI + Others run as allies      |
