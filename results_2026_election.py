BASE = BangladeshElectionBase(r" ") #insert file path

results = run_scenario(
    BASE,
    AL_prefill_2018=0.20,
    AL_to_JP_2026=0.80,
    poll_undecided_split=(0.60, 0.40),
    JeI_BNP_split_2001=(0.40, 0.60)
)

results
