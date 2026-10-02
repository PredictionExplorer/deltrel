# Pie self-play requires symmetric search strength

The pie shortcut assumes that keeping and swapping have opposite values. This
holds for the current-player-relative state only when both physical seats have
the same playout-doubling advantage (PDA). Self-play now always assigns `(0, 0)`
to pie games. Handicap compensation and random PDA in non-pie even games retain
their previous values and hashed seed streams.

The previous actor accidentally included pie in `asymmetric_pda_fraction`,
although the variant-mixture contract describes random advantages for standard
and classic non-pie games. A deterministic reproduction with seed 17, the
`cohort-v1` contract, cohort 0, manual actor identity, model identity `probe`,
fraction 0.2 and 1,000 games assigned nonzero PDA to **213 games** in each of
classic and double pie. This reproduces the code path, not a measurement of the
live replay's proportions.

Let `K(d)` be the responder's keep value at PDA `d`. After swapping, the other
physical seat moves and its PDA is `-d`. The responder's swap value is therefore
`-K(-d)`, and the opener's value is `-max(K(d), -K(-d))`. The existing shortcut
`-abs(K(d))` requires `K(d) = K(-d)`. For example, `K(+2) = -0.1` and
`K(-2) = -0.7` gives an actual opener value of -0.7, while the shortcut gives
-0.1. If `K(+2) = +0.1`, the zero-threshold rule keeps even though swapping gives
+0.7.

The native-state regression checks the input equivalence directly. After
recoloring and removing the swap option in a comparison-only keep state, relative
stones and history match. With PDA +2 before the swap and -2 afterward, global
feature 24 is the sole difference. At PDA zero, all encoded inputs match. The
production actor does not construct such a comparison state or alter replay.

New pie decisions record `pie_pda=symmetric-seats-v1` beside their actual PDA
values. Hash-based game/search streams do not advance when a PDA draw is skipped.
Legacy profiles with a nonzero random-PDA fraction remain readable; the fraction
now applies only to non-pie even games. Existing replay is not relabeled.

The arena already uses PDA zero for pie. Serving defaults to PDA zero, and the
browser uses that default. Direct native and serving APIs can still accept
nonzero PDA: callers must use PDA zero at pie opening/swap decision points until
explicit keep and swap branches are implemented for asymmetric strength. This
actor correction does not establish playing-strength improvement by itself.
