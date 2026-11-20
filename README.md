[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v4/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/funtoons/gta6-trophy-rarity-forecast-01-v4/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA 6 Trophy Rarity Forecast — v4

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, five
quantiles of its Exophase earn rate at 2026-11-26T00:00:00Z, from the
house's reads up to the deadline. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
`gta6-trophy-rarity-forecast-01` ($250, 20 places).

## How it works (`main.py`, Python 3.13 stdlib only, offline, deterministic)

- **Plateau-collapse.** Exophase refreshes on its own schedule, not on
  every read: consecutive identical reads are one observation, not fresh
  evidence.
- **Ensemble in logit space.** Four estimators per trophy — persistence,
  a log-time WLS trend, an earners/players decomposition (log earners and
  log players extrapolated separately, rate = ratio), and a real-time WLS
  trend — weighted by a leave-future-out backtest (fit to checkpoint
  −24h, scored on the last 24h), pooled per trophy group (platinum /
  100% / story / grind) and shrunk to global weights. The ensemble
  median is shrunk toward persistence: rates can crash 6× in one
  refresh (dilution) but can only climb as fast as players play.
- **Flood insurance (the v4 difference).** The history already holds a
  6× players jump. A dilution score — players growing faster than
  earners, per trophy, at each refresh jump — activates insurance: the
  median is anchored to persistence and the lower tail is widened
  asymmetrically (a late flood can only push rates down). When no flood
  signal is present the insurance stays off.
- **Structural priors.** Platinum earners are a subset of every trophy's
  earners; chapter trophies nest (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final mission);
  100% completion nests inside the final mission. Enforced on every
  quantile. A trophy at 100% stays near 100% with a pinned upper tail.
- **Adaptive spread.** Random-walk term in horizon, plus estimator
  disagreement, slope standard error × extrapolation distance, and
  backtest residual; floored and capped. Lower tail wider than upper.

## Measured (fresh 9-world synthetic suite, horizon 4.52 days)

| forecast | mean normal | flood-stress ×3 | flood-stress ×6 | late-flood world | 90% coverage |
|---|---|---|---|---|---|
| persistence baseline | 0.489 | 0.588 | 1.028 | 1.026 | 0.84 |
| trend baseline | 0.506 | 0.701 | 1.215 | 1.276 | 0.84 |
| **v4 (this entry)** | 0.523 | 0.536 | 0.937 | **0.935** | 0.92 |

Pinball loss on the logit scale (lower is better). v4 pays ~0.03–0.05
on calm worlds for its flood cover and wins decisively when a late
flood hits (players ×6 after the checkpoint, earners flat): 0.935 vs
1.51 for the house baseline's practice figure and 1.28 for the trend
baseline. Combined criterion (½ normal + ¼ ×3 + ¼ ×6): v4 0.629 vs
persistence 0.648 vs trend 0.732.

Practice cases p00/p01: valid, byte-identical across runs, ~0.09 s per
case. Edge cases (1 read, empty history, all-null rates, alternating
nulls, 0% trophy, 100% trophy, 1-minute horizon, 2 reads) all valid
with no crash. Full design, sweep tables and caveats:
`gta6-trophy-rarity-forecast-01-v4-progress.md` (kept outside this repo).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## License

MIT — see [LICENSE](LICENSE).
