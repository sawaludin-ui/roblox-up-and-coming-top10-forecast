# roblox-up-and-coming-forecast-01

A forecast for Roblox's Up-and-Coming chart (`sortId=up-and-coming`, `device=computer`,
`country=us`): the ten chart places at every hourly read of the 72-hour window that follows
the close of the archive history, plus one ordered top 10.

`main.py` reads `/input/task.json` and `/input/history.ndjson` (one chart snapshot per line,
oldest first) and writes `{probabilities, top10}` to `/output/<task.output>`. It is Python
3.13, standard library only, no network: the window is forecast, never read.

## What is scored

At each of the window's reads, the grader takes the probability given to the game that
actually held each place from 1 to 10 and sums the natural logs of those ten probabilities,
each floored at `1e-4` (`ln 1e-4 = -9.210340`). The entry's score is the mean of those read
scores over the window's 72 hours: `0` at best, `-92.103404` at worst. Any probability left
unnamed at a place is shared equally by every game of that read's chart that is not named
there, so a chart-wide tail is a real hedge against a game arriving from outside the shortlist.

## Model

A causal analogue transition model with a chart-wide tail:

- Learn source/target pairs with lags of 6 to 72 hours, entirely within supplied
  history. Source features see source-and-earlier reads only. Six twelve-hour
  horizon bins contribute equally.
- Match current rank, approximately twelve-hour rank momentum and log-player
  growth. Transfer later rank distributions, not universe identities.
- Add a weak relative-universe-ID gap against the maximum ID on each source
  chart. This is an age proxy, not a publication-date cutoff. First-seen age
  is left-censored and has zero weight in the selected model.
- Rank bandwidth 3, momentum bandwidth 8, log-player bandwidth 1,
  outcome bandwidth 2, target-read half-life 120 hours, candidate depth 40,
  relative-ID weight 0.5, ID-gap bandwidth 1 million units.
- Leave 0.20 probability mass unnamed per place. Round named probabilities to
  six decimals. Order top ten by total named probability, with current rank
  and ID breaking ties.
- For sparse history shorter than six hours or another number of places,
  retain the original decay-kernel fallback.

No embedded future outcomes, absolute game-ID eligibility cutoff, or external
network requests. `main.py` ignores observations at or after `history_until`.
Repeated sufficient statistics are combined by summing their weights without
dropping reads; this preserves the rounded payload on all checked fixtures.

## Backtest

Grader-style mean log probability scores; higher is better:

| Local comparison | Previous decay model | Transition model |
|---|---:|---:|
| Seven chronological 72h windows | -40.636405 | -38.996194 |
| Practice-aligned window | -35.158562 | -36.524518 |
| Three separated windows | -42.156465 | -39.686482 |

The model wins six of seven local windows but loses on practice. The seven
windows overlap and are not independent trials. The three separated windows
start on 2026-11-03, 2026-11-06 and 2026-11-10 UTC. Repeated archive inspection
and model selection limit any holdout claim. These retrospective comparisons
do not establish final placement or payment.

Five complete process runs on each input shape were byte-identical within
that shape, all exiting zero with 400 probability rows and ten distinct IDs.
Practice took 0.666–0.782 seconds and the 301-row dense fixture 4.429–5.764
seconds on the development machine, not a claimed sandbox benchmark.
The separate dense parity check took 42.531 seconds uncompressed and 3.841
seconds compressed with exactly equal payloads. Seven archive payloads also
matched the uncompressed reference exactly. Eight malformed payloads were
rejected; an appended future observation was filtered; empty history failed
clearly. The validator checks finite probabilities, unique ID/place pairs,
valid IDs and places, per-place row limits and mass, and distinct top-ten IDs.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

`MAKEFUN_INPUT_DIR` defaults to `/input` and `MAKEFUN_OUTPUT_DIR` to `/output`, which is what
the grader leaves unset. The output is one compact JSON object with exactly two keys:
`probabilities` (rows of `{universe_id, place, p}`, at most 100 rows per place, each place
summing to at most 1) and `top10` (ten distinct universe ids in order).
