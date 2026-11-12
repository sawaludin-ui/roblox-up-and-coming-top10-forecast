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

A decayed place-frequency model with a chart-wide tail:

- Every history read is weighted `0.5 ** (hours_before_the_last_read / 6.0)`: a read one
  half-life (6 h) older counts half.
- Per game, the weighted mean chart rank; per place, the weighted share of reads that put
  each game there (places 1 to 10).
- For place `k`, `raw = 0.70 * frequency_share(k) + 0.30 * exp(-|mean_rank - k| / 12.0)`.
  The frequency term is the empirical stickiness of the place; the exponential (Laplacian)
  kernel with a 12-place spread lets a game slide a place or two without being zeroed, and
  its long tail names games whose mean rank sits anywhere within the candidate depth.
- Candidates are games with a weighted mean rank within 45 places, plus every game on the
  last chart read within 45 places; candidates are ranked, and at most 100 are named per
  place.
- Place probabilities are normalised so each place keeps 0.80 of its mass among the named
  games (0.20 is left to the unnamed tail) and rounded to six decimals.
- The top 10 orders games by the probability mass they collect over the ten places
  (`sum(p * place)`), ties broken by their last chart rank.

## Backtest

Eleven splits of the public snapshot archive (train on the reads before a split, score the
reads after it), same scoring rule as the grader:

| Forecast | Mean log score |
|---|---|
| House baseline (persistence, `starter/main.ts`) | -47.270 |
| This model | -39.587 |

The gain comes from the later hours of each window: the frequency term tracks which games
stick at the top, and the kernel plus the unnamed tail keep mass on games the persistence
baseline never names.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

`MAKEFUN_INPUT_DIR` defaults to `/input` and `MAKEFUN_OUTPUT_DIR` to `/output`, which is what
the grader leaves unset. The output is one compact JSON object with exactly two keys:
`probabilities` (rows of `{universe_id, place, p}`, at most 100 rows per place, each place
summing to at most 1) and `top10` (ten distinct universe ids in order).
