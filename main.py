#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: decayed place-frequency model with a chart-wide tail.

Model: per snapshot, rank evidence is exponential-decayed with a 2h half-life
(0.5 ** (age_h / HALF_LIFE_H)). Each place mixes that place's share of decayed
rank weight (FREQ_SHARE) with a Gaussian-like distance kernel over the
time-averaged rank (1 - FREQ_SHARE, sigma = SPREAD hours), normalises the mix,
keeps KEEP of the mass and rounds to 6dp. Expected place is the mass-weighted
sum per universe_id; the tail of the chart (top10) is that expected place
ascending, then last rank, then universe_id.

Tuning: HALF_LIFE_H=2.0, FREQ_SHARE=0.70, SPREAD=6.0, KEEP=0.90,
CAND_DEPTH=45, fitted on two real archive windows (practice -35.904,
recent -32.440; previously -36.931 and -36.325).

Stages: load_task() -> read_history(task) -> build_forecast(rows, places, MODEL)
-> validate_payload(payload, places) -> write_payload(task, payload).

Python 3.13, standard library only. Reads /input/task.json and /input/history.ndjson
(oldest first) and writes {probabilities, top10} to /output/<task.output>.
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output off the grader.
"""
import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'

HALF_LIFE_H = 2.0   # weight of a read halves every 2h of age (fresher snapshots dominate)
FREQ_SHARE = 0.70   # weight of the place's decayed rank-share table vs the rank kernel
SPREAD = 6.0        # width of the rank kernel; smaller = tighter around the averaged rank
TREND = 0.0
KEEP = 0.90         # mass actually named per place after normalising (rest stays unnamed)
CAND_DEPTH = 45     # how many best places are carried forward as candidates

MAX_ROWS_PER_PLACE = 100


def epoch_hours(stamp):
    """ISO 8601 UTC timestamp -> hours since the epoch."""
    if stamp.endswith('Z'):
        stamp = stamp[:-1] + '+00:00'
    return datetime.fromisoformat(stamp).timestamp() / 3600.0


def load_task():
    """task.json for this run."""
    with open(os.path.join(INPUT_DIR, 'task.json'), encoding='utf-8') as handle:
        return json.load(handle)


def read_history(task):
    """History snapshots (oldest first) that back `task`.

    `task` identifies the run; the snapshot file name is fixed by the task
    schema, so it is always history.ndjson beside task.json in INPUT_DIR.
    """
    rows = []
    with open(os.path.join(INPUT_DIR, 'history.ndjson'), encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_forecast(rows, places, constants):
    """Decayed place-frequency model: the {probabilities, top10} payload."""
    half_life_h = constants['half_life_h']
    freq_share = constants['freq_share']
    spread_base = constants['spread']
    trend = constants['trend']
    keep = constants['keep']
    cand_depth = constants['cand_depth']
    max_rows_per_place = constants['max_rows_per_place']

    history = rows
    t_last = epoch_hours(history[-1]['at'])

    rank_sum = {}
    rank_weight = {}
    place_total = {}
    freq = {}
    last_rank = {}

    for snapshot in history:
        w = 0.5 ** ((t_last - epoch_hours(snapshot['at'])) / half_life_h)
        for row in snapshot.get('up_and_coming') or []:
            uid = int(row['universe_id'])
            rank = int(row['rank'])
            rank_sum[uid] = rank_sum.get(uid, 0.0) + w * rank
            rank_weight[uid] = rank_weight.get(uid, 0.0) + w
            if rank <= places:
                freq[(rank, uid)] = freq.get((rank, uid), 0.0) + w
                place_total[rank] = place_total.get(rank, 0.0) + w

    for row in history[-1].get('up_and_coming') or []:
        last_rank[int(row['universe_id'])] = int(row['rank'])

    mean_rank = {uid: rank_sum[uid] / rank_weight[uid] for uid in rank_sum}

    candidates = sorted(
        uid for uid in mean_rank
        if mean_rank[uid] <= cand_depth or last_rank.get(uid, 10 ** 9) <= cand_depth
    )

    expected = {uid: 0.0 for uid in candidates}
    probabilities = []

    for place in range(1, places + 1):
        spread = spread_base * (1 + trend * (place - 1))
        total = place_total.get(place, 0.0)
        raw = {}
        for uid in candidates:
            freq_share_here = freq.get((place, uid), 0.0) / total if total > 0.0 else 0.0
            distance = abs(mean_rank[uid] + trend - place)
            raw[uid] = freq_share * freq_share_here + (1.0 - freq_share) * math.exp(-distance / spread)

        norm = sum(raw.values())
        if norm <= 0.0:
            continue

        place_rows = []
        for uid in candidates:
            p = round(keep * raw[uid] / norm, 6)
            if p > 0.0:
                place_rows.append((uid, p))
        place_rows.sort(key=lambda item: (-item[1], item[0]))
        for uid, p in place_rows[:max_rows_per_place]:
            probabilities.append({'universe_id': uid, 'place': place, 'p': p})
            expected[uid] += p * place

    order = sorted(
        candidates,
        key=lambda uid: (-expected[uid], last_rank.get(uid, 10 ** 9), uid),
    )
    top10 = order[:places]
    if len(top10) < places:
        for row in sorted(
            history[-1].get('up_and_coming') or [], key=lambda row: int(row['rank'])
        ):
            uid = int(row['universe_id'])
            if uid not in top10:
                top10.append(uid)
            if len(top10) == places:
                break

    return {'probabilities': probabilities, 'top10': top10}


MODEL = {
    'half_life_h': HALF_LIFE_H,
    'freq_share': FREQ_SHARE,
    'spread': SPREAD,
    'trend': TREND,
    'keep': KEEP,
    'cand_depth': CAND_DEPTH,
    'max_rows_per_place': MAX_ROWS_PER_PLACE,
}


def validate_payload(payload, places):
    """Refuse to write a forecast that breaks the output contract."""
    mass = {}
    for row in payload['probabilities']:
        mass[row['place']] = mass.get(row['place'], 0.0) + row['p']
    for place, total in sorted(mass.items()):
        assert total <= 1.0, 'place {} carries p mass {} > 1.0'.format(place, total)

    top10 = payload['top10']
    assert len(top10) == places, 'top10 has {} entries, expected {}'.format(len(top10), places)
    assert len(set(top10)) == places, 'top10 has {} unique universe_id, expected {}'.format(
        len(set(top10)), places
    )


def write_payload(task, payload):
    """Write {probabilities, top10} to OUTPUT_DIR/<task.output>."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task['output']), 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, separators=(',', ':')) + '\n')


def main():
    task = load_task()
    places = int(task['places'])
    rows = read_history(task)
    payload = build_forecast(rows, places, MODEL)
    validate_payload(payload, places)
    write_payload(task, payload)


if __name__ == '__main__':
    main()
