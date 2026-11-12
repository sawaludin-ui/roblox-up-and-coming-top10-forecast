#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: decayed place-frequency model with a chart-wide tail.

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

HALF_LIFE_H = 6.0
FREQ_SHARE = 0.70
SPREAD = 12.0
TREND = 0.0
KEEP = 0.80
CAND_DEPTH = 45

MAX_ROWS_PER_PLACE = 100


def epoch_hours(stamp):
    """ISO 8601 UTC timestamp -> hours since the epoch."""
    if stamp.endswith('Z'):
        stamp = stamp[:-1] + '+00:00'
    return datetime.fromisoformat(stamp).timestamp() / 3600.0


def read_history(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    with open(os.path.join(INPUT_DIR, 'task.json'), encoding='utf-8') as handle:
        task = json.load(handle)
    places = int(task['places'])
    history = read_history(os.path.join(INPUT_DIR, 'history.ndjson'))

    t_last = epoch_hours(history[-1]['at'])

    rank_sum = {}
    rank_weight = {}
    place_total = {}
    freq = {}
    last_rank = {}

    for snapshot in history:
        w = 0.5 ** ((t_last - epoch_hours(snapshot['at'])) / HALF_LIFE_H)
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
        if mean_rank[uid] <= CAND_DEPTH or last_rank.get(uid, 10 ** 9) <= CAND_DEPTH
    )

    expected = {uid: 0.0 for uid in candidates}
    probabilities = []

    for place in range(1, places + 1):
        spread = SPREAD * (1 + TREND * (place - 1))
        total = place_total.get(place, 0.0)
        raw = {}
        for uid in candidates:
            freq_share = freq.get((place, uid), 0.0) / total if total > 0.0 else 0.0
            distance = abs(mean_rank[uid] + TREND - place)
            raw[uid] = FREQ_SHARE * freq_share + (1.0 - FREQ_SHARE) * math.exp(-distance / spread)

        norm = sum(raw.values())
        if norm <= 0.0:
            continue

        rows = []
        for uid in candidates:
            p = round(KEEP * raw[uid] / norm, 6)
            if p > 0.0:
                rows.append((uid, p))
        rows.sort(key=lambda item: (-item[1], item[0]))
        for uid, p in rows[:MAX_ROWS_PER_PLACE]:
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

    payload = {'probabilities': probabilities, 'top10': top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task['output']), 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
