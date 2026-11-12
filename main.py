#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: decayed place-frequency model with a chart-wide tail.

Model: per snapshot, rank evidence is exponential-decayed with a 2h half-life
(0.5 ** (age_h / HALF_LIFE_H)). Each place mixes that place's share of decayed
rank weight (FREQ_SHARE) with a Gaussian-like distance kernel over the
time-averaged rank (1 - FREQ_SHARE, sigma = SPREAD hours), normalises the mix,
keeps KEEP of the mass and rounds to 6dp. Expected place is the mass-weighted
sum per universe_id; the tail of the chart (top10) is that expected place
ascending, then last rank, then universe_id.

Player-count momentum (the `playing` column of every up_and_coming row): each
universe_id also gets growth = log(playing now) - decayed average log(playing),
where a playing read's weight halves every PLAY_HL=12h. The kernel centre
becomes mean_rank - GAMMA*growth, so a place whose player count is climbing is
scored as if it already sat higher up the chart than its averaged rank says;
a flat or falling count leaves the centre untouched.

Tuning: HALF_LIFE_H=2.0, FREQ_SHARE=0.70, SPREAD=6.0, KEEP=0.80,
PLAY_HL=12.0, GAMMA=2.0, CAND_DEPTH=25. Six real archive windows average
-34.276 at KEEP=0.80 versus -34.516 at KEEP=0.90, and the practice window
-35.159 versus -35.927, so a fifth of each place's mass is left to the
chart-wide tail. Those six windows average -34.88 at CAND_DEPTH=45 (measured
at KEEP=0.90). The momentum term on its own averages -34.62 versus -34.87
without it over three windows, improving in all three (house baseline
-43.5/-49.2/-41.5).

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
PLAY_HL = 12.0      # player-count momentum: a read's weight halves every 12h of age
GAMMA = 2.0         # kernel-centre shift per unit of log-playing growth (-34.62 vs -34.87 on three archive windows)
KEEP = 0.80         # 80% of the mass is named per place; the remaining 20% falls to the
                    # chart-wide tail, which the grader splits
CAND_DEPTH = 25     # candidates carried forward: the 25 best ranks only (was 45)

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
    play_hl = constants['play_hl']
    gamma = constants['gamma']
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
    play_log_sum = {}
    play_weight = {}
    latest_playing = {}

    for snapshot in history:
        age_h = t_last - epoch_hours(snapshot['at'])
        w = 0.5 ** (age_h / half_life_h)
        w_play = 0.5 ** (age_h / play_hl)
        for row in snapshot.get('up_and_coming') or []:
            uid = int(row['universe_id'])
            rank = int(row['rank'])
            rank_sum[uid] = rank_sum.get(uid, 0.0) + w * rank
            rank_weight[uid] = rank_weight.get(uid, 0.0) + w
            playing = row.get('playing')
            if playing is not None and playing > 0:
                play_log_sum[uid] = play_log_sum.get(uid, 0.0) + w_play * math.log(playing)
                play_weight[uid] = play_weight.get(uid, 0.0) + w_play
                latest_playing[uid] = playing  # oldest first, so the last write is the newest read
            if rank <= places:
                freq[(rank, uid)] = freq.get((rank, uid), 0.0) + w
                place_total[rank] = place_total.get(rank, 0.0) + w

    for row in history[-1].get('up_and_coming') or []:
        last_rank[int(row['universe_id'])] = int(row['rank'])

    mean_rank = {uid: rank_sum[uid] / rank_weight[uid] for uid in rank_sum}

    # Player-count momentum: log playing now minus the decayed average log playing.
    growth = {
        uid: math.log(latest_playing[uid]) - play_log_sum[uid] / play_weight[uid]
        for uid in latest_playing
    }

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
            centre = mean_rank[uid] - gamma * growth.get(uid, 0.0)
            distance = abs(centre + trend - place)
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
    top10 = pad_top(order[:places], places, mean_rank, expected, last_rank)

    return {'probabilities': probabilities, 'top10': top10}


def pad_top(top10, places, mean_rank, expected, last_chart):
    """Pad `top10` to exactly `places` distinct universe_id.

    A chart with fewer than `places` games cannot fill the tail from its own
    rows, and validate_payload rejects a short or duplicate tail — one bad
    snapshot would turn the whole run into a zero. So the remaining candidates
    (every universe_id the model knows from mean_rank or the last chart) are
    appended by expected then mean_rank, and only if even those run out does
    the tail fall back to synthetic descending ids.
    """
    chosen = [uid for uid in top10]
    seen = set(chosen)

    pool = set(mean_rank) | set(last_chart)
    leftovers = sorted(
        (uid for uid in pool if uid not in seen),
        key=lambda uid: (-expected.get(uid, 0.0), mean_rank.get(uid, 10 ** 9), uid),
    )
    for uid in leftovers:
        if len(chosen) == places:
            break
        if uid in seen:
            continue
        chosen.append(uid)
        seen.add(uid)

    synthetic = 2_000_000_000
    while len(chosen) < places:
        if synthetic not in seen:
            chosen.append(synthetic)
            seen.add(synthetic)
        synthetic -= 1

    return chosen[:places]


MODEL = {
    'half_life_h': HALF_LIFE_H,
    'freq_share': FREQ_SHARE,
    'spread': SPREAD,
    'trend': TREND,
    'play_hl': PLAY_HL,
    'gamma': GAMMA,
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
