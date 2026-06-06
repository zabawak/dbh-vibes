"""Validate Phase 3 re-ID output on a clip's tracks.csv (no ground-truth labels needed).

Checks the hard guarantees and label-free signals:
  - temporal soundness: no identity contains a pair of tracks whose frame spans overlap;
  - concurrency floor: identity count must be >= the max number of player tracks on the surface at
    once (a person can't be two places, so the roster is at least the peak concurrency);
  - count sanity: floor <= n_identities <= n_tracks;
  - merge team-consistency: of the track-pairs put in one identity, how many join two tracks the
    team-clusterer also called the same team (proxy for "not obviously wrong" when no GT exists).
"""
import csv
import sys
from collections import defaultdict


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["role"] == "player" and r["player"] != "":
                out[int(r["track_id"])] = dict(
                    player=int(r["player"]), first=int(r["first_frame"]),
                    last=int(r["last_frame"]), team=r["team"])
    return out


def max_concurrency(tracks):
    """Peak number of player tracks simultaneously on the surface (a floor on the roster)."""
    events = []
    for t in tracks.values():
        events.append((t["first"], 1))
        events.append((t["last"] + 1, -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def analyze(path):
    d = load(path)
    if not d:
        print(f"{path}: no player identities"); return
    byp = defaultdict(list)
    for tid, t in d.items():
        byp[t["player"]].append(tid)
    merges = [(p, ids) for p, ids in byp.items() if len(ids) > 1]

    viol = 0
    for ids in byp.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = d[ids[i]], d[ids[j]]
                if max(a["first"], b["first"]) <= min(a["last"], b["last"]):
                    viol += 1

    same = cross = checkable = pairs = 0
    for _, ids in merges:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs += 1
                ti, tj = d[ids[i]]["team"], d[ids[j]]["team"]
                if ti and tj:
                    checkable += 1
                    same += (ti == tj)
                    cross += (ti != tj)

    floor = max_concurrency(d)
    n_id, n_tr = len(byp), len(d)
    print(f"\n{path}")
    print(f"  {n_tr} player tracks -> {n_id} identities ({len(merges)} multi-track); "
          f"peak concurrency floor = {floor}")
    print(f"  temporal soundness: {viol} overlap-violation(s) -> {'PASS' if viol == 0 else 'FAIL'}")
    print(f"  count sanity (floor <= ids <= tracks): "
          f"{'PASS' if floor <= n_id <= n_tr else 'FAIL'}")
    print(f"  merged track-pairs: {pairs}; team-checkable {checkable}; "
          f"same-team {same}, cross-team {cross} "
          f"-> {'PASS' if cross == 0 else f'{cross} CROSS-TEAM'}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
