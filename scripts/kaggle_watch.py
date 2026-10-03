"""Snapshot public notebooks, discussions and the leaderboard, and print what changed since the last snapshot.

    uv run --no-project python scripts/kaggle_watch.py
"""
import csv
import datetime
import io
import json
import pathlib
import subprocess
import zipfile

COMP = "arc-prize-2026-arc-agi-2"
OUT = pathlib.Path.home() / "arc2-local" / "watch"
OUT.mkdir(parents=True, exist_ok=True)


def kaggle_csv(*args):
    res = subprocess.run(["uvx", "kaggle", *args, "--csv"], capture_output=True, text=True, encoding="utf-8",
                         env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    lines = [l for l in res.stdout.splitlines() if l and not l.startswith(("Warning", "Next Page"))]
    return list(csv.DictReader(lines))


def leaderboard():
    tmp = OUT / "lb_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(["uvx", "kaggle", "competitions", "leaderboard", COMP, "--download", "-p", str(tmp), "-q"], capture_output=True)
    with zipfile.ZipFile(next(tmp.glob("*.zip"))) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8-sig")))
    for f in tmp.iterdir():
        f.unlink()
    scores = [float(r["Score"]) for r in rows]
    n = len(scores)
    ranks = {"1": 1, "4": 4, "10": 10, "gold": 10 + n // 500, "20": 20, "silver": round(n * 0.05), "bronze": round(n * 0.10)}
    return {"teams": n, **{name: scores[r - 1] for name, r in ranks.items()}}


snap = {
    "time": datetime.datetime.now().isoformat(timespec="minutes"),
    "kernels": {},
    "topics": {},
    "lb": leaderboard(),
}
for sort in ["voteCount", "scoreDescending", "dateRun"]:
    for r in kaggle_csv("kernels", "list", "--competition", COMP, "--sort-by", sort, "--page-size", "50"):
        snap["kernels"][r["ref"]] = {"title": r["title"], "votes": int(r["totalVotes"]), "lastRun": r["lastRunTime"][:16]}
for sort in ["new", "hot", "top"]:
    for r in kaggle_csv("competitions", "topics", "list", COMP, "-s", sort):
        snap["topics"][r["id"]] = {"title": r["title"], "votes": int(r["votes"]), "comments": int(r["commentCount"]), "posted": r["postDate"][:16]}

previous = sorted(OUT.glob("snapshot_*.json"))
old = json.loads(previous[-1].read_text(encoding="utf-8")) if previous else {"kernels": {}, "topics": {}, "lb": {}, "time": "none"}
(OUT / f"snapshot_{datetime.datetime.now():%Y%m%d_%H%M}.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"snapshot {snap['time']}  (previous: {old['time']})")
print("\nLeaderboard:")
for k, v in snap["lb"].items():
    before = old["lb"].get(k)
    print(f"  {k:7} {v:8.2f}" + (f"  ({v - before:+.2f})" if before is not None and before != v else ""))

print("\nNew notebooks:")
for ref, k in snap["kernels"].items():
    if ref not in old["kernels"]:
        print(f"  [{k['votes']:3}] {k['title'][:60]:60} {ref}  run {k['lastRun']}")
print("\nNotebooks with +5 votes or a new run:")
for ref, k in snap["kernels"].items():
    o = old["kernels"].get(ref)
    if o and (k["votes"] - o["votes"] >= 5 or k["lastRun"] != o["lastRun"]):
        print(f"  [{o['votes']:3} -> {k['votes']:3}] {k['title'][:60]:60} {ref}  run {k['lastRun']}")

print("\nNew discussions:")
for tid, t in snap["topics"].items():
    if tid not in old["topics"]:
        print(f"  {tid} [{t['votes']:3}v {t['comments']:2}c] {t['title'][:90]}")
print("\nDiscussions with new comments:")
for tid, t in snap["topics"].items():
    o = old["topics"].get(tid)
    if o and t["comments"] != o["comments"]:
        print(f"  {tid} [{o['comments']} -> {t['comments']}c] {t['title'][:90]}")
