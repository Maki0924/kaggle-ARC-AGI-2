"""Fix the dev/holdout split of the public evaluation set and pick the L0 timing subset."""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
tasks = json.load(open(ROOT / "data" / "arc-agi_evaluation_challenges.json"))


def size(task):
    return sum(len(g) * len(g[0]) for ex in task["train"] for g in ex.values()) + sum(len(ex["input"]) * len(ex["input"][0]) for ex in task["test"])


keys = sorted(tasks)
random.Random(20261002).shuffle(keys)
dev, holdout = sorted(keys[:80]), sorted(keys[80:])

# 20 dev tasks spread evenly over the size range, for timing measurements
by_size = sorted(dev, key=lambda k: size(tasks[k]))
timing = sorted(by_size[i * len(by_size) // 20 + 2] for i in range(20))

out = {"dev": dev, "holdout": holdout, "timing20": timing}
(ROOT / "splits" / "eval_split.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print(len(dev), len(holdout), len(timing))
print("sizes of timing20:", [size(tasks[k]) for k in by_size if k in timing])
