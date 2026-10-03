import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
data_dir = os.getenv("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
work_dir = os.getenv("ARC_WORK_DIR", "/kaggle")
dir_outputs = os.path.join(work_dir, "inference_outputs")

if rerun_mode:
    data = ArcDataset.from_file(os.path.join(data_dir, "arc-agi_test_challenges.json"))
else:
    data = ArcDataset.from_file(os.path.join(data_dir, "arc-agi_evaluation_challenges.json"))
    data = data.load_replies(os.path.join(data_dir, "arc-agi_evaluation_solutions.json"))

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

# A valid submission.json must exist even if every worker died.
submission = data.get_submission()
try:
    decoder.load_decoded_results(dir_outputs)
    ArcDataset.fill_submission(decoder.run_selection_algo(), submission)
except Exception as e:
    print(f"*** Selection failed, writing what is available: {type(e).__name__}: {e}")

with open("submission.json", "w") as f:
    json.dump(submission, f)

if not rerun_mode:
    for name in sorted(os.listdir(work_dir)):
        if name.startswith("timing_rank"):
            print(open(os.path.join(work_dir, name)).read())
    try:
        decoder.benchmark_selection_algos()
    except ValueError as e:  # nothing solved
        print(f"*** Benchmark skipped: {e}")
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))
