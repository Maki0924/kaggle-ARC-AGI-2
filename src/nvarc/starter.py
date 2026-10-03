import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    work_dir = os.getenv("ARC_WORK_DIR", "/kaggle")

    # Fix Unsloth patching issue
    if rank > 0:
        wait_start = time.time()
        # Capped so a worker that died during import cannot block the others forever.
        while not os.path.exists(os.path.join(work_dir, f"worker{rank-1}")) and time.time() - wait_start < 900:
            time.sleep(5)
    
    from arc_solver import worker

    with open(os.path.join(work_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    data_dir = os.getenv("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
    if rerun_mode:
        test_path = os.path.join(data_dir, "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")

    num_gpus = int(os.getenv("ARC_NUM_GPUS", "4"))

    # Outside the competition rerun only a few puzzles are solved; ARC_TASKS (comma separated) overrides them.
    # The default is the screen8 subset of the dev split (splits/eval_split.json), never holdout puzzles.
    debug_keys = os.getenv("ARC_TASKS", "4c7dc4dd,71e489b6,7b3084d4,8b9c3697,b6f77b65,c4d067a0,dbff022c,eee78d87").split(",")

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in debug_keys:
                continue
        queue.put(key)
    for _ in range(num_gpus):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=num_gpus)
