"""CPU regression checks for publication, token accounting and generated notebooks."""
import ast
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

from a100.hybrid_policy import natural_answer, output_priority, output_stratum

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_output_labels_and_order(self):
        cases = [(0, 0, 0), (1, 3, 0), (3, 8, 1), (3, 8, 4)]
        self.assertEqual([output_stratum(c) for c in cases], ["Z", "O", "W", "strong"])
        self.assertEqual(sorted(cases, key=output_priority), [cases[2], cases[1], cases[0], cases[3]])

    def test_only_explicit_natural_stop(self):
        self.assertTrue(natural_answer({"finish": "stop"}))
        for row in [{}, {"finish": "length"}, {"finish": "stop", "forced_final": True},
                    {"finish": "stop", "recovery_attempted": True}]:
            self.assertFalse(natural_answer(row))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.requests = []
        self.replies = []
        self.prompt_sizes = [10]
        self.ns = dict(json=json, WORK=pathlib.Path(self.tmp.name), time=types.SimpleNamespace(time=lambda: 100),
                       challenges={"task": {"train": [{"input": [[1]], "output": [[2]]}]}},
                       PASS_POLICIES={"xhigh": {"reasoning_effort": "xhigh", "episode_budget_s": 3600}},
                       GENERATION_DEADLINE_S=10000, MAX_TURNS=64, HYBRID_TOTAL_TOKENS=65536, CONTEXT_LEN=131072,
                       stable_sample_seed=lambda *a: 42, build_agent_prompt=lambda task: "prompt",
                       response_reasoning=lambda msg: msg.get("reasoning", ""),
                       assistant_replay_message=lambda msg: msg, decode_tool_code=lambda fn: "print(1)",
                       run_python_tool=lambda *a: types.SimpleNamespace(ok=True, stdout="1", error=None),
                       validated_model_grid=lambda grid: grid,
                       parse_model_grid=lambda text: [[2]] if text == "answer" else None)
        exec(compile((ROOT / "scripts/hybrid6_runtime.py").read_text(), "hybrid6_runtime.py", "exec"), self.ns)
        self.ns["count_agent_prompt"] = lambda *a: self.prompt_sizes.pop(0)

        def chat(messages, **kwargs):
            self.requests.append(kwargs)
            return self.replies.pop(0)
        self.ns["_agent_chat"] = chat

    def response(self, finish="stop", tokens=1, content="answer", calls=None):
        return {"choices": [{"finish_reason": finish, "message": {"content": content, "tool_calls": calls or []}}],
                "usage": {"completion_tokens": tokens}}

    def solve(self):
        return self.ns["solve_one_agent"]("task", 0, [[1]], 0, "xhigh")

    def tool_response(self, tokens, content=""):
        return self.response("tool_calls", tokens, content,
                             [{"id": "call", "function": {"name": "run_python", "arguments": "{}"}}])

    def test_budget_is_total_across_tools(self):
        self.prompt_sizes = [10, 50000]
        self.replies = [self.tool_response(40000), self.response(tokens=25000)]
        result = self.solve()
        self.assertEqual([r["max_tokens"] for r in self.requests], [65536, 25536])
        self.assertEqual(result["completion_tokens"], 65000)
        self.assertEqual(result["grid"], [[2]])
        self.assertEqual(result["turns"], 2)
        saved = json.loads(next(pathlib.Path(self.tmp.name).glob("qwen64_samples/*.json")).read_text())
        self.assertEqual(len(saved["turns"]), 2)

    def test_context_room_limits_request(self):
        self.prompt_sizes = [131000]
        self.replies = [self.response(tokens=50)]
        self.assertEqual(self.solve()["finish"], "stop")
        self.assertEqual(self.requests[0]["max_tokens"], 72)

    def test_length_grid_is_never_recovered(self):
        self.replies = [self.response("length", 65536)]
        result = self.solve()
        self.assertIsNone(result["grid"])
        self.assertFalse(result["recovery_attempted"])
        self.assertEqual(len(self.requests), 1)

    def test_intermediate_grid_not_promoted(self):
        self.prompt_sizes = [10, 20]
        self.replies = [self.tool_response(2, "answer"), self.response(content="no final grid")]
        self.assertIsNone(self.solve()["grid"])

    def test_budget_exhaustion_after_tool(self):
        self.replies = [self.tool_response(65536)]
        result = self.solve()
        self.assertEqual(result["finish"], "budget_exhausted")
        self.assertIsNone(result["grid"])
        self.assertEqual(len(self.requests), 1)

    def test_bad_usage_and_missing_finish_fail_closed(self):
        for tokens in [None, -1, 65537, True]:
            self.prompt_sizes = [10]
            self.replies = [self.response(tokens=tokens)]
            result = self.solve()
            self.assertTrue(result["finish"].startswith("error:"))
            self.assertIsNone(result["grid"])
        self.prompt_sizes = [10]
        self.replies = [self.response(finish=None)]
        self.assertEqual(self.solve()["finish"], "protocol_error:missing_finish")

    def test_expired_deadline_never_launches(self):
        self.ns["GENERATION_DEADLINE_S"] = 99
        self.assertEqual(self.solve()["finish"], "episode_budget")
        self.assertEqual(self.requests, [])


class NotebookTests(unittest.TestCase):
    def test_generated_variants(self):
        for slug, effort in [("arc2-hybrid6-nvarc-qwen38-64k", "xhigh"),
                             ("arc2-hybrid6-nvarc-qwen38-medium-64k", "medium")]:
            nb = json.loads((ROOT / "kaggle" / slug / f"{slug}.ipynb").read_text(encoding="utf-8"))
            runner = None
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if source.startswith("%%writefile"):
                    path, source = source.split("\n", 1)
                    if "qwen_runner.py" in path:
                        runner = source
                elif source.startswith("!"):
                    continue
                ast.parse(source)
            tree = ast.parse(runner)
            solve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve_one_agent")
            self.assertNotIn("recover_grid", ast.get_source_segment(runner, solve))
            self.assertIn(f"reasoning_effort={effort!r}, turn_max_tokens=65536", runner)
            self.assertIn("samples_per_query=1", runner)
            self.assertIn("if grid is not None and not natural_answer(o):", runner)
            self.assertNotIn("queries = queries[:_waves * CONCURRENCY]", runner)
            self.assertNotIn('or nv["attempt_1"] == [[0]]', runner)


class ScoreTests(unittest.TestCase):
    def test_strict_merge_preserves_real_zero_grid_and_rejects_length_at_zero_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fixed, run = root / "fixed", root / "run"
            fixed.mkdir()
            run.mkdir()

            def write(path, value):
                path.write_text(json.dumps(value))

            write(fixed / "nvarc_submission.json", {"task": [
                {"attempt_1": [[0]], "attempt_2": [[2]]},
                {"attempt_1": [[1]], "attempt_2": [[2]]}]})
            write(fixed / "nvarc_confidence.json", {"task_0": [2, 8, 1], "task_1": [0, 0, 0]})
            write(fixed / "cohort_manifest.json", {"strata": {"task": "Z"}})
            write(root / "challenges.json", {"task": {"train": [{"input": [[1]], "output": [[2]]}],
                                                       "test": [{"input": [[1]]}, {"input": [[1]]}]}})
            write(root / "solutions.json", {"task": [[[0]], [[3]]]})
            write(run / "manifest.json", {"condition": {"samples": 2},
                                           "fixed": {"tasks": ["task"], "outputs": [["task", 0], ["task", 1]]}})
            rows = [{"task_id": "task", "query_index": i, "slot": s, "grid": [[3]],
                     "finish": "stop" if i == 0 else "length", "completion_tokens": 10,
                     "turns": 1, "tool_calls": 0, "wall_s": 1} for i in range(2) for s in range(2)]
            (run / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (run / "turns.jsonl").write_text("\n".join(json.dumps(
                {"key": f"task_{i}_{s}", "completion_tokens": 10}) for i in range(2) for s in range(2)))
            command = [sys.executable, str(ROOT / "a100/score.py"), str(run), "--fixed", str(fixed),
                       "--challenges", str(root / "challenges.json"), "--solutions", str(root / "solutions.json"),
                       "--policy", "strict", "--samples", "1", "--report-dir", str(root / "report")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report/run/scores.json").read_text())
            self.assertEqual(report["nvarc"], 0.5)
            self.assertEqual(report["merged"], 0.5)
            self.assertEqual(report["completion_tokens_incl_failed_attempts"], 20)
            self.assertEqual(report["net_by_output_stratum"]["W"]["outputs"], 1)
            self.assertFalse((run / "scores.json").exists())


if __name__ == "__main__":
    unittest.main()
