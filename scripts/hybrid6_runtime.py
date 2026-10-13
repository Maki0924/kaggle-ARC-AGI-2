"""Embedded in the hybrid6 runner by build_hybrid.py; globals come from that runner.

Matches qwen_ab's total-token loop, with production HTTP/tool deadlines and receipts.
No forced final, partial-turn grid recovery, or silent direct-mode fallback.
"""


def count_agent_prompt(messages, effort, deadline_s):
    body = {"model": MODEL_NAME, "messages": messages, "add_generation_prompt": True,
            "tools": [SANDBOX.TOOL_SCHEMA],
            "chat_template_kwargs": {"enable_thinking": SETTINGS["enable_thinking"],
                                     "reasoning_effort": effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=request_timeout_s(120, deadline_s)) as response:
        count = json.loads(response.read())["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("invalid tokenizer count")
    return count


def solve_one_agent(tid, qi, test_input, slot, policy_name):
    task = {"train": challenges[tid]["train"], "test_input": test_input}
    policy = PASS_POLICIES[policy_name]
    messages = [{"role": "user", "content": build_agent_prompt(task)}]
    seed = stable_sample_seed(tid, qi, slot)
    t0 = time.time()
    deadline = min(GENERATION_DEADLINE_S, t0 + policy["episode_budget_s"])
    used = turns = tool_calls = tool_successes = tool_errors = 0
    finish, final, reasoning = "max_turns", "", ""
    cap = None
    raw = []
    try:
        for turn in range(MAX_TURNS):
            if time.time() >= deadline:
                finish = "episode_budget"
                break
            if used >= HYBRID_TOTAL_TOKENS:
                finish = "budget_exhausted"
                break
            prompt_tokens = count_agent_prompt(messages, policy["reasoning_effort"], deadline)
            cap = min(HYBRID_TOTAL_TOKENS - used, CONTEXT_LEN - prompt_tokens)
            if cap <= 0:
                finish = "context_exhausted"
                break
            response = _agent_chat(messages, seed=seed, max_tokens=cap,
                                   reasoning_effort=policy["reasoning_effort"], deadline_s=deadline)
            choice = response["choices"][0]
            msg = choice["message"]
            completion = (response.get("usage") or {}).get("completion_tokens")
            if not isinstance(completion, int) or isinstance(completion, bool) or not 0 <= completion <= cap:
                raise ValueError("missing/invalid completion_tokens; cannot enforce total budget")
            used += completion
            turns += 1
            raw.append({"turn": turn, "prompt_tokens": prompt_tokens, "max_tokens": cap,
                        "response": response})
            if time.time() >= deadline:
                finish = "episode_budget"
                break
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            reasoning = response_reasoning(msg)
            calls = msg.get("tool_calls") or []
            turn_finish = choice.get("finish_reason")
            if not calls and ("<tool_call>" in content or "<tool_call>" in reasoning):
                finish = "protocol_error:tool_call_markup"
                break
            if calls and turn_finish != "length":
                messages.append(assistant_replay_message(msg))
                for call in calls:
                    if time.time() >= deadline:
                        raise TimeoutError("tool deadline reached")
                    fn = call.get("function") or {}
                    code = decode_tool_code(fn) if fn.get("name") == "run_python" else ""
                    tool_calls += 1
                    if not code:
                        output = "ERROR: unknown tool or empty code"
                        tool_errors += 1
                    else:
                        result = run_python_tool(code, task, deadline)
                        output = result.stdout if result.ok else "ERROR: " + (result.error or "")
                        tool_successes += int(result.ok)
                        tool_errors += int(not result.ok)
                    messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                     "name": fn.get("name") or "unknown_tool", "content": output or "(no output)"})
                continue
            final, finish = content, turn_finish or "protocol_error:missing_finish"
            break
    except Exception as exc:
        finish = f"error:{type(exc).__name__}:{str(exc)[:160]}"
    grid = validated_model_grid(parse_model_grid(final)) if finish == "stop" else None
    outcome = {"grid": grid, "finish": finish, "content": final, "reasoning": reasoning,
               "completion_tokens": used, "wall_s": round(time.time() - t0, 1),
               "method": "python-tool" if tool_calls else "direct", "tool_calls": tool_calls,
               "tool_successes": tool_successes, "tool_errors": tool_errors, "turns": turns,
               "context_retries": 0, "forced_final": False, "recovery_attempted": False,
               "recovery_error": None, "policy": policy_name, "reasoning_effort": policy["reasoning_effort"],
               "turn_max_tokens": HYBRID_TOTAL_TOKENS, "request_max_tokens": cap}
    # A unique file per query/slot: concurrent workers never append to the same file.
    artifact_dir = WORK / "qwen64_samples"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / f"{tid}_{qi}_{slot}.json").write_text(json.dumps(
        {"task_id": tid, "query_index": qi, "slot": slot, "messages": messages,
         "turns": raw, "outcome": outcome}), encoding="utf-8")
    return outcome
