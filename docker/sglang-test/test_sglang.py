"""Start SGLang with the competition flags on a small Qwen3.5 model and check reasoning + tool calls."""
import json, os, subprocess, sys, time, urllib.request

MODEL = sys.argv[1]
ENV = "/tmp/sg_env"
flags = sys.argv[2:]
cmd = [sys.executable, "-S", "-m", "sglang.launch_server", "--model-path", MODEL, "--served-model-name", "llm",
       "--tp", "1", "--context-length", "32768", "--mem-fraction-static", "0.80", "--port", "30000",
       "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", "--max-running-requests", "8"] + flags
print("CMD", " ".join(cmd), flush=True)
log = open("/work/sglang_server.log", "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": ENV, "PATH": ENV + "/bin:" + os.environ["PATH"]})
t0 = time.time()
while time.time() - t0 < 1200:
    if srv.poll() is not None:
        print("SERVER DIED", srv.returncode); print(open("/work/sglang_server.log").read()[-4000:]); sys.exit(1)
    try:
        urllib.request.urlopen("http://127.0.0.1:30000/v1/models", timeout=3); break
    except Exception:
        time.sleep(5)
print("READY after", round(time.time() - t0), "s", flush=True)

def chat(body):
    req = urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); return r, time.time() - t

tools = [{"type": "function", "function": {"name": "run_python", "description": "Run Python code and return stdout.",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
r, dt = chat({"model": "llm", "messages": [{"role": "user", "content": "Use the run_python tool to compute 123*457, then tell me the result."}],
              "tools": tools, "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
              "chat_template_kwargs": {"reasoning_effort": "medium"}})
m = r["choices"][0]["message"]
print("TOOL TEST finish:", r["choices"][0]["finish_reason"], "| reasoning chars:", len(m.get("reasoning_content") or ""), "| tool_calls:", json.dumps(m.get("tool_calls"))[:300], "| usage:", r.get("usage"))
r, dt = chat({"model": "llm", "messages": [{"role": "user", "content": "Write a 300-word essay about the sea."}], "max_tokens": 1500,
              "chat_template_kwargs": {"enable_thinking": False}})
n = r["usage"]["completion_tokens"]
print(f"SPEED single stream: {n} tokens in {dt:.1f}s = {n / dt:.1f} tok/s")
spec = [l for l in open("/work/sglang_server.log") if "accept" in l.lower()][-3:]
print("SPEC LOG:", "".join(spec)[-600:])
srv.terminate()
