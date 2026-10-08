set -e
python3.11 -m pip install -q --no-index --find-links /wh --target /tmp/sg_env sglang==0.5.20 flashinfer-jit-cache 2>&1 | tail -3
python3.11 -c "import sys; sys.path.insert(0,'/tmp/sg_env'); import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, torch.version.cuda)"
python3.11 /work/test_sglang.py /model "$@"
