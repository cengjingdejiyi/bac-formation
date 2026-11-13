PY ?= python
export PYTHONPATH := src

smoke:
	$(PY) scripts/smoke.py

test:
	$(PY) -m pytest -q tests

train:
	$(PY) -m bacmarl.runners.train env=formation algo=bac env.n_agents=6 algo.msg_bits=4 seed=0

baselines:
	$(PY) -m bacmarl.runners.train env=formation algo=ippo seed=0

figures:
	$(PY) scripts/make_figures.py

demo:
	$(PY) render_demo.py

.PHONY: smoke test train baselines figures demo
