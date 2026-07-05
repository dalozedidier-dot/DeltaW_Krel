.PHONY: install test smoke micro montecarlo manifest reproduce clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	pytest -q tests

micro:
	python scripts/micro_tomography_simulation.py --outdir outputs --n-sim 200 --n-null 500

montecarlo:
	python scripts/monte_carlo_control_supplement.py --n-sim 100 --n-null 200 --dim 4 --n-noise 1 --lambda-values 0,0.005 --n-values 500 --no-plots

manifest:
	python scripts/validate_manifest.py

smoke: test micro montecarlo

reproduce: smoke manifest

clean:
	rm -rf .pytest_cache __pycache__ scripts/__pycache__ tests/__pycache__ monte_carlo_outputs_control
