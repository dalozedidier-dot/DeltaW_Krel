.PHONY: install test coverage smoke micro realistic montecarlo sdp prereg report manifest manifest-update reproduce reproduce-full reproduce-core clean

SMOKE_MICRO_DIR ?= results/micro_smoke
SMOKE_REALISTIC_DIR ?= results/realistic_tomography_smoke
SMOKE_MC_DIR ?= results/mc_smoke
export PYTHONPATH := src:scripts:$(PYTHONPATH)

install:
	python -m pip install --upgrade pip
	python -m pip install -e .

test:
	python -m pytest -q tests

coverage:
	python -m pytest tests --cov=src/deltawkrel --cov=scripts --cov-report=term-missing --cov-fail-under=90

micro:
	python scripts/micro_tomography_simulation.py --outdir $(SMOKE_MICRO_DIR) --n-sim 200 --n-null 500

realistic:
	python scripts/realistic_tomography_pipeline.py --outdir $(SMOKE_REALISTIC_DIR) --n-sim 20 --n-null 40 --n-boot 10 --lambda-values 0,0.05 --n-total-values 1000 --visibility-values 0.95,1 --crosstalk-values 0,0.02 --drift-values 0,0.01

montecarlo:
	python scripts/monte_carlo_control_supplement.py --n-sim 100 --n-null 200 --dim 4 --n-noise 1 --lambda-values 0,0.005 --n-values 500 --output-dir $(SMOKE_MC_DIR) --no-plots

sdp:
	python scripts/run_sdp_validation.py

prereg:
	python scripts/freeze_preregistration.py

report:
	python scripts/generate_reproducibility_report.py

manifest:
	python scripts/validate_manifest.py

manifest-update:
	python scripts/generate_manifest.py

smoke: test micro realistic montecarlo

reproduce: smoke manifest

# Chaîne complète (requiert cvxpy + un solveur SDP : SCS/CLARABEL).
reproduce-full: prereg test sdp micro realistic montecarlo report manifest

# Chaîne sans SDP, pour les environnements où cvxpy n'est pas installable.
# Les tests SDP sont automatiquement sautés (pytest.importorskip).
reproduce-core: prereg test micro realistic montecarlo report manifest

clean:
	rm -rf .pytest_cache __pycache__ scripts/__pycache__ tests/__pycache__ monte_carlo_outputs_control
