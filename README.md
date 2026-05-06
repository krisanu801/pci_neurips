# PCI NeurIPS Experiments (CSV-first)

This directory contains a clean, self-contained experiment + plotting pipeline for the
**Posterior Covariance Invariant** paper.

Design goals:
- Every experiment run writes **canonical CSV tables** under `runs/<run_id>/tables/`.
- Every figure script reads **only CSV** and produces **PDF + PNG** under `runs/<run_id>/figures/`.
- Structured `events.jsonl` logging for debugging + provenance via `manifest.json`.

## Quickstart (Exp.1 synthetic – produces Figures 2–6)

```bash
cd /Users/krisanusarkar/Documents/ML/neurips/pci_neurips
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m pci.experiments.exp1_synthetic --config configs/exp1.yaml
# After it finishes, it prints the run directory.
python -m pci.plots.make_all --run_dir runs/<run_id>
```

## Run Exp.2–Exp.4 (pipeline + CSV + plots)

```bash
# Exp.2 (currently uses an analytic oracle adapter + random Gaussian data to validate pipeline)
python -m pci.experiments.exp2_pretrained_universality --config configs/exp2.yaml

# Exp.3 (swiss roll geometry)
python -m pci.experiments.exp3_geometry_dimension --config configs/exp3.yaml

# Exp.4 (simulated training dynamics; CSV-first)
python -m pci.experiments.exp4_training_dynamics --config configs/exp4.yaml
```

## Colab

Colab usually only needs:
```bash
pip install -r requirements.txt
pip install -e .
python -m pci.experiments.exp2_pretrained_universality --config configs/exp2.yaml
```

Exp.2–Exp.4 are scaffolded with strict CSV outputs and logging. Exp.2 requires you to
provide local paths + real adapters for pretrained checkpoints (no auto-download in this repo).
