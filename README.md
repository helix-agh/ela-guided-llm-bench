# ELA-Guided LLM Benchmark

Code accompanying the "Automatic Design of Optimization Test Problems with Large Language Models" paper. Given a set of target ELA feature values (extracted from BBOB or MA-BBOB problems), the methods evolve Python functions whose landscape characteristics match the targets.

## Requirements

- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/)

## Installation

```bash
poetry install
```

Create a `.env` file in the project root with the required API keys (e.g. `OPENROUTER_API_KEY`).

## Methods

| Method | Description | Source |
|--------|-------------|--------|
| **EoTF** | ELA-guided evolutionary algorithm with 5 mutation/crossover operators and population-based search | `ela_guided_llm_bench/eotf/` |
| **LLaMEA** | Population-based evolution with elitism and 4 mutation strategies | `ela_guided_llm_bench/llamea/` |
| **ZeroShot** | Single-prompt baseline with no evolutionary feedback | `ela_guided_llm_bench/naive/` |
| **GP Baseline** | Genetic programming via DEAP; tree-based symbolic regression with ELA-based fitness | `ela_guided_llm_bench/gp_baseline/` |

Prompts for each LLM-based method are defined in their respective `prompt.py` files.

The GP baseline implementation was inspired by [Vermetten et al. (2023)](https://arxiv.org/pdf/2305.15245) ([code](https://zenodo.org/records/7896138)).

## Running Experiments

### LLM-based methods (EoTF, LLaMEA, ZeroShot)

```bash
poetry run python3 -m ela_guided_llm_bench.experiments.main \
  --model google/gemini-2.0-flash-001 \
  --method eotf \
  --start-fid 1 --end-fid 24 \
  --dim 2 --iid 1 \
  --problem-class BBOB
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `google/gemini-2.0-flash-001` | OpenRouter model identifier |
| `--method` | `eotf` | One of `eotf`, `llamea`, `zero_shot` |
| `--start-fid` / `--end-fid` | 1 / 24 | BBOB function ID range |
| `--dim` | 2 | Problem dimensionality |
| `--iid` | 1 | Instance ID |
| `--problem-class` | `BBOB` | `BBOB` or `AFFINIC` |
| `--parallel-batch-size` | 1 | Concurrent function evaluations |

### GP baseline

```bash
poetry run python3 -m ela_guided_llm_bench.experiments.gp_baseline_main
```

### Precomputing ELA features

```bash
poetry run python3 -m ela_guided_llm_bench.experiments.calculate_ela_features
poetry run python3 -m ela_guided_llm_bench.experiments.calculate_ma_bbob_ela_features
```

Precomputed statistics are stored in `data/`.

## Reproducing Figures

| Figure(s) | Command |
|-----------|---------|
| Fig. 2, 3, 9 | `poetry run python3 -m ela_guided_llm_bench.visualizations.method_comparison` |
| Fig. 4 | `poetry run python3 -m ela_guided_llm_bench.visualizations.critical_distance_plots` |
| Fig. 5 | `poetry run python3 -m ela_guided_llm_bench.visualizations.contour_grid` |
| Fig. 6 | `poetry run python3 -m ela_guided_llm_bench.visualizations.ma_bbob_vs_bbob` |
| Fig. 7, 8 | `poetry run python3 -m ela_guided_llm_bench.visualizations.model_comparison` |

## Project Structure

```
ela_guided_llm_bench/
├── eotf/              # EoTF algorithm and prompts
├── llamea/            # LLaMEA algorithm and prompts
├── naive/             # ZeroShot baseline
├── gp_baseline/       # GP baseline (DEAP)
├── llm/               # LLM API clients (OpenRouter, Gemini)
├── experiments/       # Experiment entry points and scripts
├── visualizations/    # Figure generation and analysis
├── ela.py             # ELA feature extraction and normalization
├── function.py        # Function parsing and representation
├── experiment.py      # Experiment result management
└── experiment_config.py
data/                  # Precomputed ELA feature statistics (BBOB, MA-BBOB)
```

## Experimental Results

The repository includes directories with experimental results used to generate the visualizations in the paper. Each experiment subdirectory (e.g. `eotf_dim2_2_flash/eotf_google_gemini_2_0_flash_001_f1_iid1_dim2/`) contains a `generated_functions_info.csv` file with all generated functions and their computed ELA features, along with `target_ela_features.json` specifying the target ELA values.

| Directory | Description |
|-----------|-------------|
| `eotf_dim{d}_{model}` | EoTF results for dimension `d` using a given LLM (e.g. `eotf_dim2_2_flash` = dim 2, Gemini 2.0 Flash) |
| `eotf_dim2_2_flash_ma_bbob` | EoTF results on MA-BBOB targets (dim 2, Gemini 2.0 Flash) |
| `llamea_dim{d}` | LLaMEA results for dimension `d` |
| `zero_shot_dim{d}` | ZeroShot baseline results for dimension `d` |
| `gp_baseline_dim{d}` | GP baseline results for dimension `d` |
| `eotf_dim{d}_2_flash_algorithm_benchmark` | Algorithm benchmark results on EoTF-generated functions |
| `bbob_dim{d}_algorithm_benchmark` | Algorithm benchmark results on original BBOB functions |
