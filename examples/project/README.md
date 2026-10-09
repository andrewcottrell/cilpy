# Project scripts

Run everything from the repository root, after `pip install -e .`
The baseline and significance scripts also need `pymoo` and `scipy`.

## Experiments

| What | Command |
|---|---|
| Check the install | `python examples/project/check_install.py` |
| Everything, in order | `bash examples/project/run_all_overnight.sh` |
| Static campaign | `python examples/project/run_static_mo_experiments.py` |
| Dynamic campaign | `python examples/project/run_dynamic_experiments.py --tau-t 10` |
| Archive refresh comparison | `python examples/project/run_refresh_mode_experiments.py --tau-t 10` |
| NSGA-II baseline (static) | `python examples/project/run_nsga2_baseline.py` |
| DNSGA-II baseline (dynamic) | `python examples/project/run_dnsga2_baseline.py --tau-t 10` |
| Supplementary runs | `bash examples/project/run_supplementary.sh` |
| Single-objective validation | `cd examples/project/phase1 && python ccpso_benchmark_exa.py` |

Add `--quick` to a campaign script for a short smoke test. It overwrites the
tables in `results/`.

## Analysis

| What | Command |
|---|---|
| Dynamic and refresh tables | `python examples/project/aggregate_campaign.py --out-dir examples/project/data/out_refresh_taut10 --tau-t 10` |
| Static table, without re-running | `python examples/project/run_static_mo_experiments.py --aggregate-only --out-dir examples/project/data/out_static` |
| Comparison with DNSGA-II | `python examples/project/compare_baselines.py --dynamic-only --ccpso-out-dir "examples/project/data/out_refresh_taut{tau}"` |
| Comparison with NSGA-II | `python examples/project/compare_baselines.py --static-only --ccpso-out-dir examples/project/data/out_static` |
| Significance tests | `python examples/project/analyse_dynamic_significance.py --out-dir examples/project/data/out_refresh_taut10 --tau-t 10` |
| Sanity checks on the static results | `python examples/project/validate_results.py --out-dir examples/project/data/out_static` |
| Single-objective summary tables | `python examples/project/phase1/analysis.py` |
| Figures (needs `raw/`) | `python examples/project/generate_all_figures.py` |
| DTNK4 feasible region figure | `python examples/project/plot_dtnk4_decision_space.py` |
| Strip fronts from `raw/` into `data/` | `python examples/project/slim_campaign.py` |

## Folders

| Folder | Contents |
|---|---|
| `results/` | Result tables |
| `figures/` | Figures |
| `data/` | Campaign output without the per-iteration fronts |
| `raw/` | Full campaign output (not in git) |
| `phase1/` | Single-objective validation: scripts, output and tables |
