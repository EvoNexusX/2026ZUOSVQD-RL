# svqd-rl

The training entry point is `main.py`. Default settings are defined in `SparseNetworkOptimizationConfig` in `t.py`. Training collects experience directly from Gymnasium / MuJoCo environments; no dataset download is required.

## 1. Environment setup

The commands below use Python 3.11. Dependencies are pinned in `requirements.txt`: NumPy 1.26.4, PyTorch 2.4.0, SciPy 1.15.3, Gymnasium 1.2.3, and MuJoCo 3.8.0.

Run all commands from the directory containing `main.py` and `requirements.txt`.

### Windows / PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, confirm that `python --version` reports Python 3.11, then create the environment with `python -m venv .venv`. The commands below call the virtual environment's interpreter directly, so activation is unnecessary.

### Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The program automatically uses a GPU when PyTorch detects CUDA, and otherwise uses the CPU. Training does not open a rendering window by default.

## 2. Start training

From the same directory, choose a command in PowerShell:

```powershell
# Default run: Hopper-v5, seed 1, 50 outer iterations
.\.venv\Scripts\python.exe main.py

# Explicit environment, seed, and iteration count
.\.venv\Scripts\python.exe main.py --env_name Hopper-v5 --seed 1 --M 50

# Train in another environment
.\.venv\Scripts\python.exe main.py --env_name HalfCheetah-v5 --seed 2 --M 50

# List all command-line options
.\.venv\Scripts\python.exe main.py --help
```

On Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`:

```bash
.venv/bin/python main.py --env_name Hopper-v5 --seed 1 --M 50
```

Supported environments are `Hopper-v5`, `HalfCheetah-v5`, `Walker2d-v5`, `Ant-v5`, and `Humanoid-v5`. Use the full name, including the `-v5` suffix.

Each outer iteration performs ES search, mask search, archive updates, and TD3 refinement of policies selected through clustering. The default configuration evaluates many candidates. `--M` controls outer iterations, not environment steps.

## 3. Common options

| Option | Default | Description |
| --- | --- | --- |
| `--env_name` | `Hopper-v5` | Training environment |
| `--seed` | `1` | Random seed |
| `--M` | `50` | Number of outer iterations; use a positive integer |
| `--max_env_steps` | `0` | Environment interaction budget; `0` disables this limit |
| `--hidden_layer_size` | `256` | Width of each of the two hidden layers |
| `--es_pop_size` / `--es_max_generations` | `10` / `10` | ES population size / generations per outer iteration; population size must be a positive even integer |
| `--mask_pop_size` / `--mask_generations` | `100` / `10` | Mask population size / generations per outer iteration |
| `--max_episode_steps` | `1000` | Maximum environment steps per episode |

`--max_env_steps` includes both training and evaluation interactions. The budget is checked between phases and selected calls, so an active phase may exceed the limit before stopping.

Use `--help` for additional options, including learning rates and replay capacities. Settings without a command-line option can be edited in `t.py`. Command-line values override configuration defaults.

For an ablation run, append one ablation flag, for example:

```powershell
.\.venv\Scripts\python.exe main.py --env_name Hopper-v5 --seed 1 --M 50 --ablation_no_mask
```

Available flags: `--ablation_no_mask`, `--ablation_no_td3`, `--ablation_no_nbc`, `--ablation_shared_critic`, `--ablation_no_memory`, `--ablation_no_es`, and `--ablation_plain_actor_update`.

## 4. Results

Outputs are saved under `results/<environment>_gait_<timestamp>/` in the current working directory. The program prints `Results saved in ...` when training finishes.

```text
results/Hopper-v5_gait_YYYYMMDD_HHMMSS/
├── init_info.json          # Environment, descriptor, and initialization metadata
├── training_log.json       # Configuration, per-iteration QD metrics, and refinement records
├── metrics.csv             # QD-score, coverage, returns, interaction counts, and timing
├── log.txt                 # Tabular training metrics
└── archive/
    ├── manifest.json       # Snapshot index
    ├── iter_0000_post_train/
    └── iter_0050_final/    # Final snapshot after the default 50 iterations
```

Each snapshot contains `individual_XXXX/` directories with `actor.pth` and `info.json`, plus critic weights and `masks.npz` when available. These snapshots store policies and measurements. The current entry point does not support resuming training; snapshots omit replay buffers, optimizer state, and random-number generator state.

Set `RESULTS_ROOT` to choose another output root. For multiple seeds, use a separate directory for each seed:

```powershell
foreach ($seed in 1..5) {
    $env:RESULTS_ROOT = Join-Path $PWD "results\seed_$seed"
    .\.venv\Scripts\python.exe main.py --env_name Hopper-v5 --seed $seed --M 50
}
Remove-Item Env:RESULTS_ROOT
```

On Linux, set the output root for a single run with:

```bash
RESULTS_ROOT=./results/seed_1 .venv/bin/python main.py --env_name Hopper-v5 --seed 1 --M 50
```

## 5. Main files

- `main.py`: training entry point and outer loop.
- `t.py`: default configuration and coordination of ES and mask search.
- `mask_phase.py`, `ES.py`, `es_phase.py`: structural and parameter search.
- `networks.py`, `memory.py`, `td3_trainer.py`: networks, replay buffers, and TD3 updates.
- `nbc.py`, `retraining_scheduler.py`: clustering and refinement scheduling.
- `behavior_descriptor.py`, `qd_metrics.py`, `archive_saver.py`: behavior descriptors, metrics, and archive exports.
