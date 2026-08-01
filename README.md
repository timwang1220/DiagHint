# DiagHint: Online Plan Repair with LLM-Driven Query Hinting


## Repository Layout

- `src/train_pool.py`: offline demo pool builder. It runs Stage A exploration for seed queries and Stage B trial generation, then writes `utility_trials.jsonl`.
- `src/plan_node/train_from_jsonl.py`: cardinality bias classification training from plan nodes.
- `src/utility-model/train.py`: utility model training for source-demo/target-query reuse decisions.
- `src/run_workload_online.py`: online workload runner.
- `src/generate_pool.py`, `src/postgresql.py`, `src/plan_summarizer.py`, `src/select_demonstration.py`: shared execution, plan summarization, and demo selection logic.
- `prompt/`: offline exploration and online LLM prompt templates.
- `config/`: local DB and LLM configuration templates. Fill these locally before running.

Generated data and model artifacts are expected under:

- `data/train-query/`: offline training SQL files, named like `1a.sql`, `1b.sql`.
- `data/test-query/`: online workload SQL files.
- `outputs/demo_pool/`: generated demonstration pool.
- `models/cardinality_bias/`: cardinality bias classifier artifacts.
- `models/utility/`: utility model checkpoint.
- `outputs/online_run/`: online workload results.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Edit `config/db.conf` and `config/llm.conf`:

- `db.conf` must point to the PostgreSQL instance that contains the benchmark database and `pg_hint_plan`.
- `llm.conf` uses an OpenAI-compatible endpoint. Do not commit real API keys.

If the sentence-transformer model is stored locally, set:

```bash
export DIAGHINT_TEXT_MODEL=/path/to/all-MiniLM-L6-v2
```

Otherwise the default model id is `sentence-transformers/all-MiniLM-L6-v2`.

## 1. Offline: Generate Demo Pool

This stage explores training queries and stores, for each query, the original plan, candidate hints, hinted plans, timing results, reasoning text, and plan JSON used by later models.

```bash
PYTHONPATH=src python3 src/train_pool.py \
  --train_query_dir data/train-query \
  --target_pool outputs/demo_pool \
  --system_prompt prompt/online-system.prompt \
  --user_prompt prompt/online-user.prompt \
  --random_rounds 6 \
  --max_total_rounds 8 \
  --a_max_iterations 7
```

Important outputs:

- `outputs/demo_pool/<query_id>/query.sql`
- `outputs/demo_pool/<query_id>/0/plan.json`
- `outputs/demo_pool/<query_id>/<round>/plan.json`
- `outputs/demo_pool/<query_id>/suggest_hint.txt`
- `outputs/demo_pool/utility_trials.jsonl`

## 2. Offline: Train Cardinality Bias Classification Model

This stage extracts join/scan nodes from `plan.json`, computes q-error buckets from estimated vs. actual rows, and trains the node-level classifier/regressor used to mark cardinality bias in plans.

```bash
PYTHONPATH=src python3 src/plan_node/train_from_jsonl.py \
  --plan_dir outputs/demo_pool \
  --output_dir models/cardinality_bias \
  --dump_extracted_jsonl outputs/cardinality_nodes.jsonl \
  --epochs 300 \
  --batch_size 32 \
  --device cuda
```

For CPU-only runs, use `--device cpu`.

Main outputs:

- `models/cardinality_bias/best_model.pt`
- `models/cardinality_bias/config.json`
- `models/cardinality_bias/norm_stats.npy`

## 3. Offline: Train Utility Model

This stage trains the source-target plan scoring model using `utility_trials.jsonl`. The model learns whether a demo should be reused directly, used as in-context evidence, or skipped.

```bash
PYTHONPATH=src python3 src/utility-model/train.py \
  --train_jsonl outputs/demo_pool/utility_trials.jsonl \
  --artifacts_dir models/cardinality_bias \
  --predicate_fit_dir outputs/demo_pool \
  --out_dir models/utility \
  --encoder_mode current \
  --epochs 40
```

Main output:

- `models/utility/best.pt`

If using the BAO-hybrid encoder:

```bash
PYTHONPATH=src python3 src/utility-model/train.py \
  --train_jsonl outputs/demo_pool/utility_trials.jsonl \
  --artifacts_dir models/cardinality_bias \
  --predicate_fit_dir outputs/demo_pool \
  --out_dir models/utility \
  --encoder_mode bao_hybrid \
  --encoder_artifacts_dir models/utility/encoder_artifacts.json
```

## 4. Online: Run All Queries in a Directory

The online runner processes every `*.sql` under `--query_dir`:

1. runs plain `EXPLAIN (FORMAT JSON)` to get the baseline execution plan without executing the original query;
2. marks plan nodes with the trained cardinality bias model/artifacts;
3. scores candidate demos with the trained utility model;
4. reuses a same-template positive demo when appropriate, otherwise builds an LLM prompt with selected demos;
5. converts the LLM JSON actions to `pg_hint_plan` hints and executes the final query.

```bash
PYTHONPATH=src python3 src/run_workload_online.py \
  --query_dir data/test-query \
  --demo_pool_dir outputs/demo_pool \
  --output_dir outputs/online_run \
  --train_query_path data/train-query \
  --utility_ckpt_path models/utility/best.pt \
  --utility_artifacts_dir models/cardinality_bias \
  --llm_concurrency 1 \
  --dump_debug_files
```

Main outputs:

- `outputs/online_run/<query_file>/original_execution_plan.txt`
- `outputs/online_run/<query_file>/suggested_hint.txt`
- `outputs/online_run/<query_file>/execution_plan.txt`
- `outputs/online_run/<query_file>/reason.txt`
- `outputs/online_run/<query_file>/demo_selection.json`
- `outputs/online_run/execution_times.csv`

## Notes

- Offline pool generation/training uses executed plans with actual rows. Online planning uses plain `EXPLAIN`; only the final selected hinted SQL is executed.
- Generated pools, checkpoints, online outputs, and local query/data directories may contain sensitive SQL or timing information. Keep them out of source control.
- `config/*.conf` in this repository are templates. Replace `CHANGE_ME` locally.
