# DiagHint

## 1. Generate the experience pool

Use Python 3.11 or later and PostgreSQL 16 with `pg_hint_plan` installed.

Install DiagHint from the repository directory:

```sh
python -m pip install -e .
```

Fill in the database connection, data revision, and API key in `config/diaghint.conf`.
The default LLM configuration uses the DeepSeek API with temperature 0.

Place the training SQL files in one directory. Name each file using its template number and instance suffix, such as `1a.sql` and `1b.sql`.

Generate the Stage-I and Stage-II experience pool and training records:

```sh
diaghint generate-pool \
  --workload /path/to/training-sql \
  --output ../diaghint-artifacts/pool
```

## 2. Train the cardinality-bias classifier

Train the classifier using the plans collected in the experience pool:

```sh
diaghint train-cardinality \
  --pool ../diaghint-artifacts/pool \
  --output ../diaghint-artifacts/cardinality
```

## 3. Train the utility model

Train the utility model using the experience pool and the cardinality-bias classifier:

```sh
diaghint train-utility \
  --pool ../diaghint-artifacts/pool \
  --cardinality-model ../diaghint-artifacts/cardinality \
  --output ../diaghint-artifacts/utility
```

## 4. Optimize a query

Save the query in a SQL file and run:

```sh
diaghint optimize \
  --sql /path/to/query.sql \
  --pool ../diaghint-artifacts/pool \
  --cardinality-model ../diaghint-artifacts/cardinality \
  --utility-model ../diaghint-artifacts/utility \
  --output ../diaghint-artifacts/query-result
```

Read `../diaghint-artifacts/query-result/result.json` for the selected hint, execution time, planning time, and end-to-end time.
