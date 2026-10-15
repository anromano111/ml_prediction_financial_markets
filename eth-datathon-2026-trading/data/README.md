# Data

The competition data (ETH Datathon 2026, HRT challenge) is **not included** in this repository. Download it from the competition page and place these files directly in this folder:

| File | Content |
|---|---|
| `bars_seen_train.parquet` | first 50 OHLC bars of 1,000 training sessions |
| `bars_unseen_train.parquet` | next 50 bars of the same sessions (used to build the target) |
| `bars_seen_public_test.parquet` / `bars_seen_private_test.parquet` | first 50 bars of 10,000 + 10,000 test sessions |
| `headlines_seen_train.parquet` | news headlines published during the seen bars (train) |
| `headlines_seen_public_test.parquet` / `headlines_seen_private_test.parquet` | headlines for the test sessions |

Columns: bars have `session, bar_ix, open, high, low, close`; headlines have `session, bar_ix, headline`.
