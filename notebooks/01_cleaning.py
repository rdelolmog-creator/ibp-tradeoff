# notebooks/01_cleaning.py
# Paste as ONE Colab cell. Logic lives in src/ — this cell only orchestrates.

import sys, os
import pandas as pd

sys.path.insert(0, "/content/ibp-tradeoff")
from src.ingest import DataIngestor
from src.cleaner import DataCleaner

REPO = "/content/ibp-tradeoff"
results = {}

for data_root in ["data_primary", "data_control"]:
    print("=" * 78)
    print(f"DATASET: {data_root}")
    print("=" * 78)

    ingestor = DataIngestor(repo_root=REPO, data_root=data_root)
    raw = ingestor.load()

    cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
    df_clean, sku_master_clean, dq_report = cleaner.clean(raw)

    # Before / after reconciliation
    rows_in = sum(v for k, v in ingestor.rows_in_by_source.items() if k != "sku_master")
    print("\nRECONCILIATION")
    for source, n in ingestor.rows_in_by_source.items():
        print(f"  rows in  · {source:<12} {n:>6,}")
    print(f"  rows in  · transactional total   {rows_in:>6,}")
    for d in cleaner.dropped:
        print(f"  dropped  · {d['reason']:<45} {d['rows']:>3}")
    print(f"  rows out · clean_master          {len(df_clean):>6,}")

    print("\nDATA QUALITY REPORT")
    print(dq_report.to_string(index=False))

    print("\ndf_clean.head()")
    print(df_clean.head().to_string())
    print("\ndf_clean.dtypes")
    print(df_clean.dtypes.to_string())

    out_dir = os.path.join(REPO, data_root, "clean")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clean_master.parquet")
    df_clean.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")

    results[data_root] = (df_clean, sku_master_clean, dq_report)

# The reusability demonstration: identical code, two datasets, zero edits.
print("\n" + "=" * 78)
print("REUSABILITY CHECK — same DataCleaner, two datasets, no code change")
for tag, (df, _, dq) in results.items():
    d = dict(zip(dq.metric, dq.value))
    print(
        f"  {tag:<14} rows={int(d['rows_out']):,}  skus={int(d['skus_out'])}  "
        f"nulls={int(d['nulls_in_volume'])}  recovered={int(d['nulls_recovered_by_identity'])}  "
        f"rollforward_max={d['stock_rollforward_max_abs_diff']:.6f}"
    )

df_clean = results["data_primary"][0]   # primary is the working dataset downstream
