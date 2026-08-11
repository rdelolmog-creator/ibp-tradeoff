"""Step 4 human-review artefact.

Builds Step4_Data_Quality_Review.xlsx directly from the live objects produced
by DataIngestor.load() and DataCleaner.clean() in the SAME run that produced
clean_master.parquet. This must be called from inside the pipeline (the Colab
notebook), not reconstructed afterward from pasted output — otherwise the
review artefact is no longer evidence of what the pipeline actually did.

Usage (from notebooks/01_cleaning.py, after cleaner.clean(raw)):

    from src.report_step4 import build_review_workbook
    build_review_workbook(
        raw=raw, df_clean=df_clean, master=master, dq_report=dq_report,
        dropped=cleaner.dropped, out_path=f"{data_root}/clean/Step4_Data_Quality_Review.xlsx",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(name=FONT, size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name=FONT, size=16, bold=True, color="1F3864")
SUB_FONT = Font(name=FONT, size=10, italic=True, color="595959")
SECT_FONT = Font(name=FONT, size=12, bold=True, color="1F3864")
BODY_FONT = Font(name=FONT, size=10)
FLAG_FILL = PatternFill("solid", fgColor="FFF2CC")
OK_FILL = PatternFill("solid", fgColor="E2EFDA")
CLEAN_HDR_FILL = PatternFill("solid", fgColor="2E7D32")
APPROVE_FILL = PatternFill("solid", fgColor="FFF2CC")
_THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _style_header_row(ws, row: int, ncols: int, start_col: int = 1) -> None:
    for c in range(start_col, start_col + ncols):
        cell = ws.cell(row=row, column=c)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _autosize(ws, widths: List[float]) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _v(x: Any):
    """openpyxl-safe scalar: numpy types -> python, everything else -> str."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return str(x)


def build_review_workbook(
    raw: Dict[str, pd.DataFrame],
    df_clean: pd.DataFrame,
    master: pd.DataFrame,
    dq_report: pd.DataFrame,
    dropped: List[Dict[str, Any]],
    out_path: str | Path,
    dataset_label: str = "data_primary",
) -> Path:
    """Build the 6-tab review workbook from the actual objects a pipeline run
    just produced. Returns the path written."""

    dq = dq_report.set_index("metric")["value"]
    raw_a, raw_b = raw["system_a"], raw["system_b"]
    wb = Workbook()

    # ---- Sheet 1: Review & Approval -----------------------------------
    ws = wb.active
    ws.title = "1. Review & Approval"
    ws.sheet_view.showGridLines = False
    ws["B2"] = "IBP Trade-Off Engine — Step 4 Data Quality Review"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = "Ingestion + cleaning output, for human review before Step 5 proceeds"
    ws["B3"].font = SUB_FONT
    for r, (label, val) in enumerate(
        [
            ("Dataset", dataset_label),
            ("Pipeline", "src/ingest.py (DataIngestor) + src/cleaner.py (DataCleaner)"),
            ("Cleaning logic", "cleaning-spec.md, steps C-01 to C-13"),
        ],
        start=5,
    ):
        ws[f"B{r}"] = label
        ws[f"B{r}"].font = Font(name=FONT, bold=True, size=10)
        ws[f"C{r}"] = val
        ws[f"C{r}"].font = BODY_FONT

    ws["B9"] = "Summary of what changed"
    ws["B9"].font = SECT_FONT
    summary_rows = [
        ("Rows in (transactional, both systems)", int(len(raw_a) + len(raw_b))),
        ("Rows out (clean_master)", int(len(df_clean))),
        ("Rows dropped — see sheet 4 for reason", sum(d["rows"] for d in dropped)),
        ("Missing volumes found", int(dq["nulls_in_volume"])),
        ("Missing volumes recovered by roll-forward identity (exact, not estimated)", int(dq["nulls_recovered_by_identity"])),
        ("SKU code collisions resolved (see sheet 3)", int(dq["sku_collisions_resolved"])),
        ("Category spelling variants collapsed to 3", int(dq["category_variants_collapsed"])),
        ("Currency conversions applied (GBP to EUR)", int(dq["fx_applied_rows"])),
        ("Unit conversions applied (cases to units)", int(dq["uom_converted_rows"])),
        ("Stock roll-forward max error after cleaning (units)", round(float(dq["stock_rollforward_max_abs_diff"]), 6)),
    ]
    r0 = 10
    ws.cell(row=r0, column=2, value="Metric").font = HDR_FONT
    ws.cell(row=r0, column=3, value="Value").font = HDR_FONT
    ws.cell(row=r0, column=2).fill = HDR_FILL
    ws.cell(row=r0, column=3).fill = HDR_FILL
    for i, (k, v) in enumerate(summary_rows, start=r0 + 1):
        ws.cell(row=i, column=2, value=k).font = BODY_FONT
        ws.cell(row=i, column=3, value=v).font = BODY_FONT
        ws.cell(row=i, column=2).border = BORDER
        ws.cell(row=i, column=3).border = BORDER

    r_ap = r0 + len(summary_rows) + 3
    ws.cell(row=r_ap, column=2, value="Reviewer sign-off").font = SECT_FONT
    for offset, label in enumerate(["Reviewed by", "Date", "Decision"], start=1):
        ws.cell(row=r_ap + offset, column=2, value=label).font = Font(name=FONT, bold=True, size=10)
        ws.cell(row=r_ap + offset, column=3).fill = APPROVE_FILL
        ws.cell(row=r_ap + offset, column=3).border = BORDER
    dv = DataValidation(
        type="list",
        formula1='"Approved,Approved with comments,Rejected - see notes"',
        allow_blank=True,
    )
    ws.add_data_validation(dv)
    dv.add(ws.cell(row=r_ap + 3, column=3))
    ws.cell(row=r_ap + 4, column=2, value="Notes").font = Font(name=FONT, bold=True, size=10)
    ws.cell(row=r_ap + 4, column=3).fill = APPROVE_FILL
    ws.cell(row=r_ap + 4, column=3).border = BORDER
    ws.merge_cells(start_row=r_ap + 4, start_column=3, end_row=r_ap + 7, end_column=6)
    _autosize(ws, [3, 40, 45, 14, 14, 14, 14, 14])

    # ---- Sheet 2: Row-level trace --------------------------------------
    ws2 = wb.create_sheet("2. Row-Level Trace")
    ws2.sheet_view.showGridLines = False
    ws2["A1"] = "Raw-to-clean trace — sample SKUs, first 3 months"
    ws2["A1"].font = TITLE_FONT
    ws2["A2"] = "Every value below is from this pipeline run."
    ws2["A2"].font = SUB_FONT

    sample_a = df_clean[df_clean.sku_id.str.startswith("A-")].sku_id.iloc[0]
    sample_b = df_clean[df_clean.sku_id.str.startswith("B-")].sku_id.iloc[0]
    row_cursor = 4
    for sku_id, src_df, tag, code_col in [
        (sample_a, raw_a, "System A raw", "MATERIAL"),
        (sample_b, raw_b, "System B raw", "item_no"),
    ]:
        code_padded = sku_id.split("-")[1]
        code_bare = code_padded.lstrip("0") or "0"
        ws2.cell(row=row_cursor, column=1, value=f"SKU {sku_id} — {tag}").font = SECT_FONT
        row_cursor += 1
        match = src_df[src_df[code_col].astype(str).str.lstrip("0") == code_bare]
        sub = match.head(3)
        for j, col in enumerate(sub.columns, start=1):
            c = ws2.cell(row=row_cursor, column=j, value=str(col))
            c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        row_cursor += 1
        for _, rr in sub.iterrows():
            for j, col in enumerate(sub.columns, start=1):
                c = ws2.cell(row=row_cursor, column=j, value=_v(rr[col]))
                c.font, c.border = BODY_FONT, BORDER
            row_cursor += 1
        row_cursor += 1

        clean_sub = df_clean[df_clean.sku_id == sku_id].head(3)
        ws2.cell(row=row_cursor, column=1, value=f"SKU {sku_id} — after cleaning").font = SECT_FONT
        row_cursor += 1
        for j, col in enumerate(clean_sub.columns, start=1):
            c = ws2.cell(row=row_cursor, column=j, value=str(col))
            c.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
            c.fill, c.border = CLEAN_HDR_FILL, BORDER
        row_cursor += 1
        for _, rr in clean_sub.iterrows():
            for j, col in enumerate(clean_sub.columns, start=1):
                c = ws2.cell(row=row_cursor, column=j, value=_v(rr[col]))
                c.font, c.fill, c.border = BODY_FONT, OK_FILL, BORDER
            row_cursor += 1
        row_cursor += 2
    _autosize(ws2, [16] * 14)

    # ---- Sheet 3: Corrections applied -----------------------------------
    ws3 = wb.create_sheet("3. Corrections Applied")
    ws3.sheet_view.showGridLines = False
    ws3["A1"] = "Every correction type applied by src/cleaner.py"
    ws3["A1"].font = TITLE_FONT
    ws3["A2"] = "Maps to cleaning-spec.md steps C-03 to C-13"
    ws3["A2"].font = SUB_FONT
    corrections = [
        ("C-03", "SKU code collisions across systems",
         "Codes collide (e.g. '01' means a different SKU in System A vs System B)",
         "Built surrogate key {system}-{code}, e.g. A-01, B-01", int(dq["sku_collisions_resolved"])),
        ("C-04", "Two date formats",
         "System A: YYYYMM integer. System B: DD/MM/YYYY string, month-end.",
         "Parsed both explicitly (never inferred), snapped to month start", int(len(raw_a) + len(raw_b))),
        ("C-05", "Two rate scales",
         "System A: adherence/yield as 0-100. System B: same fields as 0-1.",
         "Rescaled System A by /100; validated both fall within [0,1]", int(len(raw_a) + len(raw_b))),
        ("C-06", "Two units of measure",
         "System A in each (EA). System B in cases (CS), case size varies by SKU.",
         "Converted System B volumes to units via case_size join", int(dq["uom_converted_rows"])),
        ("C-07", "Two currencies",
         "System A in EUR. System B in GBP, per case.",
         "Converted at assumptions.finance.fx.gbp_eur, undoing case size and FX in order", int(dq["fx_applied_rows"])),
        ("C-08", "Inconsistent category spelling",
         "Multiple casing/whitespace variants observed per category.",
         "Stripped, lowercased, collapsed; validated against assumptions.categories", int(dq["category_variants_collapsed"])),
        ("C-10", "Missing volumes",
         "SALES_QTY / shipped null on a subset of rows.",
         "Derived exactly from stock_open + production - stock_close; flagged actual_imputed=True", int(dq["nulls_recovered_by_identity"])),
        ("C-11", "Irrecoverable nulls",
         "First month of a SKU's history, no prior stock to derive from.",
         "Dropped and counted — not backfilled with an assumption", int(sum(d["rows"] for d in dropped))),
    ]
    for j, h in enumerate(["Spec ref", "Defect type", "What was found", "What was done", "Rows affected"], start=1):
        ws3.cell(row=4, column=j, value=h)
    _style_header_row(ws3, 4, 5)
    for i, (ref, defect, found, fix, n) in enumerate(corrections, start=5):
        ws3.cell(row=i, column=1, value=ref).font = BODY_FONT
        ws3.cell(row=i, column=2, value=defect).font = Font(name=FONT, bold=True, size=10)
        ws3.cell(row=i, column=3, value=found).font = BODY_FONT
        ws3.cell(row=i, column=4, value=fix).font = BODY_FONT
        ws3.cell(row=i, column=5, value=n).font = BODY_FONT
        for j in range(1, 6):
            ws3.cell(row=i, column=j).border = BORDER
            ws3.cell(row=i, column=j).alignment = Alignment(wrap_text=True, vertical="top")
        ws3.row_dimensions[i].height = 45
    _autosize(ws3, [9, 26, 45, 50, 12])

    # ---- Sheet 4: Rows dropped ------------------------------------------
    ws4 = wb.create_sheet("4. Rows Dropped")
    ws4.sheet_view.showGridLines = False
    ws4["A1"] = "Rows dropped during cleaning — full list, no silent loss"
    ws4["A1"].font = TITLE_FONT
    ws4["A2"] = "Every dropped row is listed here with its reason."
    ws4["A2"].font = SUB_FONT
    for j, h in enumerate(["Reason", "Row count"], start=1):
        ws4.cell(row=4, column=j, value=h)
    _style_header_row(ws4, 4, 2)
    for i, d in enumerate(dropped, start=5):
        ws4.cell(row=i, column=1, value=d["reason"]).font = BODY_FONT
        ws4.cell(row=i, column=2, value=d["rows"]).font = BODY_FONT
        ws4.cell(row=i, column=1).fill = FLAG_FILL
        ws4.cell(row=i, column=2).fill = FLAG_FILL
        ws4.cell(row=i, column=1).border = BORDER
        ws4.cell(row=i, column=2).border = BORDER
    _autosize(ws4, [55, 14])

    # ---- Sheet 5: Full DQ report ------------------------------------------
    ws5 = wb.create_sheet("5. Full DQ Report")
    ws5.sheet_view.showGridLines = False
    ws5["A1"] = "Complete data quality report — all metrics"
    ws5["A1"].font = TITLE_FONT
    ws5["A2"] = "null_rate__stock_open_units > 0 is expected (every SKU's first month has no prior stock)."
    ws5["A2"].font = SUB_FONT
    for j, h in enumerate(["Metric", "Value"], start=1):
        ws5.cell(row=4, column=j, value=h)
    _style_header_row(ws5, 4, 2)
    for i, rr in enumerate(dq_report.itertuples(index=False), start=5):
        ws5.cell(row=i, column=1, value=rr.metric).font = BODY_FONT
        val = float(rr.value)
        ws5.cell(row=i, column=2, value=val).font = BODY_FONT
        ws5.cell(row=i, column=1).border = BORDER
        ws5.cell(row=i, column=2).border = BORDER
        if str(rr.metric).startswith("null_rate__") and val > 0.05:
            ws5.cell(row=i, column=2).fill = FLAG_FILL
    _autosize(ws5, [40, 16])

    # ---- Sheet 6: Known limitations ---------------------------------------
    ws6 = wb.create_sheet("6. Known Limitations")
    ws6.sheet_view.showGridLines = False
    ws6["A1"] = "Known limitations of this cleaning pass"
    ws6["A1"].font = TITLE_FONT
    ws6["A2"] = "Read before approving."
    ws6["A2"].font = SUB_FONT
    limitations = [
        ("Censored actuals", "actual_units reflects what shipped, not what was demanded. Step 5 must handle this separately when estimating bias."),
        ("Zero-variance features", "sched_adherence and yield_rate are constant within source/category in this run — limits what Step 7's cost model can learn."),
        ("Tolerance is generator-scale", "The 0.5-unit roll-forward tolerance absorbs float rounding from the generator's own case-size division; not a real-world precision claim."),
        ("Single historical window only", "Built and tested against ONE 36-month load. Rolling/incremental re-runs are not yet defined — see O-08 in the decision log."),
        ("New/unseen defect types", "The pipeline fails loudly on any column, format, or value it wasn't built to expect. It does not silently corrupt data, but it also does not automatically handle a defect type it has never seen."),
    ]
    for i, (title, body) in enumerate(limitations, start=1):
        r = 4 + i * 2
        ws6.cell(row=r, column=1, value=title).font = Font(name=FONT, bold=True, size=11, color="1F3864")
        ws6.cell(row=r + 1, column=1, value=body).font = BODY_FONT
        ws6.cell(row=r + 1, column=1).alignment = Alignment(wrap_text=True)
        ws6.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=8)
        ws6.row_dimensions[r + 1].height = 40
    _autosize(ws6, [14] * 8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path
