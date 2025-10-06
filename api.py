# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import polars as pl
import pandas as pd
import io
import numpy as np
from datetime import datetime

app = FastAPI()

@app.post("/analyze")
async def analyze_je(file: UploadFile = File(...)):
    try:
        # Read file
        content = await file.read()
        if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df_full = pl.from_pandas(pd.read_excel(io.BytesIO(content)))
        else:
            df_full = pl.read_csv(io.BytesIO(content), encoding="utf-8", infer_schema_length=10000)

        if len(df_full) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # --- STEP 1: Calculate Net Amount ---
        # Try to auto-detect Debit/Credit or single Amount
        debit_candidates = [c for c in df_full.columns if "debit" in c.lower()]
        credit_candidates = [c for c in df_full.columns if "credit" in c.lower()]
        amount_candidates = [c for c in df_full.columns if "amount" in c.lower() or "amt" in c.lower()]

        amount_series = None

        # Case 1: Debit & Credit
        if debit_candidates and credit_candidates:
            dr_col = debit_candidates[0]
            cr_col = credit_candidates[0]
            dr = (
                df_full[dr_col]
                .cast(pl.Utf8)
                .str.replace(",", "")
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
            )
            cr = (
                df_full[cr_col]
                .cast(pl.Utf8)
                .str.replace(",", "")
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
            )
            amount_series = dr - cr  # Positive = Debit, Negative = Credit
        # Case 2: Single Amount
        elif amount_candidates:
            amt_col = amount_candidates[0]
            amount_series = df_full[amt_col].cast(pl.Float64, strict=False)
        # Case 3: None found
        else:
            raise HTTPException(status_code=400, detail="No amount column found (looked for 'amount', 'debit', 'credit')")

        # --- STEP 2: Choose Primary Date ---
        date_col = None
        date_candidates = [c for c in df_full.columns if "date" in c.lower() or "effective" in c.lower() or "posted" in c.lower()]
        if date_candidates:
            date_col = date_candidates[0]

        # --- Build Final DataFrame ---
        select_list = [amount_series.alias("amount")]
        if date_col:
            select_list.append(pl.col(date_col).alias("posting_date"))

        # Add key fields if present
        for col, alias in [
            ("account", "account"),
            ("description", "description"),
            ("je_id", "je_id"),
            ("created_by", "created_by"),
            ("posted_by", "posted_by"),
            ("cost_center", "cost_center"),
            ("project", "project")
        ]:
            candidates = [c for c in df_full.columns if col in c.lower()]
            if candidates:
                select_list.append(pl.col(candidates[0]).alias(alias))

        selected = df_full.select(select_list)

        # --- Smart Date Parsing ---
        if "posting_date" in selected.columns:
            date_formats = [
                "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%m/%d/%Y", "%d/%m/%Y",
                "%Y%m%d", "%B %d, %Y", "%b %d, %Y", "%d-%m-%Y", "%m-%d-%Y"
            ]
            parsed = None
            for fmt in date_formats:
                try:
                    parsed = pl.col("posting_date").str.strptime(pl.Datetime, fmt, strict=False)
                    selected = selected.with_columns(parsed)
                    break
                except:
                    continue
            if parsed is None:
                selected = selected.drop(["posting_date"]) if "posting_date" in selected.columns else selected

        # --- Clean Invalid Rows ---
        cleaned = selected.filter(pl.col("amount").is_not_null())
        if "posting_date" in cleaned.columns:
            cleaned = cleaned.filter(pl.col("posting_date").is_not_null())

        # --- ANOMALY DETECTION ---
        anomaly_exprs = []

        # 1. Round Large Amount
        anomaly_exprs.append(
            ((pl.col("amount").abs() % 1 == 0) & (pl.col("amount").abs() > 1000)).alias("Round Large Amount")
        )

        # 2. Near-Zero Amount
        anomaly_exprs.append(
            (pl.col("amount").abs() < 1).alias("Near-Zero Amount")
        )

        # 3. Suspicious Description
        if "description" in cleaned.columns:
            anomaly_exprs.append(
                pl.col("description").str.to_lowercase().is_in([
                    "adjust", "misc", "manual", "override", "error", "temp",
                    "reversal", "correction", "clearing", "suspense", "miscellaneous"
                ]).alias("Suspicious Description")
            )
        else:
            anomaly_exprs.append(pl.lit(False).alias("Suspicious Description"))

        # 4. Weekend Entry
        if "posting_date" in cleaned.columns:
            anomaly_exprs.append(
                (pl.col("posting_date").dt.weekday() >= 6).fill_null(False).alias("Weekend Entry")
            )
        else:
            anomaly_exprs.append(pl.lit(False).alias("Weekend Entry"))

        # 5. Duplicate Entry (all non-amount columns)
        group_keys = [col for col in cleaned.columns if col != "amount"]
        if len(group_keys) > 0:
            anomaly_exprs.append(
                pl.struct(group_keys).is_duplicated().alias("Duplicate Entry")
            )
        else:
            anomaly_exprs.append(pl.lit(False).alias("Duplicate Entry"))

        # 6. High-Value Entry
        if len(cleaned) > 10:
            threshold = cleaned.select(pl.col("amount").abs().quantile(0.99)).item()
            anomaly_exprs.append(
                (pl.col("amount").abs() >= threshold).alias("High-Value Entry")
            )
        else:
            anomaly_exprs.append(pl.lit(False).alias("High-Value Entry"))

        # 7. Repeating Entry
        if "description" in cleaned.columns and "account" in cleaned.columns:
            anomaly_exprs.append(
                pl.struct(["account", "amount", "description"]).is_duplicated().alias("Repeating Entry")
            )
        else:
            anomaly_exprs.append(pl.lit(False).alias("Repeating Entry"))

        # Build results
        results = cleaned.with_columns(anomaly_exprs)
        anomaly_cols = [col for col in results.columns if col != "amount" and col != "Has Anomaly" and "Entry" in col]
        results = results.with_columns(pl.any_horizontal(anomaly_cols).alias("Has Anomaly"))
        anomalies = results.filter(pl.col("Has Anomaly"))

        # --- SUMMARY ---
        total_net = results["amount"].sum()
        abs_total = results["amount"].abs().sum()
        imbalance_pct = abs(total_net) / (abs_total + 1e-6) * 100

        # --- Return JSON ---
        return {
            "success": True,
            "summary": {
                "Total Entries": len(results),
                "Net Balance": round(float(total_net), 2),
                "Imbalance %": round(imbalance_pct, 2),
                "Anomalies Found": len(anomalies)
            },
            "charts": {
                "anomalies": {col: anomalies.filter(pl.col(col)).height for col in anomaly_cols}
            },
            "anomalies_data": anomalies.to_dicts()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/export")
async def export_report(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if file.filename.endswith(".xlsx"):
            df = pl.from_pandas(pd.read_excel(io.BytesIO(content)))
        else:
            df = pl.read_csv(io.BytesIO(content), encoding="utf-8")

        # Re-run logic to get results (in production, cache or reuse)
        # For brevity, we'll export sample sheets
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Sheet 1: All Data
            df.to_pandas().to_excel(writer, index=False, sheet_name="All Transactions")

            # Sheet 2: Summary
            pd.DataFrame([{
                "Metric": "Total Entries",
                "Value": len(df)
            }]).to_excel(writer, index=False, sheet_name="Summary")

            # Sheet 3: Sample Anomalies (mock)
            if "Amount" in df.columns:
                sample_anomalies = df.to_pandas().head(50)
                sample_anomalies["Anomaly"] = "Round Amount"
                pd.DataFrame(sample_anomalies).to_excel(writer, index=False, sheet_name="Anomalies")

        excel_buffer.seek(0)
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=je_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
        )
    except Exception as e:
        return {"success": False, "error": str(e)}