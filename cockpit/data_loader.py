"""Load CSV/Excel input and run deterministic validation checks (spec 5.1 and 9.1)."""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import pandas as pd

from .schemas import BINARY_SIGNALS, CONTINUOUS_SIGNALS, RATIO_SIGNALS, REQUIRED_COLUMNS, SIGNAL_COLUMNS


@dataclass
class ValidationReport:
    rows_loaded: int = 0
    required_fields_ok: bool = True
    missing_columns: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)
    missing_values: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    duplicate_case_ids: list[str] = field(default_factory=list)
    duplicate_claim_numbers: dict[str, list[str]] = field(default_factory=dict)
    invalid_values: list[str] = field(default_factory=list)
    signals_found: int = 0
    blocking_errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.blocking_errors:
            return "blocked"
        if self.duplicate_claim_numbers or self.invalid_values or self.missing_values:
            return "ready with warnings"
        return "ready"

    def case_warnings(self, case_id: str, claim_number: str) -> list[str]:
        warnings = []
        others = [c for c in self.duplicate_claim_numbers.get(claim_number, []) if c != case_id]
        if others:
            warnings.append(
                f"Claim number {claim_number} is also used by {', '.join(others)}. "
                "Verify the record before relying on it. This is a data-quality issue, not evidence of fraud."
            )
        for msg in self.invalid_values:
            if msg.startswith(f"{case_id}:"):
                warnings.append(msg.split(":", 1)[1].strip())
        return warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


import os as _os
# Import limits (env-configurable). Large files are better loaded with `python -m cockpit.batch --file ...`.
MAX_ROWS = int(_os.environ.get("MAX_IMPORT_ROWS", "2000000"))
MAX_BYTES = int(_os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024
TEXT_COLUMNS = {"case_id": str, "claim_number": str, "care_type": str, "state": str}
COUNT_COLUMNS = ["weekly_visit_frequency", "member_provider_distance_miles", "prior_claims_last_12mo", "claim_amount_usd"]


class InputError(ValueError):
    """The file could not be read as a claims table."""


def read_input(source: Union[str, Path, BinaryIO], filename: str | None = None) -> pd.DataFrame:
    name = (filename or getattr(source, "name", None) or str(source)).lower()
    if hasattr(source, "getbuffer") and source.getbuffer().nbytes > MAX_BYTES:
        raise InputError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(source, dtype=TEXT_COLUMNS)
        elif name.endswith((".csv", ".txt")) or not name.rsplit(".", 1)[-1:] or "." not in name:
            df = pd.read_csv(source, encoding="utf-8-sig", dtype=TEXT_COLUMNS)
        else:
            raise InputError("Use a .csv or .xlsx file.")
    except InputError:
        raise
    except pd.errors.EmptyDataError as e:
        raise InputError("The file is empty.") from e
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as e:
        raise InputError(f"The file could not be parsed: {e.__class__.__name__}.") from e
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def file_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Return a cleaned frame (typed) and a report. Never silently drops rows."""
    rep = ValidationReport(rows_loaded=len(df))
    if df.empty:
        rep.blocking_errors.append("The file has no case rows.")
        return df, rep
    if len(df) > MAX_ROWS:
        rep.blocking_errors.append(f"{len(df)} rows exceeds the prototype limit of {MAX_ROWS:,}.")
        return df, rep
    rep.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    rep.unknown_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    rep.signals_found = len([c for c in SIGNAL_COLUMNS if c in df.columns])
    if rep.missing_columns:
        rep.required_fields_ok = False
        rep.blocking_errors.append(f"Missing required columns: {', '.join(rep.missing_columns)}")
        return df, rep

    df = df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()
    blank = df["case_id"].isna() | (df["case_id"] == "")
    if blank.any():
        rep.blocking_errors.append(f"{int(blank.sum())} row(s) have a blank case_id.")
        return df, rep
    df["case_id"] = df["case_id"].astype(str)
    df["claim_number"] = df["claim_number"].fillna("").astype(str)

    na = df[REQUIRED_COLUMNS].isna().sum()
    rep.missing_by_column = {k: int(v) for k, v in na.items() if v}
    rep.missing_values = int(na.sum())

    dup_ids = df["case_id"][df["case_id"].duplicated(keep=False)]
    rep.duplicate_case_ids = sorted(set(dup_ids))
    if rep.duplicate_case_ids:
        rep.blocking_errors.append(f"Duplicate case IDs: {', '.join(rep.duplicate_case_ids)}")

    grouped = df.groupby("claim_number")["case_id"].apply(list)
    rep.duplicate_claim_numbers = {k: v for k, v in grouped.items() if len(v) > 1}

    parsed_dates = pd.to_datetime(df["claim_date"], errors="coerce")
    for cid, raw, parsed in zip(df["case_id"], df["claim_date"], parsed_dates):
        if pd.isna(parsed):
            rep.invalid_values.append(f"{cid}: claim_date '{raw}' could not be parsed.")
    df["claim_date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    numeric_cols = ["claim_amount_usd"] + SIGNAL_COLUMNS
    for col in numeric_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        for cid, raw, val in zip(df["case_id"], df[col], coerced):
            if pd.isna(val) and not pd.isna(raw):
                rep.invalid_values.append(f"{cid}: {col} value '{raw}' is not numeric.")
        df[col] = coerced

    for col in numeric_cols:
        for cid, v in zip(df["case_id"], df[col]):
            if pd.notna(v) and not np.isfinite(float(v)):
                rep.invalid_values.append(f"{cid}: {col} is not a finite number.")
        df[col] = df[col].where(np.isfinite(df[col].astype(float)) | df[col].isna(), other=np.nan)
    for col in COUNT_COLUMNS:
        for cid, v in zip(df["case_id"], df[col]):
            if pd.notna(v) and float(v) != int(float(v)):
                rep.invalid_values.append(f"{cid}: {col} is {v}, expected a whole number.")
    for cid, amt in zip(df["case_id"], df["claim_amount_usd"]):
        if pd.notna(amt) and amt <= 0:
            rep.invalid_values.append(f"{cid}: claim_amount_usd is {amt}, expected a positive amount.")
    for col in BINARY_SIGNALS:
        for cid, v in zip(df["case_id"], df[col]):
            if pd.notna(v) and v not in (0, 1):
                rep.invalid_values.append(f"{cid}: {col} is {v}, expected 0 or 1.")
    for col in RATIO_SIGNALS:
        for cid, v in zip(df["case_id"], df[col]):
            if pd.notna(v) and not 0 <= v <= 1:
                rep.invalid_values.append(f"{cid}: {col} is {v}, expected between 0 and 1.")
    for col in ["weekly_visit_frequency", "member_provider_distance_miles", "prior_claims_last_12mo"]:
        for cid, v in zip(df["case_id"], df[col]):
            if pd.notna(v) and v < 0:
                rep.invalid_values.append(f"{cid}: {col} is negative ({v}).")

    for col in BINARY_SIGNALS + ["claim_amount_usd", "weekly_visit_frequency",
                                  "member_provider_distance_miles", "prior_claims_last_12mo",
                                  "amount_vs_peer_avg_pct"]:
        df[col] = df[col].round().astype("Int64")
    return df, rep


def load_file(source, filename: str | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    try:
        df = read_input(source, filename)
    except InputError as e:
        rep = ValidationReport()
        rep.required_fields_ok = False
        rep.blocking_errors.append(str(e))
        return pd.DataFrame(), rep
    return validate(df)


def records(df: pd.DataFrame) -> list[dict]:
    """Plain-python dicts with native types (JSON-safe)."""
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out
