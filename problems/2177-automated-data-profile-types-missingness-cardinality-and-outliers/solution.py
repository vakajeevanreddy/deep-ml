from typing import Any
import numpy as np
import pandas as pd


def _to_python_scalar(val: Any) -> Any:
    if hasattr(val, "item"):
        try:
            return val.item()
        except (ValueError, TypeError):
            pass
    return val


def profile_dataframe(df: pd.DataFrame) -> dict:
    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    columns_profile: dict[Any, dict[str, Any]] = {}

    for col in df.columns:
        s = df[col]
        non_missing_mask = s.notna()
        non_missing_count = int(non_missing_mask.sum())
        missing_count = n_rows - non_missing_count

        if pd.api.types.is_bool_dtype(s.dtype):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s.dtype):
            kind = "numeric"
        else:
            kind = "categorical"

        missing_pct = round((missing_count / n_rows) * 100.0, 2) if n_rows > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        profile: dict[str, Any] = {
            "kind": kind,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
        }
        n_outliers = 0

        if kind == "numeric":
            if non_missing_count == 0:
                profile["mean"] = None
                profile["std"] = None
                profile["min"] = None
                profile["median"] = None
                profile["max"] = None
                profile["n_outliers"] = 0
            else:
                num = pd.to_numeric(s[non_missing_mask], errors="coerce").dropna()
                if len(num) == 0:
                    profile["mean"] = None
                    profile["std"] = None
                    profile["min"] = None
                    profile["median"] = None
                    profile["max"] = None
                    profile["n_outliers"] = 0
                else:
                    profile["mean"] = round(float(num.mean()), 4)
                    if len(num) >= 2:
                        std_val = num.std(ddof=1)
                        profile["std"] = None if pd.isna(std_val) else round(float(std_val), 4)
                    else:
                        profile["std"] = None
                    profile["min"] = round(float(num.min()), 4)
                    profile["median"] = round(float(num.median()), 4)
                    profile["max"] = round(float(num.max()), 4)

                    q1 = float(num.quantile(0.25, interpolation="linear"))
                    q3 = float(num.quantile(0.75, interpolation="linear"))
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    n_outliers = int(((num < lower_bound) | (num > upper_bound)).sum())
                    profile["n_outliers"] = n_outliers
        else:
            if non_missing_count == 0:
                profile["top"] = None
                profile["top_freq"] = 0
            else:
                non_missing_values = s[non_missing_mask]
                counts: dict[Any, int] = {}
                first_seen: dict[Any, int] = {}
                for idx, raw_val in enumerate(non_missing_values):
                    py = _to_python_scalar(raw_val)
                    if py not in counts:
                        counts[py] = 0
                        first_seen[py] = idx
                    counts[py] += 1
                top_val = min(counts.keys(), key=lambda k: (-counts[k], first_seen[k]))
                profile["top"] = top_val
                profile["top_freq"] = int(counts[top_val])

        flags: list[str] = []
        if n_unique <= 1:
            flags.append("constant")
        if missing_pct > 50:
            flags.append("high_missing")
        if (
            kind == "categorical"
            and non_missing_count >= 5
            and n_unique > 0.9 * non_missing_count
        ):
            flags.append("high_cardinality")
        if kind == "numeric" and n_outliers > 0:
            flags.append("outliers")

        profile["flags"] = flags
        columns_profile[col] = profile

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "columns": columns_profile,
    }