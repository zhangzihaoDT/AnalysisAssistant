"""
使用模式：
1) 无参维护模式（默认）
   python scripts/index_summary.py
   增量更新 schema/index_summary_daily_matrix_2024-01-01_to_yesterday.csv，
   默认回刷最近 7 天并覆盖合并（可用 --refresh-days 调整）到 yesterday。

2) 单日模式
   python scripts/index_summary.py --date 2026-03-10
   输出该日 JSON 指标结果。
   python scripts/index_summary.py --date yesterday
   python scripts/index_summary.py --date yesterday SPLW

3) 周期模式
   python scripts/index_summary.py --start 2026-03-01 --end 2026-03-10 [--csv-out ...] [--print-json] [--include-days]
   产出周期 daily_metrics_matrix，并写出矩阵 CSV。
"""

import argparse
import csv
import json
import re
from pathlib import Path
import glob

import pandas as pd


def _parse_target_date(value: str) -> pd.Timestamp:
    value = str(value).strip()
    lower = value.lower()
    if lower in {"yesterday", "昨日"}:
        return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).normalize()
    if lower in {"today", "今日"}:
        return pd.Timestamp.today().normalize()

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if m:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d).normalize()

    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
    if m:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d).normalize()

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无法解析日期: {value}")
    return pd.Timestamp(parsed).normalize()


def _parse_single_date_args(date_args: object) -> tuple[pd.Timestamp, str | None]:
    if isinstance(date_args, list):
        parts = [str(x).strip() for x in date_args if str(x).strip()]
    else:
        parts = [str(date_args).strip()] if date_args is not None else []
    if not parts:
        raise ValueError("缺少 --date")

    base_date = _parse_target_date(parts[0])
    modifiers = {p.strip().upper() for p in parts[1:]}
    if not modifiers:
        return (base_date, None)
    if modifiers == {"SPLW"}:
        return (base_date, "SPLW")
    raise ValueError(f"无法识别 --date 修饰符: {' '.join(sorted(modifiers))}")


def _read_data_paths(md_path: Path) -> dict[str, Path]:
    raw = md_path.read_text(encoding="utf-8").splitlines()
    out: dict[str, Path] = {}
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if "：" in line:
            name, path = line.split("：", 1)
        elif ":" in line:
            name, path = line.split(":", 1)
        else:
            continue
        name = name.strip()
        path = path.strip().replace("\\_", "_").replace("\\*", "*")
        expanded = glob.glob(path)
        if expanded:
            expanded = sorted(expanded, key=lambda p: (len(p), p))
            out[name] = Path(expanded[0])
        else:
            out[name] = Path(path)
    return out


def _parse_cn_ymd_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    parts = s.str.extract(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
    dt = pd.to_datetime(
        parts["y"] + "-" + parts["m"] + "-" + parts["d"], errors="coerce"
    ).dt.normalize()
    if dt.notna().any():
        return dt

    dt2 = pd.to_datetime(s, errors="coerce").dt.normalize()
    return dt2


def _read_metrics_csv(path: Path) -> pd.DataFrame:
    attempts: list[tuple[str, str]] = [
        ("utf-16", "\t"),
        ("utf-8-sig", ","),
        ("utf-8", ","),
        ("gb18030", ","),
        ("utf-16le", "\t"),
        ("utf-16be", "\t"),
        ("utf-16le", ","),
        ("utf-16be", ","),
    ]
    errors: list[str] = []
    for enc, sep in attempts:
        try:
            return pd.read_csv(path, encoding=enc, sep=sep)
        except Exception as e:
            errors.append(f"{enc}/{repr(sep)}: {e}")
    raise RuntimeError(
        f"无法读取 CSV: {path}。已尝试编码/分隔符组合: " + " | ".join(errors)
    )


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(",", "", regex=False).str.replace("，", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _safe_ratio(numer: float | int, denom: float | int) -> float | None:
    denom = float(denom)
    if denom == 0.0:
        return None
    return float(numer) / denom


def _safe_div(numer: float | int, denom: float | int) -> float | None:
    denom = float(denom)
    if denom == 0.0:
        return None
    return float(numer) / denom


def _to_percent_1dp(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{round(value * 100.0, 1):.1f}%"


def _to_interval_days(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return (series.dt.total_seconds() / 86400.0).astype("Float64")
    as_num = pd.to_numeric(series, errors="coerce")
    if as_num.notna().any():
        return as_num.astype("Float64")
    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().any():
        return (td.dt.total_seconds() / 86400.0).astype("Float64")
    return pd.Series(pd.NA, index=series.index, dtype="Float64")


def _calc_series_assign_lock_stats(product_win: pd.DataFrame) -> dict[str, dict[str, object]]:
    targets = ["L6", "LS6", "LS8", "LS9"]
    if product_win.empty:
        return {k: {"locks": 0, "avg": None, "mid": None} for k in targets}
    if "series" not in product_win.columns or "order_number" not in product_win.columns:
        return {k: {"locks": 0, "avg": None, "mid": None} for k in targets}

    df = product_win.copy()
    df["series"] = df["series"].astype("string").str.strip()
    has_interval = "first_assign_lock_time" in df.columns

    out: dict[str, dict[str, object]] = {}
    for s in targets:
        sub = df[df["series"].eq(s)]
        lock_cnt = int(sub["order_number"].nunique())
        mean = None
        median = None
        if has_interval:
            v = _to_interval_days(sub["first_assign_lock_time"]).dropna()
            if not v.empty:
                mean = float(v.mean())
                median = float(v.median())
        out[s] = {
            "locks": lock_cnt,
            "avg": (None if mean is None else round(mean, 1)),
            "mid": (None if median is None else round(median, 1)),
        }
    return out



def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in cols_lower:
            return str(cols_lower[key])
    return None


def _pick_person_id_column(df: pd.DataFrame) -> str | None:
    return _pick_column(df, ["owner_identity_no", "buyer_identity_no"])


def _resolve_journey_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "user_phone_md5": _pick_column(df, ["lc_user_phone_md5", "ic_user_phone_md5", "user_phone_md5", "phone_md5"]),
        "main_code": _pick_column(df, ["lc_main_code", "ic_main_code", "main_code", "clue_code"]),
        "channel": _pick_column(
            df,
            [
                "lc_small_channel_name",
                "ic_small_channel_name",
                "lc_small_channel",
                "small_channel_name",
                "channel_name",
            ],
        ),
        "create_time": _pick_column(df, ["lc_create_time", "ic_create_time", "create_time", "created_time"]),
        "lock_time": _pick_column(
            df,
            ["lc_order_lock_time_min", "ic_order_lock_time_min", "order_lock_time_min", "order_lock_time", "lock_time"],
        ),
    }



def _calc_active_store_count(df: pd.DataFrame, target_date: pd.Timestamp) -> int:
    df_store = df.copy()
    if "order_create_date" in df_store.columns:
        df_store["order_create_date"] = pd.to_datetime(
            df_store["order_create_date"], errors="coerce"
        )
        df_store["date"] = df_store["order_create_date"]

    if "order_create_time" in df_store.columns:
        df_store["order_create_time"] = pd.to_datetime(
            df_store["order_create_time"], errors="coerce"
        )
        if "date" not in df_store.columns:
            df_store["date"] = df_store["order_create_time"].dt.floor("D")
        else:
            df_store["date"] = df_store["date"].fillna(
                df_store["order_create_time"].dt.floor("D")
            )

    df_store["store_create_date"] = pd.to_datetime(
        df_store["store_create_date"], errors="coerce"
    )
    df_store = df_store.dropna(subset=["store_name", "date"])
    if df_store.empty:
        return 0

    open_map = df_store.groupby("store_name")["store_create_date"].min()

    d = target_date.normalize()
    window_start = d - pd.Timedelta(days=29)
    activity = df_store[(df_store["date"] >= window_start) & (df_store["date"] <= d)]
    if activity.empty:
        return 0

    stores = pd.Index(activity["store_name"].dropna().unique())
    store_open_dates = open_map.reindex(stores)
    is_open = store_open_dates <= d
    open_stores = stores[is_open.fillna(False).to_numpy()]
    return int(open_stores.size)


def _calc_order_metrics(
    order_path: Path, target_date: pd.Timestamp
) -> dict[str, int | float | str | None]:
    cols = [
        "order_number",
        "lock_time",
        "invoice_upload_time",
        "approve_refund_time",
        "order_type",
        "owner_identity_no",
        "buyer_identity_no",
        "series",
        "product_name",
        "store_name",
        "store_city",
        "store_create_date",
        "order_create_date",
        "invoice_amount",
        "first_assign_time",
        "intention_payment_time",
        "intention_refund_time",
        "deposit_payment_time",
        "deposit_refund_time",
        "first_assign_lock_time",
    ]
    try:
        df = pd.read_parquet(order_path, columns=cols)
    except Exception:
        df = pd.read_parquet(order_path)

    if "first_assign_time" in df.columns:
        df["first_assign_time"] = pd.to_datetime(df["first_assign_time"], errors="coerce")
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
        delta = df["lock_time"] - df["first_assign_time"]
        days = (delta.dt.total_seconds() / 86400.0).astype("Float64")
        df["first_assign_lock_time"] = days.where(days.ge(0))

    start = target_date.normalize()
    end = start + pd.Timedelta(days=1)

    lock_mask = df["lock_time"].notna() & (df["lock_time"] >= start) & (df["lock_time"] < end)
    invoice_mask = df["invoice_upload_time"].notna() & (df["invoice_upload_time"] >= start) & (df["invoice_upload_time"] < end)

    locked = df.loc[lock_mask, ["order_number", "approve_refund_time", "intention_payment_time", "intention_refund_time", "deposit_payment_time", "lock_time"]]
    lock_count = int(locked["order_number"].nunique())
    invoice_count = int(df.loc[invoice_mask, "order_number"].nunique())

    user_car_mask = df["order_type"] == "用户车"
    hq_mask = (df["store_city"].astype("string") == "拉萨市") & (
        df["store_name"].astype("string") == "总部主理店"
    )
    hq_mask = hq_mask.fillna(False)
    user_lock_mask = lock_mask & user_car_mask & (~hq_mask)
    locked_user_car = df.loc[
        user_lock_mask,
        [
            "order_number",
            "intention_payment_time",
            "intention_refund_time",
            "deposit_payment_time",
            "lock_time",
        ],
    ]
    user_car_lock_count = int(locked_user_car["order_number"].nunique())

    cond_intention = locked_user_car["intention_payment_time"].notna() & (
        locked_user_car["intention_refund_time"].isna()
        | (locked_user_car["intention_refund_time"] > locked_user_car["lock_time"])
    )
    cond_deposit = (~cond_intention) & locked_user_car["deposit_payment_time"].notna()
    locked_deposit = locked_user_car[cond_deposit]

    path_intention = int(locked_user_car.loc[cond_intention, "order_number"].nunique())
    
    path_retained_deposit_conv = int(locked_deposit.loc[locked_deposit["deposit_payment_time"] < start, "order_number"].nunique())
    path_today_deposit_conv = int(locked_deposit.loc[(locked_deposit["deposit_payment_time"] >= start) & (locked_deposit["deposit_payment_time"] < end), "order_number"].nunique())

    historic_deposit = (
        user_car_mask
        & (~hq_mask)
        & df["deposit_payment_time"].notna()
        & (df["deposit_payment_time"] >= start)
        & (df["deposit_payment_time"] < end)
    )
    not_locked_yet = df["lock_time"].isna() | (df["lock_time"] >= end)
    not_refunded_yet = df["deposit_refund_time"].isna() | (df["deposit_refund_time"] >= end)
    path_today_deposit_unlocked = int(df.loc[historic_deposit & not_locked_yet & not_refunded_yet, "order_number"].nunique())

    lock_refund_count = int(
        locked.loc[locked["approve_refund_time"].notna(), "order_number"].nunique()
    )
    lock_retained_count = int(
        locked.loc[locked["approve_refund_time"].isna(), "order_number"].nunique()
    )

    person_col = _pick_person_id_column(df)
    lock_people_count = int(
        df.loc[lock_mask & df[person_col].notna(), person_col].nunique() if person_col else 0
    )
    lock_people_count_non_test_drive = int(
        df.loc[
            lock_mask & df[person_col].notna() & (df["order_type"] != "试驾车"),
            person_col,
        ].nunique()
        if person_col
        else 0
    )

    active_store_count = _calc_active_store_count(
        df[
            [
                "store_name",
                "store_create_date",
                "order_create_date",
            ]
        ],
        target_date,
    )

    lock_win = df.loc[lock_mask, ["order_number", "store_name", "store_city"]].copy()
    if not lock_win.empty:
        lock_win["order_number"] = lock_win["order_number"].astype(str)

    cr5_store = None
    if not lock_win.empty and "store_name" in lock_win.columns:
        store_counts = (
            lock_win.dropna(subset=["store_name"])
            .groupby("store_name")["order_number"]
            .nunique()
        )
        total = float(store_counts.sum())
        if total > 0:
            top5 = float(store_counts.nlargest(5).sum())
            cr5_store = top5 / total

    cr5_city = None
    if not lock_win.empty and "store_city" in lock_win.columns:
        city_counts = (
            lock_win.dropna(subset=["store_city"])
            .groupby("store_city")["order_number"]
            .nunique()
        )
        total = float(city_counts.sum())
        if total > 0:
            top5 = float(city_counts.nlargest(5).sum())
            cr5_city = top5 / total

    atp_wan = None
    if "invoice_amount" in df.columns:
        inv_win = df.loc[invoice_mask & df["lock_time"].notna(), ["order_type", "invoice_amount"]].copy()
        if not inv_win.empty and "order_type" in inv_win.columns and "invoice_amount" in inv_win.columns:
            user_orders = inv_win[inv_win["order_type"] == "用户车"].copy()
            if not user_orders.empty:
                amt = pd.to_numeric(user_orders["invoice_amount"], errors="coerce")
                amt = amt[amt > 0]
                if not amt.empty:
                    atp_wan = float(amt.mean() / 10000.0)

    product_cols = ["order_number", "series", "product_name"]
    if "first_assign_lock_time" in df.columns:
        product_cols.append("first_assign_lock_time")
    product_win = df.loc[lock_mask & (df["order_type"] != "试驾车"), product_cols].copy()
    if not product_win.empty:
        product_win["order_number"] = product_win["order_number"].astype(str)
        product_win = product_win.dropna(subset=["order_number"])
        product_win = product_win.drop_duplicates(subset=["order_number"], keep="first")
        if "series" in product_win.columns:
            product_win["series"] = product_win["series"].astype("string").str.strip()
        if "product_name" in product_win.columns:
            product_win["product_name"] = product_win["product_name"].astype("string")

    denom_orders = int(product_win["order_number"].nunique()) if not product_win.empty else 0

    share_reev = None
    if denom_orders > 0:
        pn = product_win.get("product_name")
        if pn is not None:
            pn = pn.astype("string").fillna("")
            is_reev = pn.str.contains("52", regex=False) | pn.str.contains("66", regex=False)
            share_reev = _safe_ratio(int(is_reev.sum()), denom_orders)

    series_stats = _calc_series_assign_lock_stats(product_win)

    return {
        "锁单数": lock_count,
        "锁单数（order_type =用户车&非总部）": user_car_lock_count,
        "锁单路径盘点": [
            {"留存小订转化": path_intention},
            {"留存大定转化": path_retained_deposit_conv},
            {"当日大定转化": path_today_deposit_conv},
            {"当日大定留存": path_today_deposit_unlocked},
        ],
        "锁单人数": lock_people_count,
        "锁单人数（order_type != 试驾车）": lock_people_count_non_test_drive,
        "开票数": invoice_count,
        "累计锁单退订数": lock_refund_count,
        "留存锁单数": lock_retained_count,
        "在营门店数": active_store_count,
        "CR5门店销量集中度": _to_percent_1dp(cr5_store),
        "CR5门店城市销量集中度": _to_percent_1dp(cr5_city),
        "整体ATP(用户车,万元)": (None if atp_wan is None else round(atp_wan, 2)),
        "L6": series_stats["L6"],
        "LS6": series_stats["LS6"],
        "LS8": series_stats["LS8"],
        "LS9": series_stats["LS9"],
        "share_reev": _to_percent_1dp(share_reev),
    }


def _calc_attribution_metrics(
    attribution_path: Path,
    target_date: pd.Timestamp,
    order_lock_people_non_test_drive: int,
) -> dict[str, object]:
    cols = [
        "lc_main_code",
        "lc_user_phone_md5",
        "lc_create_time",
        "lc_order_lock_time_min",
        "lc_small_channel_name",
    ]
    try:
        df = pd.read_parquet(attribution_path, columns=cols)
    except Exception:
        df = pd.read_parquet(attribution_path)

    cols_map = _resolve_journey_columns(df)
    main_code_col = cols_map["main_code"] or "lc_main_code"
    user_col = cols_map["user_phone_md5"] or "lc_user_phone_md5"
    create_time_col = cols_map["create_time"] or "lc_create_time"
    lock_time_col = cols_map["lock_time"] or "lc_order_lock_time_min"
    channel_col = cols_map["channel"]

    if user_col not in df.columns or create_time_col not in df.columns or lock_time_col not in df.columns:
        return {
            "锁单用户数": 0,
            "数据完整度": None,
            "平均触达次数": None,
            "平均转化时长(天)": None,
            "锁单用户主要渠道Top5": [],
            "锁单用户分类占比（观察口径）": [],
            "跨渠道锁单用户主要助攻渠道Top5": [],
        }

    df[create_time_col] = pd.to_datetime(df[create_time_col], errors="coerce")
    df[lock_time_col] = pd.to_datetime(df[lock_time_col], errors="coerce")

    if main_code_col in df.columns:
        df = df.drop_duplicates(subset=[main_code_col], keep="first").copy()

    df = df.dropna(subset=[user_col, create_time_col]).copy()
    sort_cols = [user_col, create_time_col]
    if main_code_col in df.columns:
        sort_cols.append(main_code_col)
    df = df.sort_values(sort_cols, kind="mergesort")
    df["touch_index"] = df.groupby(user_col, dropna=False).cumcount() + 1
    delta = df[lock_time_col] - df[create_time_col]
    df["time_to_lock_days"] = (delta.dt.total_seconds() / 86400).astype("Float64")

    start = target_date.normalize()
    end = start + pd.Timedelta(days=1)
    locked = df[df[lock_time_col].notna() & (df[lock_time_col] >= start) & (df[lock_time_col] < end)].copy()

    locked_users = 0
    touch_mean = None
    ttl_mean = None
    channel_top_out: list[dict[str, object]] = []
    lens_out_records: list[dict[str, object]] = []
    assist_out_records: list[dict[str, object]] = []
    if not locked.empty:
        sort_cols = [user_col, lock_time_col]
        if main_code_col in locked.columns:
            sort_cols.append(main_code_col)
        locked = locked.sort_values(sort_cols, kind="mergesort")
        per_user_day = locked.groupby(user_col, dropna=False, as_index=False).first()
        locked_users = int(per_user_day[user_col].nunique(dropna=True))
        touch_mean = float(per_user_day["touch_index"].mean()) if not per_user_day.empty else None
        ttl = per_user_day["time_to_lock_days"].dropna()
        ttl_mean = (float(ttl.mean()) if not ttl.empty else None)

        if channel_col is not None and channel_col in df.columns and main_code_col in df.columns:
            lock_time_by_user = per_user_day[
                [user_col, lock_time_col, channel_col, "time_to_lock_days", main_code_col]
            ].copy()
            lock_time_by_user = lock_time_by_user.rename(
                columns={
                    user_col: "user",
                    lock_time_col: "first_lock_time",
                    channel_col: "lock_channel",
                    "time_to_lock_days": "ttl_days",
                    main_code_col: "lock_main_code",
                }
            )
            lock_time_by_user["user"] = lock_time_by_user["user"].astype("string")
            lock_time_by_user["lock_channel"] = lock_time_by_user["lock_channel"].astype("string")

            touches = df[[user_col, main_code_col, channel_col, create_time_col]].copy()
            touches = touches[touches[create_time_col].notna()].copy()
            touches["user"] = touches[user_col].astype("string")
            touches = touches.merge(lock_time_by_user[["user", "first_lock_time"]], on="user", how="inner")
            touches = touches[touches[create_time_col].le(touches["first_lock_time"])].copy()

            touch_agg = touches.groupby("user", dropna=False).agg(
                touches_to_lock=(create_time_col, "size"),
                distinct_channels_to_lock=(channel_col, lambda s: int(pd.Series(s.dropna()).nunique())),
            )
            touch_agg = touch_agg.reset_index()

            user_summary = lock_time_by_user.merge(touch_agg, on="user", how="left")
            user_summary["touches_to_lock"] = user_summary["touches_to_lock"].fillna(0).astype(int)
            user_summary["distinct_channels_to_lock"] = user_summary["distinct_channels_to_lock"].fillna(0).astype(int)

            channel_series = user_summary["lock_channel"].astype("string").fillna("(missing)")
            vc = channel_series.value_counts(dropna=False)
            total_users = int(vc.sum())
            top_n = 5
            top = vc.head(max(int(top_n), 1))
            channel_top = pd.DataFrame({"channel": top.index.astype("string"), "locked_users": top.values})
            other_cnt = int(vc.iloc[len(top) :].sum())
            if other_cnt:
                channel_top = pd.concat(
                    [channel_top, pd.DataFrame([{"channel": "其他", "locked_users": other_cnt}])],
                    ignore_index=True,
                )
            channel_top["pct"] = channel_top["locked_users"] / max(total_users, 1)
            channel_top["pct"] = channel_top["pct"].map(
                lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
            )
            channel_top_out = channel_top.to_dict(orient="records")

            one_touch_users = int(user_summary["touches_to_lock"].astype(int).eq(1).sum())
            same_channel_multi_users = int(
                (
                    user_summary["touches_to_lock"].astype(int).gt(1)
                    & user_summary["distinct_channels_to_lock"].astype(int).eq(1)
                ).sum()
            )
            cross_channel_users = int(user_summary["distinct_channels_to_lock"].astype(int).gt(1).sum())
            long_users = int(user_summary["ttl_days"].astype("Float64").gt(14).fillna(False).sum())
            long_14_60_users = int(
                (
                    user_summary["ttl_days"].astype("Float64").gt(14)
                    & user_summary["ttl_days"].astype("Float64").lt(60)
                )
                .fillna(False)
                .sum()
            )
            prior_lock_users = df.loc[
                df[lock_time_col].notna() & df[lock_time_col].lt(start),
                user_col,
            ].astype("string")
            prior_lock_users = set(prior_lock_users.dropna().tolist())
            repeat_lock_users = int(user_summary["user"].astype("string").isin(prior_lock_users).sum())

            lens_out = pd.DataFrame(
                [
                    {"category": "One-Touch (Decisive)", "users": one_touch_users},
                    {"category": "Hesitant (Same Channel, Multiple Touches)", "users": same_channel_multi_users},
                    {"category": "Cross-Channel (Comparison Shopper)", "users": cross_channel_users},
                    {"category": "Long Consideration (>14 Days)", "users": long_users},
                    {"category": "Long Consideration (>14 Days & <60 Days)", "users": long_14_60_users},
                    {"category": "Repeat Lockers (Had Prior Locks)", "users": repeat_lock_users},
                ]
            )
            lens_out["pct"] = lens_out["users"] / max(int(user_summary.shape[0]), 1)
            lens_out["pct"] = lens_out["pct"].map(
                lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
            )
            lens_out_records = lens_out.to_dict(orient="records")

            if cross_channel_users <= 0:
                assist_out = pd.DataFrame(columns=["assist_channel", "assist_touches", "pct"])
            else:
                cross_users = user_summary.loc[user_summary["distinct_channels_to_lock"].gt(1), "user"].copy()
                assist = touches[touches["user"].astype("string").isin(cross_users.astype("string"))].copy()
                assist = assist.merge(lock_time_by_user[["user", "lock_channel", "lock_main_code"]], on="user", how="left")
                assist["assist_channel"] = assist[channel_col].astype("string")
                assist = assist[
                    assist["assist_channel"].notna()
                    & assist["lock_channel"].notna()
                    & assist["lock_main_code"].notna()
                ].copy()
                assist = assist[assist["assist_channel"] != assist["lock_channel"]].copy()
                assist = assist[assist[main_code_col] != assist["lock_main_code"]].copy()

                vc_assist = assist["assist_channel"].value_counts(dropna=False)
                assist_total = int(len(assist))
                top_assist = vc_assist.head(max(int(top_n), 1))
                assist_out = pd.DataFrame(
                    {"assist_channel": top_assist.index.astype("string"), "assist_touches": top_assist.values}
                )
                assist_out["pct"] = assist_out["assist_touches"] / max(assist_total, 1)
                assist_out["pct"] = assist_out["pct"].map(
                    lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
                )

            assist_out_records = assist_out.to_dict(orient="records")

    return {
        "锁单用户数": locked_users,
        "数据完整度": _to_percent_1dp(_safe_ratio(locked_users, order_lock_people_non_test_drive)),
        "平均触达次数": (None if touch_mean is None else round(touch_mean, 2)),
        "平均转化时长(天)": (None if ttl_mean is None else round(ttl_mean, 2)),
        "锁单用户主要渠道Top5": channel_top_out,
        "锁单用户分类占比（观察口径）": lens_out_records,
        "跨渠道锁单用户主要助攻渠道Top5": assist_out_records,
    }


def _calc_assign_metrics(
    assign_path: Path, target_date: pd.Timestamp
) -> dict[str, int | float | str | None]:
    df = _read_metrics_csv(assign_path)
    df["_date"] = _parse_cn_ymd_series(df["Assign Time 年/月/日"])
    day = df[df["_date"] == target_date.normalize()].copy()
    col_map = {str(c).strip(): c for c in day.columns}

    def _sum(col: str) -> int:
        col2 = col_map.get(str(col).strip())
        if col2 is None:
            return 0
        return int(_to_numeric_clean(day[col2]).fillna(0).sum())

    def _sum_any(candidates: list[str]) -> int:
        for col in candidates:
            v = _sum(col)
            if v != 0:
                return v
        return 0

    leads = _sum("下发线索数")
    td0 = _sum("下发线索当日试驾数")

    lock0_source = None
    lock0 = 0
    if "下发线索当日锁单数 (门店)" in col_map:
        lock0 = _sum("下发线索当日锁单数 (门店)")
        lock0_source = "下发线索当日锁单数 (门店)"
    elif "下发线索当日锁单数" in col_map:
        lock0 = _sum("下发线索当日锁单数")
        lock0_source = "下发线索当日锁单数"

    lock7 = _sum("下发线索 7 日锁单数")
    lock30 = _sum("下发线索 30 日锁单数")
    store_cnt = _sum("下发门店数")
    store_leads = _sum("下发线索数 (门店)")

    leads_live = _sum_any(["下发线索数（直播）", "下发线索数（直播)", "下发线索数 (直播)", "下发线索数 (直播)"])
    leads_platform = _sum_any(["下发线索数（平台)", "下发线索数（平台）", "下发线索数 (平台)", "下发线索数 (平台)"])
    leads_app = _sum_any(["下发线索数（APP小程序)", "下发线索数（APP小程序）", "下发线索数 (APP小程序)", "下发线索数 (APP小程序)"])
    leads_flash = _sum_any(["下发线索数（快慢闪)", "下发线索数（快慢闪）", "下发线索数 (快慢闪)", "下发线索数 (快慢闪)"])

    lock7_store = _sum_any(["下发线索 7 日锁单数 (门店)", "下发线索 7日锁单数 (门店)", "下发线索7日锁单数 (门店)"])
    lock7_live = _sum_any(["下发线索 7 日锁单数 (直播)", "下发线索 7日锁单数 (直播)", "下发线索7日锁单数 (直播)"])
    lock7_platform = _sum_any(["下发线索 7 日锁单数 (平台)", "下发线索 7日锁单数 (平台)", "下发线索7日锁单数 (平台)"])

    lock30_store = _sum_any(["下发线索 30 日锁单数 (门店)", "下发线索 30日锁单数 (门店)", "下发线索30日锁单数 (门店)"])
    lock30_live = _sum_any(["下发线索 30 日锁单数 (直播)", "下发线索 30日锁单数 (直播)", "下发线索30日锁单数 (直播)"])
    lock30_platform = _sum_any(["下发线索 30 日锁单数 (平台)", "下发线索 30日锁单数 (平台)", "下发线索30日锁单数 (平台)"])
    lock30_app = _sum_any(["下发线索 30 日锁单数 (APP小程序)", "下发线索 30日锁单数 (APP小程序)", "下发线索30日锁单数 (APP小程序)"])
    lock30_flash = _sum_any(["下发线索 30 日锁单数 (快慢闪)", "下发线索 30日锁单数 (快慢闪)", "下发线索30日锁单数 (快慢闪)"])

    lock0_rate_denom = store_leads if store_leads > 0 else leads

    out: dict[str, int | float | str | None] = {
        "下发线索数": leads,
        "下发线索数 (门店)": store_leads,
        "下发线索数（直播）": leads_live,
        "下发线索数（平台)": leads_platform,
        "下发线索数（APP小程序)": leads_app,
        "下发线索数（快慢闪)": leads_flash,
        "下发线索当日试驾数": td0,
        "下发 (门店)线索当日锁单数": lock0,
        "下发线索 7 日锁单数": lock7,
        "下发线索 30 日锁单数": lock30,
        "下发门店数": store_cnt,
        "门店线索占比": _to_percent_1dp(_safe_ratio(store_leads, leads)),
        "下发线索当日试驾率": _to_percent_1dp(_safe_ratio(td0, leads)),
        "下发 (门店)线索当日锁单率": _to_percent_1dp(_safe_ratio(lock0, lock0_rate_denom)),
        "下发线索当7日锁单率": _to_percent_1dp(_safe_ratio(lock7, leads)),
        "下发线索当30日锁单率": _to_percent_1dp(_safe_ratio(lock30, leads)),
        "下发线索数（门店)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_store, store_leads)),
        "下发线索数（直播)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_live, leads_live)),
        "下发线索数（平台)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_platform, leads_platform)),
        "下发线索数（门店)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_store, store_leads)),
        "下发线索数（直播)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_live, leads_live)),
        "下发线索数（平台)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_platform, leads_platform)),
        "下发线索数（APP小程序)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_app, leads_app)),
        "下发线索数（快慢闪)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_flash, leads_flash)),
    }

    if lock0_source is not None:
        out["下发 (门店)线索当日锁单数_字段来源"] = lock0_source

    return out


def _calc_test_drive_metrics(test_drive_path: Path, target_date: pd.Timestamp) -> dict[str, int]:
    df = _read_metrics_csv(test_drive_path)
    df["_date"] = _parse_cn_ymd_series(df["create_date 年/月/日"])
    day = df[df["_date"] == target_date.normalize()]
    effective = int(_to_numeric_clean(day.get("有效试驾数")).fillna(0).sum())
    return {"有效试驾数": effective}


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return []
    return [pd.Timestamp(d).normalize() for d in pd.date_range(start, end, freq="D")]


def _load_order_df(order_path: Path) -> pd.DataFrame:
    cols = [
        "order_number",
        "lock_time",
        "invoice_upload_time",
        "approve_refund_time",
        "order_type",
        "owner_identity_no",
        "buyer_identity_no",
        "series",
        "product_name",
        "store_name",
        "store_city",
        "store_create_date",
        "order_create_date",
        "first_assign_time",
        "intention_payment_time",
        "intention_refund_time",
        "deposit_payment_time",
        "deposit_refund_time",
        "first_assign_lock_time",
    ]
    try:
        df = pd.read_parquet(order_path, columns=cols)
    except Exception:
        df = pd.read_parquet(order_path)

    if "first_assign_time" in df.columns:
        df["first_assign_time"] = pd.to_datetime(df["first_assign_time"], errors="coerce")
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
        delta = df["lock_time"] - df["first_assign_time"]
        days = (delta.dt.total_seconds() / 86400.0).astype("Float64")
        df["first_assign_lock_time"] = days.where(days.ge(0))

    return df


def _calc_order_metrics_from_df(
    df: pd.DataFrame, target_date: pd.Timestamp
) -> dict[str, int | float | str | None]:
    start = target_date.normalize()
    end = start + pd.Timedelta(days=1)

    lock_mask = df["lock_time"].notna() & (df["lock_time"] >= start) & (df["lock_time"] < end)
    invoice_mask = df["invoice_upload_time"].notna() & (df["invoice_upload_time"] >= start) & (df["invoice_upload_time"] < end)

    locked = df.loc[lock_mask, ["order_number", "approve_refund_time", "intention_payment_time", "intention_refund_time", "deposit_payment_time", "lock_time"]]
    lock_count = int(locked["order_number"].nunique())
    invoice_count = int(df.loc[invoice_mask, "order_number"].nunique())

    user_car_mask = df["order_type"] == "用户车"
    hq_mask = (df["store_city"].astype("string") == "拉萨市") & (
        df["store_name"].astype("string") == "总部主理店"
    )
    hq_mask = hq_mask.fillna(False)
    user_lock_mask = lock_mask & user_car_mask & (~hq_mask)
    locked_user_car = df.loc[
        user_lock_mask,
        [
            "order_number",
            "intention_payment_time",
            "intention_refund_time",
            "deposit_payment_time",
            "lock_time",
        ],
    ]
    user_car_lock_count = int(locked_user_car["order_number"].nunique())

    cond_intention = locked_user_car["intention_payment_time"].notna() & (
        locked_user_car["intention_refund_time"].isna()
        | (locked_user_car["intention_refund_time"] > locked_user_car["lock_time"])
    )
    cond_deposit = (~cond_intention) & locked_user_car["deposit_payment_time"].notna()
    locked_deposit = locked_user_car[cond_deposit]

    path_intention = int(locked_user_car.loc[cond_intention, "order_number"].nunique())
    
    path_retained_deposit_conv = int(locked_deposit.loc[locked_deposit["deposit_payment_time"] < start, "order_number"].nunique())
    path_today_deposit_conv = int(locked_deposit.loc[(locked_deposit["deposit_payment_time"] >= start) & (locked_deposit["deposit_payment_time"] < end), "order_number"].nunique())

    historic_deposit = (
        user_car_mask
        & (~hq_mask)
        & df["deposit_payment_time"].notna()
        & (df["deposit_payment_time"] >= start)
        & (df["deposit_payment_time"] < end)
    )
    not_locked_yet = df["lock_time"].isna() | (df["lock_time"] >= end)
    not_refunded_yet = df["deposit_refund_time"].isna() | (df["deposit_refund_time"] >= end)
    path_today_deposit_unlocked = int(df.loc[historic_deposit & not_locked_yet & not_refunded_yet, "order_number"].nunique())

    lock_refund_count = int(
        locked.loc[locked["approve_refund_time"].notna(), "order_number"].nunique()
    )
    lock_retained_count = int(
        locked.loc[locked["approve_refund_time"].isna(), "order_number"].nunique()
    )

    person_col = _pick_person_id_column(df)
    lock_people_count = int(
        df.loc[lock_mask & df[person_col].notna(), person_col].nunique() if person_col else 0
    )
    lock_people_count_non_test_drive = int(
        df.loc[
            lock_mask & df[person_col].notna() & (df["order_type"] != "试驾车"),
            person_col,
        ].nunique()
        if person_col
        else 0
    )

    active_store_count = _calc_active_store_count(
        df[
            [
                "store_name",
                "store_create_date",
                "order_create_date",
            ]
        ],
        target_date,
    )

    lock_win = df.loc[lock_mask, ["order_number", "store_name", "store_city"]].copy()
    if not lock_win.empty:
        lock_win["order_number"] = lock_win["order_number"].astype(str)

    cr5_store = None
    if not lock_win.empty and "store_name" in lock_win.columns:
        store_counts = (
            lock_win.dropna(subset=["store_name"])
            .groupby("store_name")["order_number"]
            .nunique()
        )
        total = float(store_counts.sum())
        if total > 0:
            top5 = float(store_counts.nlargest(5).sum())
            cr5_store = top5 / total

    cr5_city = None
    if not lock_win.empty and "store_city" in lock_win.columns:
        city_counts = (
            lock_win.dropna(subset=["store_city"])
            .groupby("store_city")["order_number"]
            .nunique()
        )
        total = float(city_counts.sum())
        if total > 0:
            top5 = float(city_counts.nlargest(5).sum())
            cr5_city = top5 / total

    atp_wan = None
    if "invoice_amount" in df.columns:
        inv_win = df.loc[invoice_mask & df["lock_time"].notna(), ["order_type", "invoice_amount"]].copy()
        if not inv_win.empty and "order_type" in inv_win.columns and "invoice_amount" in inv_win.columns:
            user_orders = inv_win[inv_win["order_type"] == "用户车"].copy()
            if not user_orders.empty:
                amt = pd.to_numeric(user_orders["invoice_amount"], errors="coerce")
                amt = amt[amt > 0]
                if not amt.empty:
                    atp_wan = float(amt.mean() / 10000.0)

    product_cols = ["order_number", "series", "product_name"]
    if "first_assign_lock_time" in df.columns:
        product_cols.append("first_assign_lock_time")
    product_win = df.loc[lock_mask & (df["order_type"] != "试驾车"), product_cols].copy()
    if not product_win.empty:
        product_win["order_number"] = product_win["order_number"].astype(str)
        product_win = product_win.dropna(subset=["order_number"])
        product_win = product_win.drop_duplicates(subset=["order_number"], keep="first")
        if "series" in product_win.columns:
            product_win["series"] = product_win["series"].astype("string").str.strip()
        if "product_name" in product_win.columns:
            product_win["product_name"] = product_win["product_name"].astype("string")

    denom_orders = int(product_win["order_number"].nunique()) if not product_win.empty else 0

    share_reev = None
    if denom_orders > 0:
        pn = product_win.get("product_name")
        if pn is not None:
            pn = pn.astype("string").fillna("")
            is_reev = pn.str.contains("52", regex=False) | pn.str.contains("66", regex=False)
            share_reev = _safe_ratio(int(is_reev.sum()), denom_orders)

    series_stats = _calc_series_assign_lock_stats(product_win)

    return {
        "锁单数": lock_count,
        "锁单数（order_type =用户车&非总部）": user_car_lock_count,
        "锁单路径盘点": [
            {"留存小订转化": path_intention},
            {"留存大定转化": path_retained_deposit_conv},
            {"当日大定转化": path_today_deposit_conv},
            {"当日大定留存": path_today_deposit_unlocked},
        ],
        "锁单人数": lock_people_count,
        "锁单人数（order_type != 试驾车）": lock_people_count_non_test_drive,
        "开票数": invoice_count,
        "累计锁单退订数": lock_refund_count,
        "留存锁单数": lock_retained_count,
        "在营门店数": active_store_count,
        "CR5门店销量集中度": _to_percent_1dp(cr5_store),
        "CR5门店城市销量集中度": _to_percent_1dp(cr5_city),
        "整体ATP(用户车,万元)": (None if atp_wan is None else round(atp_wan, 2)),
        "L6": series_stats["L6"],
        "LS6": series_stats["LS6"],
        "LS8": series_stats["LS8"],
        "LS9": series_stats["LS9"],
        "share_reev": _to_percent_1dp(share_reev),
    }


def _load_assign_df(assign_path: Path) -> pd.DataFrame:
    df = _read_metrics_csv(assign_path)
    df["_date"] = _parse_cn_ymd_series(df["Assign Time 年/月/日"])
    return df


def _calc_assign_metrics_from_df(
    df: pd.DataFrame, target_date: pd.Timestamp
) -> dict[str, int | float | str | None]:
    day = df[df["_date"] == target_date.normalize()].copy()
    col_map = {str(c).strip(): c for c in day.columns}

    def _sum(col: str) -> int:
        col2 = col_map.get(str(col).strip())
        if col2 is None:
            return 0
        return int(_to_numeric_clean(day[col2]).fillna(0).sum())

    def _sum_any(candidates: list[str]) -> int:
        for col in candidates:
            v = _sum(col)
            if v != 0:
                return v
        return 0

    leads = _sum("下发线索数")
    td0 = _sum("下发线索当日试驾数")

    lock0_source = None
    lock0 = 0
    if "下发线索当日锁单数 (门店)" in col_map:
        lock0 = _sum("下发线索当日锁单数 (门店)")
        lock0_source = "下发线索当日锁单数 (门店)"
    elif "下发线索当日锁单数" in col_map:
        lock0 = _sum("下发线索当日锁单数")
        lock0_source = "下发线索当日锁单数"

    lock7 = _sum("下发线索 7 日锁单数")
    lock30 = _sum("下发线索 30 日锁单数")
    store_cnt = _sum("下发门店数")
    store_leads = _sum("下发线索数 (门店)")

    leads_live = _sum_any(["下发线索数（直播）", "下发线索数（直播)", "下发线索数 (直播)", "下发线索数 (直播)"])
    leads_platform = _sum_any(["下发线索数（平台)", "下发线索数（平台）", "下发线索数 (平台)", "下发线索数 (平台)"])
    leads_app = _sum_any(["下发线索数（APP小程序)", "下发线索数（APP小程序）", "下发线索数 (APP小程序)", "下发线索数 (APP小程序)"])
    leads_flash = _sum_any(["下发线索数（快慢闪)", "下发线索数（快慢闪）", "下发线索数 (快慢闪)", "下发线索数 (快慢闪)"])

    lock7_store = _sum_any(["下发线索 7 日锁单数 (门店)", "下发线索 7日锁单数 (门店)", "下发线索7日锁单数 (门店)"])
    lock7_live = _sum_any(["下发线索 7 日锁单数 (直播)", "下发线索 7日锁单数 (直播)", "下发线索7日锁单数 (直播)"])
    lock7_platform = _sum_any(["下发线索 7 日锁单数 (平台)", "下发线索 7日锁单数 (平台)", "下发线索7日锁单数 (平台)"])

    lock30_store = _sum_any(["下发线索 30 日锁单数 (门店)", "下发线索 30日锁单数 (门店)", "下发线索30日锁单数 (门店)"])
    lock30_live = _sum_any(["下发线索 30 日锁单数 (直播)", "下发线索 30日锁单数 (直播)", "下发线索30日锁单数 (直播)"])
    lock30_platform = _sum_any(["下发线索 30 日锁单数 (平台)", "下发线索 30日锁单数 (平台)", "下发线索30日锁单数 (平台)"])
    lock30_app = _sum_any(["下发线索 30 日锁单数 (APP小程序)", "下发线索 30日锁单数 (APP小程序)", "下发线索30日锁单数 (APP小程序)"])
    lock30_flash = _sum_any(["下发线索 30 日锁单数 (快慢闪)", "下发线索 30日锁单数 (快慢闪)", "下发线索30日锁单数 (快慢闪)"])

    lock0_rate_denom = store_leads if store_leads > 0 else leads

    out: dict[str, int | float | str | None] = {
        "下发线索数": leads,
        "下发线索数 (门店)": store_leads,
        "下发线索数（直播）": leads_live,
        "下发线索数（平台)": leads_platform,
        "下发线索数（APP小程序)": leads_app,
        "下发线索数（快慢闪)": leads_flash,
        "下发线索当日试驾数": td0,
        "下发 (门店)线索当日锁单数": lock0,
        "下发线索 7 日锁单数": lock7,
        "下发线索 30 日锁单数": lock30,
        "下发门店数": store_cnt,
        "门店线索占比": _to_percent_1dp(_safe_ratio(store_leads, leads)),
        "下发线索当日试驾率": _to_percent_1dp(_safe_ratio(td0, leads)),
        "下发 (门店)线索当日锁单率": _to_percent_1dp(_safe_ratio(lock0, lock0_rate_denom)),
        "下发线索当7日锁单率": _to_percent_1dp(_safe_ratio(lock7, leads)),
        "下发线索当30日锁单率": _to_percent_1dp(_safe_ratio(lock30, leads)),
        "下发线索数（门店)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_store, store_leads)),
        "下发线索数（直播)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_live, leads_live)),
        "下发线索数（平台)7日锁单率": _to_percent_1dp(_safe_ratio(lock7_platform, leads_platform)),
        "下发线索数（门店)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_store, store_leads)),
        "下发线索数（直播)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_live, leads_live)),
        "下发线索数（平台)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_platform, leads_platform)),
        "下发线索数（APP小程序)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_app, leads_app)),
        "下发线索数（快慢闪)30日锁单率": _to_percent_1dp(_safe_ratio(lock30_flash, leads_flash)),
    }

    if lock0_source is not None:
        out["下发 (门店)线索当日锁单数_字段来源"] = lock0_source

    return out


def _load_test_drive_df(test_drive_path: Path) -> pd.DataFrame:
    df = _read_metrics_csv(test_drive_path)
    df["_date"] = _parse_cn_ymd_series(df["create_date 年/月/日"])
    return df


def _calc_test_drive_metrics_from_df(df: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, int]:
    day = df[df["_date"] == target_date.normalize()]
    effective = int(_to_numeric_clean(day.get("有效试驾数")).fillna(0).sum())
    return {"有效试驾数": effective}


def _prepare_attribution_df(attribution_path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols = [
        "lc_main_code",
        "lc_user_phone_md5",
        "lc_create_time",
        "lc_order_lock_time_min",
        "lc_small_channel_name",
    ]
    try:
        df = pd.read_parquet(attribution_path, columns=cols)
    except Exception:
        df = pd.read_parquet(attribution_path)

    cols_map = _resolve_journey_columns(df)
    main_code_col = cols_map["main_code"] or "lc_main_code"
    user_col = cols_map["user_phone_md5"] or "lc_user_phone_md5"
    create_time_col = cols_map["create_time"] or "lc_create_time"
    lock_time_col = cols_map["lock_time"] or "lc_order_lock_time_min"

    if user_col not in df.columns or create_time_col not in df.columns or lock_time_col not in df.columns:
        return df, cols_map

    df = df.copy()
    df[create_time_col] = pd.to_datetime(df[create_time_col], errors="coerce")
    df[lock_time_col] = pd.to_datetime(df[lock_time_col], errors="coerce")

    if main_code_col in df.columns:
        df = df.drop_duplicates(subset=[main_code_col], keep="first").copy()

    df = df.dropna(subset=[user_col, create_time_col]).copy()
    sort_cols = [user_col, create_time_col]
    if main_code_col in df.columns:
        sort_cols.append(main_code_col)
    df = df.sort_values(sort_cols, kind="mergesort")
    df["touch_index"] = df.groupby(user_col, dropna=False).cumcount() + 1
    delta = df[lock_time_col] - df[create_time_col]
    df["time_to_lock_days"] = (delta.dt.total_seconds() / 86400).astype("Float64")
    return df, cols_map


def _calc_attribution_metrics_from_df(
    df: pd.DataFrame,
    cols_map: dict[str, str | None],
    target_date: pd.Timestamp,
    order_lock_people_non_test_drive: int,
) -> dict[str, object]:
    main_code_col = cols_map["main_code"] or "lc_main_code"
    user_col = cols_map["user_phone_md5"] or "lc_user_phone_md5"
    create_time_col = cols_map["create_time"] or "lc_create_time"
    lock_time_col = cols_map["lock_time"] or "lc_order_lock_time_min"
    channel_col = cols_map["channel"]

    if user_col not in df.columns or create_time_col not in df.columns or lock_time_col not in df.columns:
        return {
            "锁单用户数": 0,
            "数据完整度": None,
            "平均触达次数": None,
            "平均转化时长(天)": None,
            "锁单用户主要渠道Top5": [],
            "锁单用户分类占比（观察口径）": [],
            "跨渠道锁单用户主要助攻渠道Top5": [],
        }

    start = target_date.normalize()
    end = start + pd.Timedelta(days=1)
    locked = df[df[lock_time_col].notna() & (df[lock_time_col] >= start) & (df[lock_time_col] < end)].copy()

    locked_users = 0
    touch_mean = None
    ttl_mean = None
    channel_top_out: list[dict[str, object]] = []
    lens_out_records: list[dict[str, object]] = []
    assist_out_records: list[dict[str, object]] = []
    if not locked.empty:
        sort_cols = [user_col, lock_time_col]
        if main_code_col in locked.columns:
            sort_cols.append(main_code_col)
        locked = locked.sort_values(sort_cols, kind="mergesort")
        per_user_day = locked.groupby(user_col, dropna=False, as_index=False).first()
        locked_users = int(per_user_day[user_col].nunique(dropna=True))
        touch_mean = float(per_user_day["touch_index"].mean()) if not per_user_day.empty else None
        ttl = per_user_day["time_to_lock_days"].dropna()
        ttl_mean = (float(ttl.mean()) if not ttl.empty else None)

        if channel_col is not None and channel_col in df.columns and main_code_col in df.columns:
            lock_time_by_user = per_user_day[
                [user_col, lock_time_col, channel_col, "time_to_lock_days", main_code_col]
            ].copy()
            lock_time_by_user = lock_time_by_user.rename(
                columns={
                    user_col: "user",
                    lock_time_col: "first_lock_time",
                    channel_col: "lock_channel",
                    "time_to_lock_days": "ttl_days",
                    main_code_col: "lock_main_code",
                }
            )
            lock_time_by_user["user"] = lock_time_by_user["user"].astype("string")
            lock_time_by_user["lock_channel"] = lock_time_by_user["lock_channel"].astype("string")

            touches = df[[user_col, main_code_col, channel_col, create_time_col]].copy()
            touches = touches[touches[create_time_col].notna()].copy()
            touches["user"] = touches[user_col].astype("string")
            touches = touches.merge(lock_time_by_user[["user", "first_lock_time"]], on="user", how="inner")
            touches = touches[touches[create_time_col].le(touches["first_lock_time"])].copy()

            touch_agg = touches.groupby("user", dropna=False).agg(
                touches_to_lock=(create_time_col, "size"),
                distinct_channels_to_lock=(channel_col, lambda s: int(pd.Series(s.dropna()).nunique())),
            )
            touch_agg = touch_agg.reset_index()

            user_summary = lock_time_by_user.merge(touch_agg, on="user", how="left")
            user_summary["touches_to_lock"] = user_summary["touches_to_lock"].fillna(0).astype(int)
            user_summary["distinct_channels_to_lock"] = user_summary["distinct_channels_to_lock"].fillna(0).astype(int)

            channel_series = user_summary["lock_channel"].astype("string").fillna("(missing)")
            vc = channel_series.value_counts(dropna=False)
            total_users = int(vc.sum())
            top_n = 5
            top = vc.head(max(int(top_n), 1))
            channel_top = pd.DataFrame({"channel": top.index.astype("string"), "locked_users": top.values})
            other_cnt = int(vc.iloc[len(top) :].sum())
            if other_cnt:
                channel_top = pd.concat(
                    [channel_top, pd.DataFrame([{"channel": "其他", "locked_users": other_cnt}])],
                    ignore_index=True,
                )
            channel_top["pct"] = channel_top["locked_users"] / max(total_users, 1)
            channel_top["pct"] = channel_top["pct"].map(
                lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
            )
            channel_top_out = channel_top.to_dict(orient="records")

            one_touch_users = int(user_summary["touches_to_lock"].astype(int).eq(1).sum())
            same_channel_multi_users = int(
                (
                    user_summary["touches_to_lock"].astype(int).gt(1)
                    & user_summary["distinct_channels_to_lock"].astype(int).eq(1)
                ).sum()
            )
            cross_channel_users = int(user_summary["distinct_channels_to_lock"].astype(int).gt(1).sum())
            long_users = int(user_summary["ttl_days"].astype("Float64").gt(14).fillna(False).sum())
            long_14_60_users = int(
                (
                    user_summary["ttl_days"].astype("Float64").gt(14)
                    & user_summary["ttl_days"].astype("Float64").lt(60)
                )
                .fillna(False)
                .sum()
            )
            prior_lock_users = df.loc[
                df[lock_time_col].notna() & df[lock_time_col].lt(start),
                user_col,
            ].astype("string")
            prior_lock_users = set(prior_lock_users.dropna().tolist())
            repeat_lock_users = int(user_summary["user"].astype("string").isin(prior_lock_users).sum())

            lens_out = pd.DataFrame(
                [
                    {"category": "One-Touch (Decisive)", "users": one_touch_users},
                    {"category": "Hesitant (Same Channel, Multiple Touches)", "users": same_channel_multi_users},
                    {"category": "Cross-Channel (Comparison Shopper)", "users": cross_channel_users},
                    {"category": "Long Consideration (>14 Days)", "users": long_users},
                    {"category": "Long Consideration (>14 Days & <60 Days)", "users": long_14_60_users},
                    {"category": "Repeat Lockers (Had Prior Locks)", "users": repeat_lock_users},
                ]
            )
            lens_out["pct"] = lens_out["users"] / max(int(user_summary.shape[0]), 1)
            lens_out["pct"] = lens_out["pct"].map(
                lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
            )
            lens_out_records = lens_out.to_dict(orient="records")

            if cross_channel_users <= 0:
                assist_out = pd.DataFrame(columns=["assist_channel", "assist_touches", "pct"])
            else:
                cross_users = user_summary.loc[user_summary["distinct_channels_to_lock"].gt(1), "user"].copy()
                assist = touches[touches["user"].astype("string").isin(cross_users.astype("string"))].copy()
                assist = assist.merge(lock_time_by_user[["user", "lock_channel", "lock_main_code"]], on="user", how="left")
                assist["assist_channel"] = assist[channel_col].astype("string")
                assist = assist[
                    assist["assist_channel"].notna()
                    & assist["lock_channel"].notna()
                    & assist["lock_main_code"].notna()
                ].copy()
                assist = assist[assist["assist_channel"] != assist["lock_channel"]].copy()
                assist = assist[assist[main_code_col] != assist["lock_main_code"]].copy()

                vc_assist = assist["assist_channel"].value_counts(dropna=False)
                assist_total = int(len(assist))
                top_assist = vc_assist.head(max(int(top_n), 1))
                assist_out = pd.DataFrame(
                    {"assist_channel": top_assist.index.astype("string"), "assist_touches": top_assist.values}
                )
                assist_out["pct"] = assist_out["assist_touches"] / max(assist_total, 1)
                assist_out["pct"] = assist_out["pct"].map(
                    lambda x: _to_percent_1dp(float(x)) if pd.notna(x) else None
                )

            assist_out_records = assist_out.to_dict(orient="records")

    return {
        "锁单用户数": locked_users,
        "数据完整度": _to_percent_1dp(_safe_ratio(locked_users, order_lock_people_non_test_drive)),
        "平均触达次数": (None if touch_mean is None else round(touch_mean, 2)),
        "平均转化时长(天)": (None if ttl_mean is None else round(ttl_mean, 2)),
        "锁单用户主要渠道Top5": channel_top_out,
        "锁单用户分类占比（观察口径）": lens_out_records,
        "跨渠道锁单用户主要助攻渠道Top5": assist_out_records,
    }


def _build_daily_metrics_matrix(days: list[dict[str, object]]) -> dict[str, object]:
    section_order = ["订单分析", "归因分析", "下发线索转化率", "试驾分析"]
    columns = [str(d.get("date")) for d in days if isinstance(d, dict)]
    matrix_values: dict[str, dict[str, object]] = {}
    metric_order: list[str] = []

    def _matrix_cell_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    for day in days:
        if not isinstance(day, dict):
            continue
        date = str(day.get("date"))
        for section_name in section_order:
            section = day.get(section_name)
            if not isinstance(section, dict):
                continue
            for metric_name, value in section.items():
                row_name = f"{section_name}.{metric_name}"
                if row_name not in matrix_values:
                    matrix_values[row_name] = {}
                    metric_order.append(row_name)
                matrix_values[row_name][date] = _matrix_cell_value(value)
    rows = [
        {"metric": metric_name, "values": [matrix_values[metric_name].get(col) for col in columns]}
        for metric_name in metric_order
    ]
    return {
        "columns": columns,
        "rows": rows,
    }


def _write_daily_matrix_csv(matrix: dict[str, object], output_path: Path) -> None:
    columns = matrix.get("columns")
    rows = matrix.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("daily_metrics_matrix 结构不合法，无法导出 CSV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *[str(c) for c in columns]])
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric = row.get("metric")
            values = row.get("values")
            if not isinstance(values, list):
                values = []
            writer.writerow([metric, *values])


def _read_daily_matrix_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {"columns": [], "rows": []}
        columns = [str(c).strip() for c in header[1:]]
        rows: list[dict[str, object]] = []
        for line in reader:
            if not line:
                continue
            metric = str(line[0]).strip()
            values = list(line[1:])
            if len(values) < len(columns):
                values.extend([None] * (len(columns) - len(values)))
            elif len(values) > len(columns):
                values = values[: len(columns)]
            rows.append({"metric": metric, "values": values})
    return {"columns": columns, "rows": rows}


def _parse_matrix_dates(columns: list[str]) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for c in columns:
        try:
            out.append(_parse_target_date(str(c)))
        except Exception:
            continue
    return sorted(set(out))


def _merge_daily_matrices(base: dict[str, object], inc: dict[str, object]) -> dict[str, object]:
    base_cols = [str(c) for c in (base.get("columns") or [])]
    inc_cols = [str(c) for c in (inc.get("columns") or [])]
    merged_cols = base_cols + [c for c in inc_cols if c not in set(base_cols)]

    merged_values: dict[str, dict[str, object]] = {}
    metric_order: list[str] = []

    for matrix in [base, inc]:
        cols = [str(c) for c in (matrix.get("columns") or [])]
        rows = matrix.get("rows") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric = str(row.get("metric") or "").strip()
            if not metric:
                continue
            values = row.get("values")
            if not isinstance(values, list):
                values = []
            if metric not in merged_values:
                merged_values[metric] = {}
                metric_order.append(metric)
            for i, col in enumerate(cols):
                if i >= len(values):
                    continue
                merged_values[metric][col] = values[i]

    removed_metrics = {
        "订单分析.当日锁单退订数",
        "订单分析.锁单数（order_type =用户车）",
        "订单分析.share_l6",
        "订单分析.share_ls6",
        "订单分析.share_ls9",
    }
    merged_rows = [
        {"metric": m, "values": [merged_values[m].get(c) for c in merged_cols]}
        for m in metric_order
        if m not in removed_metrics
    ]
    return {"columns": merged_cols, "rows": merged_rows}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "schema").exists():
            return p
    return here.parent


def _default_range_csv_path(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Path:
    repo_root = _find_repo_root()
    return repo_root / "out" / f"index_summary_daily_matrix_{start_date.date()}_{end_date.date()}.csv"


def _default_maintenance_csv_path() -> Path:
    repo_root = _find_repo_root()
    return repo_root / "schema" / "index_summary_daily_matrix_2024-01-01_to_yesterday.csv"


def main() -> None:
    repo_root = _find_repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--include-days", action="store_true")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--data-path-md",
        default=str(repo_root / "schema" / "data_path.md"),
    )
    parser.add_argument("--refresh-days", type=int, default=7)
    args = parser.parse_args()

    start_arg = args.start
    end_arg = args.end
    maintenance_mode = (args.date is None) and (not args.start) and (not args.end)
    if maintenance_mode:
        maintenance_csv_path = _default_maintenance_csv_path()
        if args.csv_out is None:
            args.csv_out = str(maintenance_csv_path)
        else:
            maintenance_csv_path = Path(str(args.csv_out)).expanduser().resolve()

        yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start_arg = "2024-01-01"
        end_arg = str(yesterday.date())
        refresh_days = max(1, int(args.refresh_days or 1))
        full_start = _parse_target_date("2024-01-01")
        if maintenance_csv_path.exists():
            existing_matrix = _read_daily_matrix_csv(maintenance_csv_path)
            hist_dates = _parse_matrix_dates([str(c) for c in (existing_matrix.get("columns") or [])])
            if hist_dates:
                hist_max = min(hist_dates[-1], yesterday)
                refresh_start = hist_max - pd.Timedelta(days=refresh_days - 1)
                if refresh_start < full_start:
                    refresh_start = full_start
                start_arg = str(refresh_start.date())

    is_range = bool(start_arg or end_arg)
    if is_range and (not start_arg or not end_arg):
        raise ValueError("使用日期范围时必须同时提供 --start 与 --end")

    data_paths = _read_data_paths(Path(args.data_path_md))

    order_path = data_paths["订单分析"]
    attribution_path = data_paths.get("锁单归因")
    assign_path = data_paths["下发线索转化率"]
    test_drive_path = data_paths["试驾分析"]

    if not is_range:
        if not args.date:
            raise ValueError("单日模式必须提供 --date")
        target_date, compare_mode = _parse_single_date_args(args.date)

        order_metrics = _calc_order_metrics(order_path, target_date)
        attribution_metrics = (
            None
            if attribution_path is None
            else _calc_attribution_metrics(
                attribution_path=attribution_path,
                target_date=target_date,
                order_lock_people_non_test_drive=int(order_metrics.get("锁单人数（order_type != 试驾车）") or 0),
            )
        )
        assign_metrics = _calc_assign_metrics(assign_path, target_date)

        avg_locks_per_store = _safe_div(
            order_metrics.get("锁单数", 0), order_metrics.get("在营门店数", 0)
        )
        order_metrics["店均锁单数"] = (
            None if avg_locks_per_store is None else round(avg_locks_per_store, 2)
        )

        avg_daily_leads_per_store = _safe_div(
            assign_metrics.get("下发线索数", 0), assign_metrics.get("下发门店数", 0)
        )
        order_metrics["店日均下发线索数"] = (
            None if avg_daily_leads_per_store is None else round(avg_daily_leads_per_store, 2)
        )

        result = {
            "date": str(target_date.date()),
            "订单分析": order_metrics,
            "归因分析": attribution_metrics,
            "下发线索转化率": assign_metrics,
            "试驾分析": _calc_test_drive_metrics(test_drive_path, target_date),
        }
        if compare_mode == "SPLW":
            splw_date = target_date - pd.Timedelta(days=7)
            splw_order_metrics = _calc_order_metrics(order_path, splw_date)
            splw_attribution_metrics = (
                None
                if attribution_path is None
                else _calc_attribution_metrics(
                    attribution_path=attribution_path,
                    target_date=splw_date,
                    order_lock_people_non_test_drive=int(
                        splw_order_metrics.get("锁单人数（order_type != 试驾车）") or 0
                    ),
                )
            )
            splw_assign_metrics = _calc_assign_metrics(assign_path, splw_date)
            splw_avg_locks_per_store = _safe_div(
                splw_order_metrics.get("锁单数", 0), splw_order_metrics.get("在营门店数", 0)
            )
            splw_order_metrics["店均锁单数"] = (
                None if splw_avg_locks_per_store is None else round(splw_avg_locks_per_store, 2)
            )
            splw_avg_daily_leads_per_store = _safe_div(
                splw_assign_metrics.get("下发线索数", 0), splw_assign_metrics.get("下发门店数", 0)
            )
            splw_order_metrics["店日均下发线索数"] = (
                None if splw_avg_daily_leads_per_store is None else round(splw_avg_daily_leads_per_store, 2)
            )
            result["compare"] = {
                "type": "SPLW",
                "date": str(splw_date.date()),
                "订单分析": splw_order_metrics,
                "归因分析": splw_attribution_metrics,
                "下发线索转化率": splw_assign_metrics,
                "试驾分析": _calc_test_drive_metrics(test_drive_path, splw_date),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    start_date = _parse_target_date(str(start_arg))
    end_date = _parse_target_date(str(end_arg))
    dates = _iter_days(start_date, end_date)
    if not dates and maintenance_mode:
        csv_out_path = Path(str(args.csv_out)).expanduser().resolve()
        print(str(csv_out_path))
        return
    if not dates:
        raise ValueError("日期范围为空，请检查 --start/--end")

    order_df = _load_order_df(order_path)
    assign_df = _load_assign_df(assign_path)
    test_drive_df = _load_test_drive_df(test_drive_path)
    attr_df = None
    attr_cols = None
    if attribution_path is not None:
        df_prepared, cols_map = _prepare_attribution_df(attribution_path)
        attr_df = df_prepared
        attr_cols = cols_map

    day_results: list[dict[str, object]] = []
    for d in dates:
        order_metrics = _calc_order_metrics_from_df(order_df, d)
        assign_metrics = _calc_assign_metrics_from_df(assign_df, d)
        avg_locks_per_store = _safe_div(
            order_metrics.get("锁单数", 0), order_metrics.get("在营门店数", 0)
        )
        order_metrics["店均锁单数"] = (
            None if avg_locks_per_store is None else round(avg_locks_per_store, 2)
        )
        avg_daily_leads_per_store = _safe_div(
            assign_metrics.get("下发线索数", 0), assign_metrics.get("下发门店数", 0)
        )
        order_metrics["店日均下发线索数"] = (
            None if avg_daily_leads_per_store is None else round(avg_daily_leads_per_store, 2)
        )

        attribution_metrics = None
        if attr_df is not None and attr_cols is not None:
            attribution_metrics = _calc_attribution_metrics_from_df(
                df=attr_df,
                cols_map=attr_cols,
                target_date=d,
                order_lock_people_non_test_drive=int(order_metrics.get("锁单人数（order_type != 试驾车）") or 0),
            )

        day_results.append(
            {
                "date": str(d.date()),
                "订单分析": order_metrics,
                "归因分析": attribution_metrics,
                "下发线索转化率": assign_metrics,
                "试驾分析": _calc_test_drive_metrics_from_df(test_drive_df, d),
            }
        )

    out = {
        "start": str(start_date.date()),
        "end": str(end_date.date()),
        "days": (day_results if args.include_days else None),
        "daily_metrics_matrix": _build_daily_metrics_matrix(day_results),
    }
    if not args.include_days:
        out.pop("days", None)
    csv_out_path = (
        Path(str(args.csv_out)).expanduser().resolve()
        if args.csv_out
        else _default_range_csv_path(start_date, end_date)
    )
    matrix_to_write = out["daily_metrics_matrix"]
    if maintenance_mode and csv_out_path.exists():
        history_matrix = _read_daily_matrix_csv(csv_out_path)
        matrix_to_write = _merge_daily_matrices(history_matrix, matrix_to_write)
    _write_daily_matrix_csv(matrix=matrix_to_write, output_path=csv_out_path)
    if args.print_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(str(csv_out_path))


if __name__ == "__main__":
    main()
