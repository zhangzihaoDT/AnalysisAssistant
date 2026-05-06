import contextlib
import io
import json
import os
import re
import glob
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
import warnings


def _safe_read_csv(file_path: Path) -> pd.DataFrame:
    candidates = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-16", "sep": "\t"},
        {"encoding": "utf-16le", "sep": "\t"},
        {"encoding": "gbk", "sep": ","},
        {"encoding": "gb18030", "sep": ","},
    ]
    for option in candidates:
        try:
            return pd.read_csv(file_path, low_memory=False, **option)
        except Exception:
            continue
    return pd.read_csv(file_path, low_memory=False)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _try_parse_datetime_series(series: pd.Series) -> pd.Series | None:
    s = series
    if not isinstance(s, pd.Series):
        return None
    if s.empty:
        return None

    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    raw = s.astype(str)
    raw_stripped = raw.str.strip()
    parsed_cn = pd.to_datetime(raw, errors="coerce", format="%Y年%m月%d日")
    if float(parsed_cn.notna().mean()) >= 0.8:
        return parsed_cn

    iso_date_ratio = float(raw_stripped.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False).mean())
    if iso_date_ratio >= 0.8:
        parsed_iso_date = pd.to_datetime(raw_stripped, errors="coerce", format="%Y-%m-%d")
        if float(parsed_iso_date.notna().mean()) >= 0.8:
            return parsed_iso_date

    iso_dt_raw = raw_stripped.str.replace("T", " ", regex=False)
    iso_dt_ratio = float(
        iso_dt_raw.str.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(\.\d+)?$", na=False).mean()
    )
    if iso_dt_ratio >= 0.8:
        parsed_iso_dt_base = pd.to_datetime(iso_dt_raw, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        parsed_iso_dt_frac = pd.to_datetime(iso_dt_raw, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
        parsed_iso_dt = parsed_iso_dt_base.fillna(parsed_iso_dt_frac)
        if float(parsed_iso_dt.notna().mean()) >= 0.8:
            return parsed_iso_dt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        parsed_any = pd.to_datetime(raw_stripped, errors="coerce")
    if float(parsed_any.notna().mean()) >= 0.8:
        return parsed_any

    return None


def _try_parse_datetime_value(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, str) and _ISO_DATE_RE.match(value.strip()):
        try:
            return pd.to_datetime(value.strip(), errors="raise")
        except Exception:
            return None
    return None


def _try_parse_numeric_series(series: pd.Series) -> pd.Series | None:
    if not isinstance(series, pd.Series) or series.empty:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    raw = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    parsed = pd.to_numeric(raw, errors="coerce")
    if float(parsed.notna().mean()) >= 0.8:
        return parsed
    return None


class QueryTool:
    def __init__(self, data_path_file: str, schema_dir: str):
        self.data_path_file = Path(data_path_file)
        self.schema_dir = Path(schema_dir)
        self.datasets: dict[str, pd.DataFrame] = {}
        self._loaded = False
        self._business_definition: dict[str, Any] = {}
        try:
            bdef_path = self.schema_dir / "business_definition.json"
            if bdef_path.exists():
                self._business_definition = json.loads(bdef_path.read_text(encoding="utf-8"))
        except Exception:
            self._business_definition = {}

    def _parse_dataset_paths(self) -> list[Path]:
        if not self.data_path_file.exists():
            return []
        paths: list[Path] = []
        for raw in self.data_path_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            value = line
            if "：" in value:
                value = value.split("：", 1)[1].strip()
            elif ":" in value and not value.startswith(("/", "~")) and not re.match(r"^[A-Za-z]:[\\/]", value):
                value = value.split(":", 1)[1].strip()

            value = value.strip().strip("`").strip('"').strip("'")
            value = os.path.expandvars(value)
            value = value.replace("\\_", "_").replace("\\*", "*")

            expanded = glob.glob(str(Path(value).expanduser()))
            if expanded:
                for p in sorted(expanded, key=lambda x: (len(x), x)):
                    paths.append(Path(p))
            else:
                paths.append(Path(value).expanduser())
        return paths

    def _load_datasets(self) -> None:
        if self._loaded:
            return
        for file_path in self._parse_dataset_paths():
            if not file_path.exists():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif suffix == ".csv":
                df = _safe_read_csv(file_path)
            else:
                continue
            stem_key = file_path.stem
            name_key = file_path.name
            self.datasets[stem_key] = df
            self.datasets[name_key] = df
        self._loaded = True

    def _schema_context(self) -> dict[str, str]:
        schema: dict[str, str] = {}
        schema_file = self.schema_dir / "schema.md"
        business_file = self.schema_dir / "business_definition.json"
        schema["schema_md"] = schema_file.read_text(encoding="utf-8") if schema_file.exists() else ""
        schema["business_definition"] = business_file.read_text(encoding="utf-8") if business_file.exists() else ""
        return schema

    def answer_question(self, question: str) -> str:
        """Fallback simple keyword matching"""
        self._load_datasets()
        lowered = question.strip()
        if not lowered:
            return "问题为空，请提供要查询的指标问题。"
        for dataset_name, df in self.datasets.items():
            if "." in dataset_name:
                continue
            for column in df.columns:
                if column in lowered:
                    series = df[column]
                    if "平均" in lowered:
                        return f"{dataset_name}.{column} 平均值: {series.mean()}"
                    if "最大" in lowered:
                        return f"{dataset_name}.{column} 最大值: {series.max()}"
                    if "最小" in lowered:
                        return f"{dataset_name}.{column} 最小值: {series.min()}"
                    if "总和" in lowered or "合计" in lowered:
                        return f"{dataset_name}.{column} 总和: {series.sum()}"
                    if "中位" in lowered:
                        return f"{dataset_name}.{column} 中位数: {series.median()}"
                    if "多少" in lowered or "数量" in lowered or "总数" in lowered:
                        return f"{dataset_name}.{column} 非空数量: {series.count()}"
        return "未能从问题中匹配到字段，请在问题中包含明确字段名。"

    def execute_analysis_df(self, plan: dict) -> Union[pd.DataFrame, str]:
        print(f"[QueryTool] 接收到 DSL 计划: {json.dumps(plan, ensure_ascii=False)}")

        self._load_datasets()
        dataset_name = plan.get("dataset")
        if not dataset_name or dataset_name not in self.datasets:
            if not self.datasets:
                return "未加载到任何数据集。"
            if not dataset_name:
                if "assign_data" in self.datasets:
                    dataset_name = "assign_data"
                elif "order_data" in self.datasets:
                    dataset_name = "order_data"
                else:
                    dataset_name = list(self.datasets.keys())[0]
            elif dataset_name not in self.datasets:
                return f"找不到数据集: {dataset_name}。可用数据集: {list(self.datasets.keys())}"

        df = self.datasets[dataset_name].copy()

        filters = plan.get("filters", [])
        needs_series_group = any(
            isinstance(f, dict) and f.get("field") == "series_group_logic" for f in (filters or [])
        )
        dimensions = plan.get("dimensions", [])
        if not isinstance(dimensions, list):
            dimensions = []
        if not needs_series_group and isinstance(dimensions, list):
            needs_series_group = "series_group_logic" in dimensions
        if needs_series_group and "series_group_logic" not in df.columns:
            try:
                from operators.series_group_logic import apply_series_group_logic

                df = apply_series_group_logic(df=df, business_definition=self._business_definition)
            except Exception:
                pass

        for f in filters:
            field = f.get("field")
            op = f.get("op")
            value = f.get("value")
            if field not in df.columns:
                continue

            if op in {">", "<", ">=", "<="}:
                parsed_series = _try_parse_datetime_series(df[field])
                parsed_value = _try_parse_datetime_value(value)
                if parsed_series is not None and parsed_value is not None:
                    s = parsed_series
                    v = parsed_value
                    if op == ">":
                        df = df[s > v]
                    elif op == "<":
                        df = df[s < v]
                    elif op == ">=":
                        df = df[s >= v]
                    elif op == "<=":
                        df = df[s <= v]
                    continue

            if op in {"==", "!="}:
                parsed_series = _try_parse_datetime_series(df[field])
                parsed_value = _try_parse_datetime_value(value)
                if parsed_series is not None and parsed_value is not None:
                    s = parsed_series
                    v = parsed_value
                    if op == "==":
                        df = df[s == v]
                    else:
                        df = df[s != v]
                    continue

            if op == "==":
                if value is None:
                    df = df[df[field].isna()]
                else:
                    df = df[df[field] == value]
            elif op == "!=":
                if value is None:
                    df = df[df[field].notna()]
                else:
                    df = df[df[field] != value]
            elif op == ">":
                df = df[df[field] > value]
            elif op == "<":
                df = df[df[field] < value]
            elif op == ">=":
                df = df[df[field] >= value]
            elif op == "<=":
                df = df[df[field] <= value]
            elif op == "in" and isinstance(value, list):
                df = df[df[field].isin(value)]
            elif op == "contains":
                df = df[df[field].astype(str).str.contains(str(value), na=False, regex=False)]
            elif op == "not contains":
                df = df[~df[field].astype(str).str.contains(str(value), na=False, regex=False)]
            elif op == "matches":
                df = df[df[field].astype(str).str.contains(str(value), na=False, regex=True)]
            elif op == "not matches":
                df = df[~df[field].astype(str).str.contains(str(value), na=False, regex=True)]

        metrics = plan.get("metrics", [])

        result_df = df

        if metrics:
            agg_dict: dict[str, Any] = {}
            alias_by_field: dict[str, str] = {}

            for m in metrics:
                field = m.get("field")
                agg_func = m.get("agg", "count")
                alias = m.get("alias")

                if agg_func == "count" and field == "*":
                    field = df.columns[0]

                if not field or field not in df.columns:
                    continue

                if isinstance(agg_func, str) and agg_func in {"sum", "mean", "min", "max"}:
                    numeric_series = _try_parse_numeric_series(df[field])
                    if numeric_series is not None:
                        df[field] = numeric_series

                if field in agg_dict:
                    existing = agg_dict[field]
                    if isinstance(existing, list):
                        existing.append(agg_func)
                    else:
                        agg_dict[field] = [existing, agg_func]
                else:
                    agg_dict[field] = agg_func

                if isinstance(agg_func, str) and alias:
                    alias_by_field[field] = alias

            if dimensions:
                valid_dims = [d for d in dimensions if d in df.columns]
                if valid_dims:
                    try:
                        result_df = df.groupby(valid_dims).agg(agg_dict).reset_index()
                    except Exception as e:
                        return f"聚合计算失败: {e}"
                else:
                    result_df = df.agg(agg_dict).to_frame().T
            else:
                try:
                    result_df = df.agg(agg_dict).to_frame().T
                except Exception as e:
                    return f"聚合计算失败: {e}"

            if alias_by_field:
                rename_map = {field: alias for field, alias in alias_by_field.items() if field in result_df.columns}
                if rename_map:
                    result_df = result_df.rename(columns=rename_map)

        sort_opts = plan.get("sort", [])
        if sort_opts and not result_df.empty:
            if isinstance(sort_opts, dict):
                sort_opts = [sort_opts]

            by = []
            ascending = []
            for s in sort_opts:
                field = s.get("field")
                if field in result_df.columns:
                    by.append(field)
                    ascending.append(s.get("order", "asc") == "asc")

            if by:
                result_df = result_df.sort_values(by=by, ascending=ascending)

        limit = plan.get("limit")
        if limit:
            result_df = result_df.head(int(limit))

        return result_df

    def execute_analysis(self, plan: dict) -> str:
        result_df = self.execute_analysis_df(plan)
        if isinstance(result_df, str):
            return result_df
        return result_df.to_string(index=False)


QUERY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "perform_analysis",
        "description": "执行 BI 数据分析。根据 DSL 计划执行数据查询、聚合、排序等操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "BI 分析计划 (DSL)",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "数据集名称 (e.g. 'assign_data', 'order_data')"
                        },
                        "metrics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "agg": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                                    "alias": {"type": "string"}
                                },
                                "required": ["field", "agg"]
                            }
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "op": {"type": "string", "enum": ["==", "!=", ">", "<", ">=", "<=", "in", "contains", "not contains", "matches", "not matches"]},
                                    "value": {} 
                                },
                                "required": ["field", "op", "value"]
                            }
                        },
                        "sort": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "order": {"type": "string", "enum": ["asc", "desc"]}
                                }
                            }
                        },
                        "limit": {"type": "integer"}
                    },
                    "required": ["dataset", "metrics"]
                }
            },
            "required": ["plan"]
        }
    }
}
