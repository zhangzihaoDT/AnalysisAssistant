import json
import datetime
import re

from openai import OpenAI
from agent.llm_config import DEEPSEEK_PLANNER_MODEL
from agent.state import AgentRuntimeState

LOOP_RUNTIME_SYSTEM_PROMPT = """
你是一个数据分析 Agent Loop 调度器。
你需要根据目标与历史执行结果，决定下一步动作，并且必须输出 JSON。

输出格式:
{
  "action": "run_dsl 或 finish",
  "reason": "为什么这样决策",
  "query": "下一步要执行的自然语言查询（action=run_dsl 时必填）",
  "analysis": "你对当前进展的理解"
}

规则:
1. 如果信息还不足以回答用户目标，输出 run_dsl。
2. 如果信息已足够，输出 finish。
3. 最多执行 5 步，避免重复查询。
4. query 必须具体，且与目标直接相关。
5. 不允许输出除 JSON 以外的文本。
6. 若需要拆分时间区间多次查询，必须从历史 DSL 中读取实际查询窗口（filters/time），并确保窗口按 [start, end)（左闭右开）无重叠、无漏数：下一段 start 应等于上一段 end。
"""


PLANNING_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_planning_dsl",
        "description": "将自然语言问题转为规划 DSL（可拆解为多个子问题对应多个 plan）。",
        "parameters": {
            "type": "object",
            "properties": {
                "plans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "dataset": {"type": "string"},
                            "metric": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "agg": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                                    "alias": {"type": "string"},
                                    "business_name": {"type": "string"},
                                },
                                "required": ["field", "agg"],
                            },
                            "time": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "start": {"type": "string"},
                                    "end": {"type": "string"},
                                },
                                "required": ["field", "start", "end"],
                            },
                            "dimensions": {"type": "array", "items": {"type": "string"}},
                            "filters": {"type": "array"},
                            "comparison": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["none", "yoy", "wow", "dod"]},
                                },
                                "required": ["type"],
                            },
                            "ranking": {
                                "type": "object",
                                "properties": {
                                    "order": {"type": "string", "enum": ["asc", "desc"]},
                                    "top_k": {"type": "integer"},
                                },
                            },
                            "statistics": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "weekly_decline_ratio",
                                            "daily_threshold_count",
                                            "daily_mean",
                                            "daily_percentile_rank",
                                            "weekend_percentile_rank",
                                            "weekday_percentile_rank",
                                        ],
                                    },
                                    "time_field": {"type": "string"},
                                    "window_weeks": {"type": "integer"},
                                    "window_days": {"type": "integer"},
                                    "window_weekends": {"type": "integer"},
                                    "reference_date": {"type": "string"},
                                    "weekdays": {"type": "array", "items": {"type": "integer"}},
                                    "op": {"type": "string", "enum": [">", ">=", "<", "<=", "==", "!="]},
                                    "threshold": {"type": "number"},
                                    "numerator_metric": {
                                        "type": "object",
                                        "properties": {
                                            "field": {"type": "string"},
                                            "agg": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                                            "alias": {"type": "string"},
                                        },
                                    },
                                    "denominator_metric": {
                                        "type": "object",
                                        "properties": {
                                            "field": {"type": "string"},
                                            "agg": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                                            "alias": {"type": "string"},
                                        },
                                    },
                                    "value_metric": {
                                        "type": "object",
                                        "properties": {
                                            "field": {"type": "string"},
                                            "agg": {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]},
                                            "alias": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "fast_path": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["numeric_ratio", "current_iso_week", "small_talk_contextual"]},
                                    "current": {"type": "number"},
                                    "base": {"type": "number"},
                                },
                            },
                        },
                        "required": ["dataset", "metric", "time", "comparison"],
                    },
                },
                "clarification": {
                    "type": "object",
                    "properties": {
                        "need": {"type": "boolean"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "object"},
                    },
                },
            },
            "required": ["plans"],
        },
    },
}

TIME_REWRITE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rewrite_time_window",
        "description": "重写时间窗口（左闭右开），仅返回 start/end（YYYY-MM-DD）。",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
            "required": ["start", "end"],
        },
    },
}


class PlanningAgent:
    _SERIES_CANDIDATES = ("LS9", "LS8", "LS7", "LS6", "L7", "L6")
    _TIME_REWRITE_WHITELIST = {
        "yesterday",
        "this_month_to_today",
        "last_month",
        "month_to_today",
        "month",
        "year",
        "date",
        "date_range",
    }

    def __init__(self, client: OpenAI, schema_md: str, business_definition: str):
        self.client = client
        self.schema_md = schema_md or ""
        self.business_definition = business_definition or ""

    @staticmethod
    def _parse_comparison_type(user_query: str) -> str:
        q = user_query or ""
        if "日环比" in q or "天环比" in q:
            return "dod"
        if ("昨天" in q or "昨日" in q or "今天" in q or "今日" in q) and ("环比" in q) and ("周环比" not in q):
            return "dod"
        if "同比" in q or "年同比" in q:
            return "yoy"
        if "环比" in q or "周环比" in q:
            return "wow"
        return "none"

    @staticmethod
    def _classify_intent(user_query: str) -> str:
        q = (user_query or "").strip()
        if not q:
            return "query"

        stat_keywords = [
            "有多少天",
            "多少天",
            "几天",
            "多少周",
            "几周",
            "连续",
            "超过",
            "大于",
            "小于",
            "高于",
            "低于",
            "阈值",
            "下降",
            "日均",
            "均值",
            "平均值",
            "平均",
            "分位",
            "百分位",
            "处于什么水平",
            "什么水平",
        ]
        has_stat_keyword = any(k in q for k in stat_keywords)
        has_explicit_window = PlanningAgent._parse_time_window(q, datetime.date.today()) is not None
        has_time_window = has_explicit_window or bool(re.search(r"近\s*\d+\s*(日|天|周|月)", q)) or any(
            k in q for k in ["昨天", "昨日", "本周", "上周", "本月", "上月", "今年", "去年"]
        )
        if has_stat_keyword and has_time_window:
            return "statistics"
        if any(k in q for k in ["同比", "年同比", "环比", "周环比"]):
            return "comparison"
        return "query"

    @staticmethod
    def _parse_time_window(user_query: str, today: datetime.date) -> tuple[str, str] | None:
        q = user_query or ""
        if "昨天" in q or "昨日" in q:
            start = today - datetime.timedelta(days=1)
            end = today
            return (start.isoformat(), end.isoformat())

        def _normalize_year(y: str) -> int | None:
            if not y:
                return None
            y = str(y).strip()
            if not y.isdigit():
                return None
            if len(y) == 2:
                return 2000 + int(y)
            if len(y) == 4:
                return int(y)
            return None

        def _safe_date(y: int, m: int, d: int) -> datetime.date | None:
            try:
                return datetime.date(int(y), int(m), int(d))
            except Exception:
                return None

        def _month_window(year: int, month: int) -> tuple[str, str] | None:
            if month < 1 or month > 12:
                return None
            start = _safe_date(year, month, 1)
            if not start:
                return None
            if month == 12:
                end = _safe_date(year + 1, 1, 1)
            else:
                end = _safe_date(year, month + 1, 1)
            if not end:
                return None
            return (start.isoformat(), end.isoformat())

        def _year_window(year: int) -> tuple[str, str] | None:
            start = _safe_date(year, 1, 1)
            end = _safe_date(year + 1, 1, 1)
            if not start or not end:
                return None
            return (start.isoformat(), end.isoformat())

        def _parse_month_token(token: str) -> int | None:
            if not token:
                return None
            raw = str(token).strip()
            if not raw:
                return None
            if raw.isdigit():
                try:
                    v = int(raw)
                    return v if 1 <= v <= 12 else None
                except Exception:
                    return None
            mapping = {
                "正": 1,
                "一": 1,
                "二": 2,
                "两": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
                "十一": 11,
                "十二": 12,
            }
            if raw in mapping:
                return mapping[raw]
            return None

        if "本月" in q:
            start = _safe_date(today.year, today.month, 1)
            if start:
                return (start.isoformat(), today.isoformat())
        if "上月" in q:
            year = today.year
            month = today.month - 1
            if month <= 0:
                year -= 1
                month = 12
            window = _month_window(year, month)
            if window:
                return window

        m = re.search(
            r"(?:(?P<y>\d{2,4})\s*年\s*)?(?P<m>\d{1,2}|正|十一|十二|[一二两三四五六七八九十])\s*月\s*"
            r"(?:(?:到|至|[-~—–－])\s*)?(?:至今|到今|现在|目前|今天|今日|截至今日|截至今天|截至昨日|昨日)",
            q,
        )
        if m:
            year = _normalize_year(m.group("y")) or today.year
            month = _parse_month_token(m.group("m")) or today.month
            start = _safe_date(year, month, 1)
            if start:
                return (start.isoformat(), today.isoformat())

        if "前年" in q:
            window = _year_window(today.year - 2)
            if window:
                return window
        if "去年" in q:
            window = _year_window(today.year - 1)
            if window:
                return window
        if "今年" in q:
            window = _year_window(today.year)
            if window:
                return window

        m = re.search(
            r"(?P<y>\d{2,4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]?\s*(?:到|至|[-~—–－])\s*"
            r"(?:(?P<y2>\d{2,4})\s*年\s*)?(?:(?P<m2>\d{1,2})\s*月\s*)?(?P<d2>\d{1,2})\s*[日号]?",
            q,
        )
        if m:
            y1 = _normalize_year(m.group("y")) or today.year
            m1 = int(m.group("m"))
            d1 = int(m.group("d"))
            y2 = _normalize_year(m.group("y2")) or y1
            m2 = int(m.group("m2") or m1)
            d2 = int(m.group("d2"))
            start_date = _safe_date(y1, m1, d1)
            end_date = _safe_date(y2, m2, d2)
            if start_date and end_date:
                if "留存" in q or "预售期" in q:
                    return (start_date.isoformat(), end_date.isoformat())
                end_open = end_date + datetime.timedelta(days=1)
                return (start_date.isoformat(), end_open.isoformat())

        m = re.search(
            r"(?P<y>\d{2,4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?:整月|全月|整个月)",
            q,
        )
        if m:
            year = _normalize_year(m.group("y")) or today.year
            month = int(m.group("m"))
            window = _month_window(year, month)
            if window:
                return window

        m = re.search(r"(?P<y>\d{2,4})\s*年\s*(?P<m>\d{1,2})\s*月(?!\s*\d)", q)
        if m:
            year = _normalize_year(m.group("y")) or today.year
            month = int(m.group("m"))
            window = _month_window(year, month)
            if window:
                return window

        m = re.search(r"(?:(?P<y>\d{2,4})\s*年\s*)?(?P<m>正|十一|十二|[一二两三四五六七八九十])\s*月(?!\s*\d)", q)
        if m:
            year = _normalize_year(m.group("y")) or today.year
            month = _parse_month_token(m.group("m"))
            if month:
                window = _month_window(year, month)
                if window:
                    return window

        m = re.search(r"(?P<y>\d{2,4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]?", q)
        if m:
            year = _normalize_year(m.group("y")) or today.year
            month = int(m.group("m"))
            day = int(m.group("d"))
            start_date = _safe_date(year, month, day)
            if start_date:
                end_open = start_date + datetime.timedelta(days=1)
                return (start_date.isoformat(), end_open.isoformat())

        m = re.search(r"(?P<y>\d{2,4})年(?!\s*\d|\s*月)", q)
        if m:
            year = _normalize_year(m.group("y")) or today.year
            window = _year_window(year)
            if window:
                return window

        m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:到|至|[-~—–－])\s*(\d{4}-\d{2}-\d{2})", q)
        if m:
            start_date = datetime.date.fromisoformat(m.group(1))
            end_date = datetime.date.fromisoformat(m.group(2))
            if "留存" in q or "预售期" in q:
                return (start_date.isoformat(), end_date.isoformat())
            end_open = end_date + datetime.timedelta(days=1)
            return (start_date.isoformat(), end_open.isoformat())

        m = re.search(
            r"(?P<y>\d{4})-(?P<m1>\d{1,2})-(?P<d1>\d{1,2})\s*(?:到|至|[-~—–－])\s*(?P<m2>\d{1,2})-(?P<d2>\d{1,2})",
            q,
        )
        if m:
            year = int(m.group("y"))
            start_date = _safe_date(year, int(m.group("m1")), int(m.group("d1")))
            end_date = _safe_date(year, int(m.group("m2")), int(m.group("d2")))
            if start_date and end_date:
                if "留存" in q or "预售期" in q:
                    return (start_date.isoformat(), end_date.isoformat())
                end_open = end_date + datetime.timedelta(days=1)
                return (start_date.isoformat(), end_open.isoformat())

        m = re.search(r"(\d{4}-\d{2}-\d{2})", q)
        if m:
            start = datetime.date.fromisoformat(m.group(1))
            end = start + datetime.timedelta(days=1)
            return (start.isoformat(), end.isoformat())

        return None

    @staticmethod
    def _metric_defaults(user_query: str) -> dict | None:
        q = user_query or ""
        if "在营门店" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "store_name", "agg": "count", "alias": "在营门店数", "business_name": "在营门店数"},
                "time_field": "order_create_date",
                "non_null_field": "order_create_date",
            }
        if "锁单" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "order_number", "agg": "count", "alias": "锁单数", "business_name": "锁单量"},
                "time_field": "lock_time",
                "non_null_field": "lock_time",
            }
        if "交付" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "order_number", "agg": "count", "alias": "交付数", "business_name": "交付数"},
                "time_field": "delivery_date",
                "non_null_field": "delivery_date",
            }
        if "开票金额" in q or ("开票" in q and "金额" in q):
            return {
                "dataset": "order_data",
                "metric": {"field": "invoice_amount", "agg": "sum", "alias": "开票金额", "business_name": "开票金额"},
                "time_field": "invoice_upload_time",
                "non_null_field": "invoice_upload_time",
            }
        if "开票" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "order_number", "agg": "count", "alias": "开票数", "business_name": "开票数"},
                "time_field": "invoice_upload_time",
                "non_null_field": "invoice_upload_time",
            }
        if "小订" in q or "意向金" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "order_number", "agg": "count", "alias": "小订数", "business_name": "小订数"},
                "time_field": "intention_payment_time",
                "non_null_field": "intention_payment_time",
            }
        if "大定" in q or "定金" in q:
            return {
                "dataset": "order_data",
                "metric": {"field": "order_number", "agg": "count", "alias": "大定数", "business_name": "大定数"},
                "time_field": "deposit_payment_time",
                "non_null_field": "deposit_payment_time",
            }
        return None

    @staticmethod
    def _parse_fast_path_query(user_query: str) -> dict | None:
        q = (user_query or "").strip()
        if not q:
            return None
        q_lower = q.lower()
        if any(k in q_lower for k in ["welldone", "well done", "good job", "nice", "great"]) or any(
            k in q for k in ["你好", "您好", "在吗", "辛苦了", "谢谢", "感谢", "赞", "太棒了", "干得好", "厉害"]
        ):
            return {"type": "small_talk_contextual"}
        iso_week_hint = any(k in q for k in ["第几周", "ISO周", "ISO 周", "isoweek", "iso week"])
        has_today = any(k in q for k in ["今天", "今日", "当前日期"])
        if iso_week_hint and has_today:
            return {"type": "current_iso_week"}
        if any(k in q for k in ["锁单", "交付", "开票", "门店", "线索", "试驾", "订单", "在营"]):
            return None
        has_compare_intent = any(k in q for k in ["环比", "同比", "提升", "增长", "下降", "减少", "涨幅", "降幅", "相比", "较", "比"])
        has_ask = any(k in q for k in ["多少", "几", "百分比", "%", "百分点"])
        if not (has_compare_intent and has_ask):
            return None
        nums = re.findall(r"-?\d+(?:\.\d+)?", q.replace(",", ""))
        if len(nums) < 2:
            return None
        try:
            current = float(nums[0])
            base = float(nums[1])
        except Exception:
            return None
        return {"type": "numeric_ratio", "current": current, "base": base}

    @staticmethod
    def _infer_series_tokens(user_query: str) -> list[str]:
        q = (user_query or "").upper()
        tokens: list[str] = []
        for s in PlanningAgent._SERIES_CANDIDATES:
            if s in q:
                tokens.append(s)
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _has_field_filter(filters: list, fields: set[str]) -> bool:
        for f in filters:
            if not isinstance(f, dict):
                continue
            if str(f.get("field") or "") in fields:
                return True
        return False

    @staticmethod
    def _apply_semantic_filters(filters: list, user_query: str) -> list:
        q = (user_query or "").replace(" ", "")
        has_trial_metric_context = any(k in q for k in ["试驾率", "试驾数", "有效试驾"])
        ask_trial_car = (("试驾车" in q) or ("试驾" in q and not has_trial_metric_context)) and not any(
            k in q for k in ["非试驾车", "不是试驾车", "排除试驾", "不含试驾", "剔除试驾"]
        )
        ask_non_trial_car = any(k in q for k in ["非试驾车", "不是试驾车", "排除试驾", "不含试驾", "剔除试驾"])
        ask_user_car = "用户车" in q

        has_order_type_filter = PlanningAgent._has_field_filter(filters, {"order_type"})
        if not has_order_type_filter:
            if ask_trial_car and not ask_user_car:
                filters.append({"field": "order_type", "op": "==", "value": "试驾车"})
            elif ask_user_car and not ask_trial_car:
                filters.append({"field": "order_type", "op": "==", "value": "用户车"})
            elif ask_non_trial_car:
                filters.append({"field": "order_type", "op": "!=", "value": "试驾车"})

        series_tokens = PlanningAgent._infer_series_tokens(q)
        has_series_filter = PlanningAgent._has_field_filter(
            filters, {"series", "product_name", "drive_series_cn", "belong_intent_series"}
        )
        if (not has_series_filter) and series_tokens:
            if len(series_tokens) == 1:
                filters.append({"field": "series", "op": "==", "value": series_tokens[0]})
            elif len(series_tokens) > 1:
                filters.append({"field": "series", "op": "in", "value": series_tokens})

        return filters

    @staticmethod
    def _should_sales_clarify(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        if not any(k in q for k in ["销量", "卖了多少", "成交量"]):
            return False
        if any(k in q for k in ["锁单", "交付", "开票"]):
            return False
        return True

    @staticmethod
    def _sales_clarification(original_question: str) -> dict:
        return {
            "need": True,
            "question": "你提到的“销量”具体是指哪个业务口径？",
            "options": ["锁单量", "交付数", "开票数"],
            "context": {"original_question": original_question},
        }

    @staticmethod
    def _extract_city_token(user_query: str) -> str | None:
        q = (user_query or "").strip().lstrip(" \t\r\n\"'“”‘’")
        if not q:
            return None
        stop = {"查询", "统计", "汇总", "查看", "分析", "对比", "输出", "导出", "列出", "展示", "打印", "生成"}
        m = re.search(r"([\u4e00-\u9fff]{2,8})市", q)
        if m:
            city = (m.group(1) or "").strip()
            if not city or city in stop:
                return None
            if any(bad in city for bad in stop):
                return None
            return city
        m = re.search(r"^([\u4e00-\u9fff]{2,8})(?=(昨天|昨日|今日|今天|本周|上周|本月|上月|今年|去年|前年|\d{2,4}年))", q)
        if m:
            city = (m.group(1) or "").strip()
            if not city or city in stop:
                return None
            if any(bad in city for bad in stop):
                return None
            return city
        return None

    @staticmethod
    def _should_city_clarify(user_query: str) -> tuple[bool, str | None]:
        q = user_query or ""
        if not q:
            return (False, None)
        if any(k in q for k in ["门店城市", "store_city", "上牌城市", "license_city"]):
            return (False, None)
        if not any(k in q for k in ["锁单", "交付", "开票", "小订", "意向金", "订单"]):
            return (False, None)
        city = PlanningAgent._extract_city_token(q)
        if not city:
            return (False, None)
        return (True, city)

    @staticmethod
    def _city_clarification(city: str, original_question: str) -> dict:
        return {
            "need": True,
            "question": f"你问的“{city}”是指门店城市(store_city)，还是上牌城市(license_city)？",
            "options": ["门店城市", "上牌城市", "两者都要"],
            "context": {"city": city, "original_question": original_question},
        }

    @staticmethod
    def _append_filter(filters: list[dict], new_filter: dict) -> list[dict]:
        if not isinstance(new_filter, dict):
            return filters
        field = new_filter.get("field")
        op = new_filter.get("op")
        value = new_filter.get("value", None)
        for f in filters:
            if not isinstance(f, dict):
                continue
            if f.get("field") == field and f.get("op") == op and f.get("value", None) == value:
                return filters
        return [*filters, new_filter]

    @staticmethod
    def _parse_recent_weeks(user_query: str) -> int | None:
        q = user_query or ""
        m = re.search(r"近\s*(\d{1,3})\s*(?:个)?\s*周", q)
        if not m:
            return None
        try:
            v = int(m.group(1))
            return v if v > 0 else None
        except Exception:
            return None

    @staticmethod
    def _parse_recent_days(user_query: str) -> int | None:
        q = user_query or ""
        m = re.search(r"近\s*(\d{1,3})\s*(?:日|天)", q)
        if not m:
            return None
        try:
            v = int(m.group(1))
            return v if v > 0 else None
        except Exception:
            return None

    @staticmethod
    def _parse_recent_weekends(user_query: str) -> int | None:
        q = user_query or ""
        m = re.search(r"近\s*(\d{1,3})\s*(?:个)?\s*周末", q)
        if not m:
            return None
        try:
            v = int(m.group(1))
            return v if v > 0 else None
        except Exception:
            return None

    @staticmethod
    def _parse_threshold_condition(user_query: str) -> tuple[str, float] | None:
        q = (user_query or "").replace(" ", "")
        pattern_map = [
            (r"(?:大于等于|不低于|不少于|>=)\s*(\d+(?:\.\d+)?)", ">="),
            (r"(?:小于等于|不高于|不大于|<=)\s*(\d+(?:\.\d+)?)", "<="),
            (r"(?:大于|高于|超过|>)\s*(\d+(?:\.\d+)?)", ">"),
            (r"(?:小于|低于|<)\s*(\d+(?:\.\d+)?)", "<"),
            (r"(?:等于|==)\s*(\d+(?:\.\d+)?)", "=="),
            (r"(?:不等于|!=)\s*(\d+(?:\.\d+)?)", "!="),
        ]
        for pattern, op in pattern_map:
            m = re.search(pattern, q)
            if not m:
                continue
            try:
                return (op, float(m.group(1)))
            except Exception:
                continue
        return None

    @staticmethod
    def _parse_weekdays(user_query: str) -> list[int]:
        q = user_query or ""
        mapping = {
            "周一": 1,
            "星期一": 1,
            "周二": 2,
            "星期二": 2,
            "周三": 3,
            "星期三": 3,
            "周四": 4,
            "星期四": 4,
            "周五": 5,
            "星期五": 5,
            "周六": 6,
            "星期六": 6,
            "周日": 7,
            "周天": 7,
            "星期日": 7,
            "星期天": 7,
        }
        out: list[int] = []
        for k, v in mapping.items():
            if k in q:
                out.append(v)
        return sorted(list(dict.fromkeys(out)))

    @staticmethod
    def _is_weekly_decline_ratio_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_rate = "锁单率" in q
        has_decline = ("下降" in q and "多少" in q) or "下降周数" in q
        has_week_window = "近" in q and "周" in q
        has_source = "下发线索" in q and "门店" in q
        has_weekday = ("周四" in q or "星期四" in q or "周五" in q or "星期五" in q)
        return has_rate and has_decline and has_week_window and has_source and has_weekday

    @staticmethod
    def _is_daily_threshold_count_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_day_count = ("多少天" in q) or ("几天" in q)
        has_recent_day = ("近" in q) and (("日" in q) or ("天" in q))
        has_threshold = PlanningAgent._parse_threshold_condition(q) is not None
        return has_day_count and has_recent_day and has_threshold

    @staticmethod
    def _is_daily_mean_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_mean = any(k in q for k in ["日均", "均值", "平均值", "平均"])
        has_recent_day = bool(re.search(r"近\s*\d+\s*(日|天)", q))
        has_relative_day = any(k in q for k in ["昨天", "昨日", "今天", "今日", "本周", "上周", "本月", "上月"])
        has_explicit_window = PlanningAgent._extract_explicit_time_window(q, datetime.date.today()) is not None
        return has_mean and (has_recent_day or has_relative_day or has_explicit_window)

    @staticmethod
    def _is_daily_percentile_rank_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_percentile = any(k in q for k in ["分位", "百分位", "分位值", "百分位值"])
        has_level = any(k in q for k in ["处于什么水平", "什么水平", "处于什么位置", "高低水平"])
        has_ref = any(k in q for k in ["昨天", "昨日", "今天", "今日"])
        has_recent_day = bool(re.search(r"近\s*\d+\s*(日|天)", q))
        return (has_percentile or has_level) and has_ref and has_recent_day

    @staticmethod
    def _is_weekend_percentile_rank_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_level = any(k in q for k in ["处于什么水平", "什么水平", "处于什么位置", "高低水平", "分位", "百分位"])
        has_weekend = "周末" in q
        has_recent_weekend = bool(re.search(r"近\s*\d+\s*(?:个)?\s*周末", q))
        has_ref = any(k in q for k in ["本周末", "这个周末", "上周末", "上一个周末"])
        return has_level and has_weekend and (has_recent_weekend or has_ref)

    @staticmethod
    def _is_weekday_percentile_rank_query(user_query: str) -> bool:
        q = user_query or ""
        if not q:
            return False
        has_level = any(k in q for k in ["处于什么水平", "什么水平", "处于什么位置", "高低水平", "分位", "百分位"])
        has_weekday = bool(PlanningAgent._parse_weekdays(q))
        has_recent_weeks = bool(re.search(r"近\s*\d+\s*(?:个)?\s*周", q))
        has_ref = any(k in q for k in ["昨天", "昨日", "今天", "今日"])
        return has_level and has_weekday and has_recent_weeks and has_ref

    @staticmethod
    def _extract_explicit_time_window(user_query: str, today: datetime.date) -> tuple[str, str] | None:
        q = user_query or ""
        sanitized = re.sub(r"(昨天|昨日|今天|今日|本周|上周|本月|上月|今年|去年|前年)", " ", q)
        return PlanningAgent._parse_time_window(sanitized, today)

    def _build_yesterday_vs_range_daily_mean_plans(self, user_query: str) -> list[dict] | None:
        q = user_query or ""
        if not q:
            return None
        has_compare = any(k in q for k in ["对比", "相比", "对照", "较"])
        has_yesterday = any(k in q for k in ["昨天", "昨日"])
        has_mean = any(k in q for k in ["日均", "均值", "平均值", "平均"])
        if not (has_compare and has_yesterday and has_mean):
            return None

        today = datetime.date.today()
        explicit_window = self._extract_explicit_time_window(q, today)
        if not explicit_window:
            return None

        metric_defaults = self._metric_defaults(q)
        if not metric_defaults:
            return None

        range_start, range_end = explicit_window
        start_date = datetime.date.fromisoformat(range_start)
        end_date = datetime.date.fromisoformat(range_end)
        window_days = max(1, (end_date - start_date).days)
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]
        yesterday_start = (today - datetime.timedelta(days=1)).isoformat()
        yesterday_end = today.isoformat()

        yesterday_plan = self._normalize_plan(
            {
                "dataset": metric_defaults["dataset"],
                "metric": value_metric,
                "time": {"field": time_field, "start": yesterday_start, "end": yesterday_end},
                "dimensions": [],
                "filters": [{"field": time_field, "op": "!=", "value": None}],
                "comparison": {"type": "none"},
                "question": "昨天的锁单数",
            }
        )
        mean_plan = self._normalize_plan(
            {
                "dataset": metric_defaults["dataset"],
                "metric": value_metric,
                "time": {"field": time_field, "start": range_start, "end": range_end},
                "dimensions": [time_field],
                "filters": [{"field": time_field, "op": "!=", "value": None}],
                "comparison": {"type": "none"},
                "statistics": {
                    "type": "daily_mean",
                    "time_field": time_field,
                    "window_days": window_days,
                    "value_metric": value_metric,
                },
                "question": "指定区间日均锁单数",
            }
        )
        return [yesterday_plan, mean_plan]

    def _build_weekly_decline_ratio_plan(self, user_query: str) -> dict:
        today = datetime.date.today()
        weeks = self._parse_recent_weeks(user_query) or 10
        start = today - datetime.timedelta(days=weeks * 7)
        end = today
        weekdays = self._parse_weekdays(user_query) or [4, 5]
        time_field = "Assign Time 年/月/日"
        numerator = {"field": "下发线索当日锁单数 (门店)", "agg": "sum", "alias": "门店当日锁单数"}
        denominator = {"field": "下发线索数 (门店)", "agg": "sum", "alias": "门店线索数"}
        plan = {
            "dataset": "assign_data",
            "metric": numerator,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "weekly_decline_ratio",
                "time_field": time_field,
                "window_weeks": weeks,
                "weekdays": weekdays,
                "numerator_metric": numerator,
                "denominator_metric": denominator,
            },
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    def _build_daily_threshold_count_plan(self, user_query: str) -> dict | None:
        metric_defaults = self._metric_defaults(user_query)
        threshold_cond = self._parse_threshold_condition(user_query)
        if not metric_defaults or not threshold_cond:
            return None
        op, threshold = threshold_cond
        today = datetime.date.today()
        days = self._parse_recent_days(user_query) or 30
        start = today - datetime.timedelta(days=days)
        end = today
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]

        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": value_metric,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [{"field": time_field, "op": "!=", "value": None}],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "daily_threshold_count",
                "time_field": time_field,
                "window_days": days,
                "op": op,
                "threshold": threshold,
                "value_metric": value_metric,
            },
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    def _build_daily_mean_plan(self, user_query: str) -> dict | None:
        metric_defaults = self._metric_defaults(user_query)
        if not metric_defaults:
            return None
        today = datetime.date.today()
        explicit_window = self._extract_explicit_time_window(user_query, today)
        if explicit_window:
            start_s, end_s = explicit_window
            start = datetime.date.fromisoformat(start_s)
            end = datetime.date.fromisoformat(end_s)
            days = max(1, (end - start).days)
        else:
            days = self._parse_recent_days(user_query) or 30
            start = today - datetime.timedelta(days=days)
            end = today
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]
        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": value_metric,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [{"field": time_field, "op": "!=", "value": None}],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "daily_mean",
                "time_field": time_field,
                "window_days": days,
                "value_metric": value_metric,
            },
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    def _build_daily_percentile_rank_plan(self, user_query: str) -> dict | None:
        metric_defaults = self._metric_defaults(user_query)
        if not metric_defaults:
            return None
        today = datetime.date.today()
        days = self._parse_recent_days(user_query) or 30
        start = today - datetime.timedelta(days=days)
        end = today
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]
        reference_date = (today - datetime.timedelta(days=1)).isoformat() if any(k in user_query for k in ["昨天", "昨日"]) else today.isoformat()
        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": value_metric,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [{"field": time_field, "op": "!=", "value": None}],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "daily_percentile_rank",
                "time_field": time_field,
                "window_days": days,
                "reference_date": reference_date,
                "value_metric": value_metric,
            },
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    def _build_weekend_percentile_rank_plan(self, user_query: str) -> dict | None:
        metric_defaults = self._metric_defaults(user_query)
        if not metric_defaults:
            return None
        today = datetime.date.today()
        weekends = self._parse_recent_weekends(user_query) or 10
        start = today - datetime.timedelta(days=(weekends * 7 + 14))
        end = today + datetime.timedelta(days=1)
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]
        reference_date = today.isoformat()
        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": value_metric,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [{"field": time_field, "op": "!=", "value": None}],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "weekend_percentile_rank",
                "time_field": time_field,
                "window_weekends": weekends,
                "reference_date": reference_date,
                "value_metric": value_metric,
            },
            "question": "周末锁单数在近N个周末中的分位",
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    def _build_weekday_percentile_rank_plan(self, user_query: str) -> dict | None:
        metric_defaults = self._metric_defaults(user_query)
        if not metric_defaults:
            return None
        today = datetime.date.today()
        weeks = self._parse_recent_weeks(user_query) or 10
        weekdays = self._parse_weekdays(user_query) or [7]
        start = today - datetime.timedelta(days=(weeks * 7 + 7))
        end = today + datetime.timedelta(days=1)
        time_field = metric_defaults["time_field"]
        value_metric = metric_defaults["metric"]
        reference_date = (
            (today - datetime.timedelta(days=1)).isoformat()
            if any(k in user_query for k in ["昨天", "昨日"])
            else today.isoformat()
        )
        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": value_metric,
            "time": {"field": time_field, "start": start.isoformat(), "end": end.isoformat()},
            "dimensions": [time_field],
            "filters": [{"field": time_field, "op": "!=", "value": None}],
            "comparison": {"type": "none"},
            "statistics": {
                "type": "weekday_percentile_rank",
                "time_field": time_field,
                "window_weeks": weeks,
                "weekdays": weekdays,
                "reference_date": reference_date,
                "value_metric": value_metric,
            },
            "question": "指定周内日锁单数在近N周中的分位",
        }
        return self._fill_defaults(self._normalize_plan(plan), user_query)

    @staticmethod
    def _statistics_plan_valid(statistics: dict, plan: dict) -> bool:
        stype = statistics.get("type")
        if stype == "weekly_decline_ratio":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            weekdays = statistics.get("weekdays")
            numerator = statistics.get("numerator_metric")
            denominator = statistics.get("denominator_metric")
            if not isinstance(time_field, str) or not time_field:
                return False
            if not isinstance(weekdays, list) or not weekdays:
                return False
            if not isinstance(numerator, dict) or not isinstance(denominator, dict):
                return False
            if not numerator.get("field") or not denominator.get("field"):
                return False
            if not numerator.get("agg") or not denominator.get("agg"):
                return False
            if (
                numerator.get("field") == denominator.get("field")
                and numerator.get("agg") == denominator.get("agg")
                and (numerator.get("alias") or "") == (denominator.get("alias") or "")
            ):
                return False
            return True

        if stype == "daily_threshold_count":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            op = statistics.get("op")
            threshold = statistics.get("threshold")
            value_metric = statistics.get("value_metric")
            if not isinstance(time_field, str) or not time_field:
                return False
            if op not in {">", ">=", "<", "<=", "==", "!="}:
                return False
            if not isinstance(value_metric, dict):
                return False
            if not value_metric.get("field") or not value_metric.get("agg"):
                return False
            try:
                float(threshold)
            except Exception:
                return False
            return True

        if stype == "daily_mean":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            value_metric = statistics.get("value_metric")
            if not isinstance(time_field, str) or not time_field:
                return False
            if not isinstance(value_metric, dict):
                return False
            if not value_metric.get("field") or not value_metric.get("agg"):
                return False
            return True

        if stype == "weekday_percentile_rank":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            value_metric = statistics.get("value_metric")
            weekdays = statistics.get("weekdays")
            window_weeks = statistics.get("window_weeks")
            reference_date = statistics.get("reference_date")
            if not isinstance(time_field, str) or not time_field:
                return False
            if not isinstance(value_metric, dict):
                return False
            if not value_metric.get("field") or not value_metric.get("agg"):
                return False
            if not isinstance(weekdays, list) or not weekdays:
                return False
            try:
                window_weeks = int(window_weeks)
            except Exception:
                return False
            if reference_date is not None and not isinstance(reference_date, str):
                return False
            return window_weeks > 0

        if stype == "daily_percentile_rank":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            value_metric = statistics.get("value_metric")
            reference_date = statistics.get("reference_date")
            if not isinstance(time_field, str) or not time_field:
                return False
            if not isinstance(value_metric, dict):
                return False
            if not value_metric.get("field") or not value_metric.get("agg"):
                return False
            if reference_date is not None and not isinstance(reference_date, str):
                return False
            return True

        if stype == "weekend_percentile_rank":
            time_field = statistics.get("time_field") or (plan.get("time", {}) or {}).get("field")
            value_metric = statistics.get("value_metric")
            reference_date = statistics.get("reference_date")
            if not isinstance(time_field, str) or not time_field:
                return False
            if not isinstance(value_metric, dict):
                return False
            if not value_metric.get("field") or not value_metric.get("agg"):
                return False
            if reference_date is not None and not isinstance(reference_date, str):
                return False
            return True

        return False

    @staticmethod
    def _normalize_plan(plan: dict) -> dict:
        dataset = plan.get("dataset")
        if isinstance(dataset, str):
            lowered = dataset.lower()
            if lowered.endswith(".parquet") or lowered.endswith(".csv"):
                plan["dataset"] = dataset.rsplit(".", 1)[0]

        filters = plan.get("filters")
        if not isinstance(filters, list):
            filters = []

        normalized_filters: list[dict] = []
        seen_filters: set[tuple[str, str, str]] = set()
        for f in filters:
            if not isinstance(f, dict):
                continue
            field = f.get("field")
            op = f.get("op")
            value = f.get("value", None)

            if field in {"store_city", "license_city"} and isinstance(value, str):
                raw = value.strip()
                if "市" in raw:
                    raw = raw.split("市", 1)[0] + "市"
                else:
                    m_city = re.search(r"([\u4e00-\u9fff]{2,8})", raw)
                    if m_city:
                        raw = m_city.group(1)

                base = raw[:-1] if raw.endswith("市") and len(raw) > 1 else raw
                if base:
                    value = base

            if op in {"not null", "not_null", "is not null", "is_not_null"}:
                normalized = {"field": field, "op": "!=", "value": None}
                key = (str(field), "!=", "None")
                if key not in seen_filters:
                    normalized_filters.append(normalized)
                    seen_filters.add(key)
                continue
            if op in {"null", "is null", "is_null"}:
                normalized = {"field": field, "op": "==", "value": None}
                key = (str(field), "==", "None")
                if key not in seen_filters:
                    normalized_filters.append(normalized)
                    seen_filters.add(key)
                continue
            if op in {"=", "eq"}:
                normalized = {"field": field, "op": "==", "value": value}
                key = (str(field), "==", repr(value))
                if key not in seen_filters:
                    normalized_filters.append(normalized)
                    seen_filters.add(key)
                continue

            if field in {"store_city", "license_city"} and op == "==" and isinstance(value, str) and value:
                variants = []
                base = value[:-1] if value.endswith("市") and len(value) > 1 else value
                if base:
                    variants.append(base)
                    variants.append(f"{base}市")
                variants = list(dict.fromkeys(variants))
                normalized = {"field": field, "op": "in", "value": variants}
                key = (str(field), "in", repr(variants))
                if key not in seen_filters:
                    normalized_filters.append(normalized)
                    seen_filters.add(key)
                continue

            if op == "!=" and "value" not in f:
                normalized = {"field": field, "op": "!=", "value": None}
                key = (str(field), "!=", "None")
                if key not in seen_filters:
                    normalized_filters.append(normalized)
                    seen_filters.add(key)
                continue

            normalized = {"field": field, "op": op, "value": value}
            key = (str(field), str(op), repr(value))
            if key not in seen_filters:
                normalized_filters.append(normalized)
                seen_filters.add(key)

        plan["filters"] = normalized_filters

        dims = plan.get("dimensions")
        if dims is None:
            plan["dimensions"] = []
        elif not isinstance(dims, list):
            plan["dimensions"] = [str(dims)]

        comparison = plan.get("comparison")
        if not isinstance(comparison, dict):
            plan["comparison"] = {"type": "none"}
        else:
            ctype = comparison.get("type")
            if ctype not in {"none", "yoy", "wow", "dod"}:
                plan["comparison"] = {"type": "none"}

        time = plan.get("time")
        if not isinstance(time, dict):
            plan["time"] = {}

        metric = plan.get("metric")
        if not isinstance(metric, dict):
            plan["metric"] = {}

        statistics = plan.get("statistics")
        if statistics is not None and not isinstance(statistics, dict):
            plan["statistics"] = {}
        elif isinstance(statistics, dict):
            stype = statistics.get("type")
            if stype not in {
                "weekly_decline_ratio",
                "daily_threshold_count",
                "daily_mean",
                "daily_percentile_rank",
                "weekend_percentile_rank",
                "weekday_percentile_rank",
            }:
                plan["statistics"] = {}
            elif stype == "weekly_decline_ratio":
                wdays = statistics.get("weekdays")
                if isinstance(wdays, list):
                    normalized_wdays = []
                    for w in wdays:
                        if isinstance(w, (int, float, str)) and str(w).isdigit():
                            iv = int(w)
                            if 1 <= iv <= 7:
                                normalized_wdays.append(iv)
                    statistics["weekdays"] = sorted(list(dict.fromkeys(normalized_wdays)))
                wweeks = statistics.get("window_weeks")
                if isinstance(wweeks, str) and wweeks.isdigit():
                    statistics["window_weeks"] = int(wweeks)
            elif stype == "daily_threshold_count":
                op = statistics.get("op")
                if op not in {">", ">=", "<", "<=", "==", "!="}:
                    statistics["op"] = ">"
                threshold = statistics.get("threshold")
                try:
                    statistics["threshold"] = float(threshold)
                except Exception:
                    statistics["threshold"] = 0.0
                wdays = statistics.get("window_days")
                if isinstance(wdays, str) and wdays.isdigit():
                    statistics["window_days"] = int(wdays)
            elif stype == "daily_mean":
                wdays = statistics.get("window_days")
                if isinstance(wdays, str) and wdays.isdigit():
                    statistics["window_days"] = int(wdays)
            elif stype == "daily_percentile_rank":
                wdays = statistics.get("window_days")
                if isinstance(wdays, str) and wdays.isdigit():
                    statistics["window_days"] = int(wdays)
            elif stype == "weekend_percentile_rank":
                wends = statistics.get("window_weekends")
                if isinstance(wends, str) and wends.isdigit():
                    statistics["window_weekends"] = int(wends)
            elif stype == "weekday_percentile_rank":
                wdays = statistics.get("weekdays")
                if isinstance(wdays, list):
                    normalized = []
                    for w in wdays:
                        if isinstance(w, (int, float, str)) and str(w).isdigit():
                            iv = int(w)
                            if 1 <= iv <= 7:
                                normalized.append(iv)
                    statistics["weekdays"] = sorted(list(dict.fromkeys(normalized)))
                wweeks = statistics.get("window_weeks")
                if isinstance(wweeks, str) and wweeks.isdigit():
                    statistics["window_weeks"] = int(wweeks)
            if not PlanningAgent._statistics_plan_valid(statistics, plan):
                plan["statistics"] = {}

        fast_path = plan.get("fast_path")
        if fast_path is not None and not isinstance(fast_path, dict):
            plan["fast_path"] = {}
        elif isinstance(fast_path, dict):
            fp_type = fast_path.get("type")
            if fp_type not in {"numeric_ratio", "current_iso_week", "small_talk_contextual"}:
                plan["fast_path"] = {}
            elif fp_type == "numeric_ratio":
                try:
                    fast_path["current"] = float(fast_path.get("current"))
                    fast_path["base"] = float(fast_path.get("base"))
                except Exception:
                    plan["fast_path"] = {}

        return plan

    def _rule_based_plan(self, user_query: str) -> dict | None:
        today = datetime.date.today()
        comparison_type = self._parse_comparison_type(user_query)
        metric_defaults = self._metric_defaults(user_query)
        if not metric_defaults:
            return None

        time_window = self._parse_time_window(user_query, today) or (
            (today - datetime.timedelta(days=1)).isoformat(),
            today.isoformat(),
        )

        start, end = time_window
        time_field = metric_defaults["time_field"]
        non_null_field = metric_defaults.get("non_null_field")

        plan = {
            "dataset": metric_defaults["dataset"],
            "metric": metric_defaults["metric"],
            "time": {"field": time_field, "start": start, "end": end},
            "dimensions": [],
            "filters": [],
            "comparison": {"type": comparison_type},
        }

        if non_null_field:
            plan["filters"].append({"field": non_null_field, "op": "!=", "value": None})

        return self._normalize_plan(plan)

    @staticmethod
    def _split_user_query(user_query: str) -> list[str]:
        q = (user_query or "").strip()
        if not q:
            return []
        parts = re.split(r"[？?\n；;]+", q)
        return [p.strip() for p in parts if p and p.strip()]

    def _finalize_plans(self, plans: list[dict], user_query: str) -> list[dict]:
        finalized: list[dict] = []
        for plan in plans:
            if not isinstance(plan, dict) or not plan:
                continue
            q = plan.get("question") or user_query
            normalized = self._fill_defaults(self._normalize_plan(plan), q)
            has_compare = any(k in q for k in ["对比", "相比", "对照", "较"])
            if self._is_weekday_percentile_rank_query(q):
                stat = normalized.get("statistics")
                if not isinstance(stat, dict) or stat.get("type") != "weekday_percentile_rank":
                    weekday_plan = self._build_weekday_percentile_rank_plan(q)
                    if isinstance(weekday_plan, dict) and weekday_plan:
                        weekday_plan["question"] = q
                        finalized.append(weekday_plan)
                        continue
            if self._is_weekend_percentile_rank_query(q):
                stat = normalized.get("statistics")
                if not isinstance(stat, dict) or stat.get("type") != "weekend_percentile_rank":
                    weekend_plan = self._build_weekend_percentile_rank_plan(q)
                    if isinstance(weekend_plan, dict) and weekend_plan:
                        weekend_plan["question"] = q
                        finalized.append(weekend_plan)
                        continue
            if self._is_daily_mean_query(q):
                stat = normalized.get("statistics")
                if has_compare and any(k in q for k in ["昨天", "昨日"]):
                    pair = self._build_yesterday_vs_range_daily_mean_plans(q)
                    if isinstance(pair, list) and pair:
                        finalized.extend(pair)
                        continue
                if not has_compare and (not isinstance(stat, dict) or stat.get("type") != "daily_mean"):
                    daily_plan = self._build_daily_mean_plan(q)
                    if isinstance(daily_plan, dict) and daily_plan:
                        daily_plan["question"] = q
                        finalized.append(daily_plan)
                        continue
            if self._is_daily_percentile_rank_query(q):
                stat = normalized.get("statistics")
                if not isinstance(stat, dict) or stat.get("type") != "daily_percentile_rank":
                    percentile_plan = self._build_daily_percentile_rank_plan(q)
                    if isinstance(percentile_plan, dict) and percentile_plan:
                        percentile_plan["question"] = q
                        finalized.append(percentile_plan)
                        continue
            normalized["question"] = q
            finalized.append(normalized)
        return [p for p in finalized if isinstance(p, dict) and p]

    def create_plans(self, user_query: str, memory_context: dict | None = None) -> list[dict]:
        parts = self._split_user_query(user_query) or [user_query]
        fp = self._parse_fast_path_query(user_query)
        if isinstance(fp, dict) and fp.get("type"):
            return [
                self._normalize_plan(
                    {
                        "question": user_query,
                        "dataset": "order_data",
                        "metric": {"field": "order_number", "agg": "count", "alias": "count", "business_name": "订单计数"},
                        "time": {
                            "field": "order_create_time",
                            "start": datetime.date.today().isoformat(),
                            "end": (datetime.date.today() + datetime.timedelta(days=1)).isoformat(),
                        },
                        "dimensions": [],
                        "filters": [],
                        "comparison": {"type": "none"},
                        "fast_path": fp,
                    }
                )
            ]
        for part in parts:
            if self._should_sales_clarify(part):
                return [{"question": part, "clarification": self._sales_clarification(part)}]
            need, city = self._should_city_clarify(part)
            if need and city:
                return [{"question": part, "clarification": self._city_clarification(city, part)}]

        current_date = datetime.date.today().isoformat()
        memory_context = memory_context if isinstance(memory_context, dict) else {}
        memory_facts = memory_context.get("facts") if isinstance(memory_context.get("facts"), dict) else {}
        memory_working = memory_context.get("working_memory") if isinstance(memory_context.get("working_memory"), dict) else {}
        memory_text = (
            "执行记忆:\n"
            f"- 已有 facts: {json.dumps(memory_facts, ensure_ascii=False)}\n"
            f"- 当前 working_memory: {json.dumps(memory_working, ensure_ascii=False)}\n\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个智能数据分析助手 (Planning Agent)。"
                    "你的任务是把用户问题转成可执行前的规划 DSL（包含时间范围、对比类型、拆解维度、过滤口径）。"
                    "不要直接回答结论，必须调用 create_planning_dsl 工具返回 plans。\n\n"
                    f"今天是: {current_date}\n\n"
                    "数据集与 Schema:\n"
                    f"{self.schema_md}\n\n"
                    "业务定义:\n"
                    f"{self.business_definition}\n\n"
                    f"{memory_text}"
                    "约束:\n"
                    "- 默认返回 1 个 plan；仅当用户明确包含多个子问题时才拆成多个 plan，并保持原顺序。\n"
                    "- 每个 plan 必须填写 question 字段用于回显。\n"
                    "- time.start/time.end 必须是 YYYY-MM-DD，且 end 为开区间。\n"
                    "- 遇到歧义必须返回 clarification.need=true，禁止自行猜测。\n"
                    "- 澄清规则与指标口径以 Schema 文档为准。\n"
                    "- 锁单量口径：order_number count 且 lock_time 非空，时间筛选基于 lock_time。\n"
                    "- 路由优先级：Fast Path > Operators > Comparison/Statistics/Query。\n"
                    "- 纯数字比较问题（如“405环比382提升多少”）输出 fast_path={type:numeric_ratio,current,base}。\n"
                    "- 日期周序问题（如“今天是第几周/ISO周数”）输出 fast_path={type:current_iso_week}。\n"
                    "- 闲聊/致谢/鼓励问题（如 welldone、谢谢、辛苦了）输出 fast_path={type:small_talk_contextual}。\n"
                    "- 在营门店数问题优先走固定算子，plan.statistics 置空或不设置；时间字段优先 order_create_date。\n"
                    "- 用户出现‘试驾车’时 filters 必须含 order_type == 试驾车；出现‘用户车’时必须含 order_type == 用户车。\n"
                    "- 用户出现系列词（L6/L7/LS6/LS7/LS8/LS9）时，filters 应补充 series 约束。\n"
                    "- 若用户出现 CM0/CM1/CM2/DM0/DM1 这类二级车型分组，需使用 business_definition.series_group_logic（product_name 逻辑）生成 filters，禁止把这些 token 直接写到 series 字段。\n"
                    "- 注意：LS6/L6 是 series 车系，不要按 model_series_mapping 展开成 CM0/CM1/CM2 或 DM0/DM1；只有当用户明确问 CM0/CM1/CM2/DM0/DM1 时才使用 series_group_logic。\n"
                    "- 同比/年同比用 comparison.type=yoy；周环比用 comparison.type=wow；日环比（如“昨天日环比”）用 comparison.type=dod。\n"
                    "- 时序统计类按类型输出 statistics：weekly_decline_ratio / daily_threshold_count / daily_mean / daily_percentile_rank / weekend_percentile_rank / weekday_percentile_rank，并补齐各自必需字段。\n"
                    "- 若用户问“近 N 个周的周日/周一..周日 处于什么水平”，使用 weekday_percentile_rank，并设置 window_weeks=N、weekdays=[对应周内日]、reference_date=昨天/今天。\n"
                ),
            },
            {"role": "user", "content": user_query},
        ]

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_PLANNER_MODEL,
                messages=messages,
                tools=[PLANNING_TOOL_SCHEMA],
                tool_choice={"type": "function", "function": {"name": "create_planning_dsl"}},
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            for tool_call in tool_calls:
                if tool_call.function.name != "create_planning_dsl":
                    continue
                args = json.loads(tool_call.function.arguments or "{}")
                raw_plans = args.get("plans")
                clarification = args.get("clarification")
                if isinstance(clarification, dict) and clarification.get("need"):
                    return [{"question": user_query, "clarification": clarification}]
                if isinstance(raw_plans, list):
                    plans: list[dict] = []
                    for p in raw_plans:
                        if not isinstance(p, dict):
                            continue
                        plans.append(p)
                    return self._finalize_plans(plans, user_query)
                raw_plan = args.get("plan")
                if isinstance(raw_plan, dict):
                    return self._finalize_plans([raw_plan], user_query)

            content = message.content or ""
            obj = json.loads(content)
            if isinstance(obj, dict) and isinstance(obj.get("plans"), list):
                plans: list[dict] = []
                for p in obj["plans"]:
                    if not isinstance(p, dict):
                        continue
                    plans.append(p)
                return self._finalize_plans(plans, user_query)
            if isinstance(obj, dict) and isinstance(obj.get("clarification"), dict) and obj["clarification"].get("need"):
                return [{"question": user_query, "clarification": obj["clarification"]}]
            if isinstance(obj, dict) and isinstance(obj.get("plan"), dict):
                return self._finalize_plans([obj["plan"]], user_query)
        except Exception:
            pass

        for part in parts:
            intent = self._classify_intent(part)
            if intent == "statistics":
                ps = self._build_yesterday_vs_range_daily_mean_plans(part)
                if isinstance(ps, list) and ps:
                    return ps
            if intent == "statistics" and self._is_weekly_decline_ratio_query(part):
                p = self._build_weekly_decline_ratio_plan(part)
                p["question"] = part
                return [p]
            if intent == "statistics" and self._is_daily_threshold_count_query(part):
                p = self._build_daily_threshold_count_plan(part)
                if isinstance(p, dict) and p:
                    p["question"] = part
                    return [p]
            if intent == "statistics" and self._is_daily_mean_query(part) and not any(k in part for k in ["对比", "相比", "对照", "较"]):
                p = self._build_daily_mean_plan(part)
                if isinstance(p, dict) and p:
                    p["question"] = part
                    return [p]
            if intent == "statistics" and self._is_daily_percentile_rank_query(part):
                p = self._build_daily_percentile_rank_plan(part)
                if isinstance(p, dict) and p:
                    p["question"] = part
                    return [p]
            if intent == "statistics" and self._is_weekend_percentile_rank_query(part):
                p = self._build_weekend_percentile_rank_plan(part)
                if isinstance(p, dict) and p:
                    p["question"] = part
                    return [p]
            if intent == "statistics" and self._is_weekday_percentile_rank_query(part):
                p = self._build_weekday_percentile_rank_plan(part)
                if isinstance(p, dict) and p:
                    p["question"] = part
                    return [p]

        base_defaults = self._metric_defaults(user_query)
        rule_plans: list[dict] = []
        for part in parts:
            effective_part = part
            if base_defaults and not self._metric_defaults(part):
                metric_hint = base_defaults.get("metric", {}).get("business_name") or base_defaults.get("metric", {}).get("alias")
                if metric_hint:
                    effective_part = f"{metric_hint} {part}"
            plan = self._rule_based_plan(effective_part)
            if isinstance(plan, dict) and plan:
                plan["question"] = part
                rule_plans.append(plan)
        return self._finalize_plans(rule_plans, user_query)

    def _fill_defaults(self, plan: dict, user_query: str) -> dict:
        metric_defaults = self._metric_defaults(user_query)
        today = datetime.date.today()
        default_start = (today - datetime.timedelta(days=1)).isoformat()
        default_end = today.isoformat()

        if not plan.get("dataset"):
            plan["dataset"] = metric_defaults["dataset"] if metric_defaults else "order_data"

        metric = plan.get("metric")
        if not isinstance(metric, dict):
            metric = {}
        if not metric.get("field") or not metric.get("agg"):
            if metric_defaults:
                plan["metric"] = metric_defaults["metric"]

        time = plan.get("time")
        if not isinstance(time, dict):
            time = {}
        if not time.get("start") or not time.get("end"):
            time["start"] = time.get("start") or default_start
            time["end"] = time.get("end") or default_end
        if not time.get("field") and metric_defaults:
            time["field"] = metric_defaults["time_field"]
        plan["time"] = time

        filters = plan.get("filters")
        if not isinstance(filters, list):
            filters = []

        time_field = time.get("field")
        if isinstance(time_field, str) and time_field in {"lock_time", "delivery_date", "invoice_upload_time", "intention_payment_time"}:
            has_non_null = any(
                isinstance(f, dict) and f.get("field") == time_field and f.get("op") == "!=" and f.get("value") is None
                for f in filters
            )
            if not has_non_null:
                filters.append({"field": time_field, "op": "!=", "value": None})
        if metric_defaults and (metric_defaults.get("metric") or {}).get("alias") == "在营门店数":
            if isinstance(time, dict):
                time["field"] = "order_create_date"
                if not any(
                    isinstance(f, dict) and f.get("field") == "order_create_date" and f.get("op") == "!=" and f.get("value") is None
                    for f in filters
                ):
                    filters.append({"field": "order_create_date", "op": "!=", "value": None})
            q = user_query or ""
            if any(k in q for k in ["最大", "最高", "峰值", "最小", "最低"]):
                dims = plan.get("dimensions")
                if not isinstance(dims, list) or not dims:
                    plan["dimensions"] = ["order_create_date"]
                if isinstance(plan.get("statistics"), dict):
                    plan["statistics"] = {}

        dims = plan.get("dimensions")
        if not isinstance(dims, list):
            dims = []
        if not dims:
            q = (user_query or "").replace(" ", "")
            dataset = str(plan.get("dataset") or "")
            want_series = any(k in q for k in ["分车型", "按车型", "车型分别", "分车系", "按车系", "车系分别", "按系列", "分系列"])
            want_product = any(k in q for k in ["按产品名称", "分产品名称", "产品名称", "按产品", "分产品"])
            want_region = any(k in q for k in ["按大区", "分大区", "大区分别"])
            want_store = any(k in q for k in ["按门店", "分门店", "门店分别"])
            want_store_city = any(k in q for k in ["按门店城市", "分门店城市", "门店城市分别"])
            want_license_city = any(k in q for k in ["按上牌城市", "分上牌城市", "上牌城市分别"])
            if dataset == "order_data":
                if want_product:
                    plan["dimensions"] = ["product_name"]
                elif want_series:
                    plan["dimensions"] = ["series"]
                elif want_store_city:
                    plan["dimensions"] = ["store_city"]
                elif want_license_city:
                    plan["dimensions"] = ["license_city"]
                elif want_store:
                    plan["dimensions"] = ["store_name"]
                elif want_region:
                    plan["dimensions"] = ["parent_region_name"]
        filters = self._apply_semantic_filters(filters, user_query)
        plan["filters"] = filters


        comparison = plan.get("comparison")
        if not isinstance(comparison, dict) or comparison.get("type") not in {"none", "yoy", "wow", "dod"}:
            plan["comparison"] = {"type": self._parse_comparison_type(user_query)}

        fast_path = self._parse_fast_path_query(user_query)
        if fast_path:
            plan["fast_path"] = fast_path
        elif "fast_path" not in plan:
            plan["fast_path"] = {}

        time = plan.get("time")
        if isinstance(time, dict):
            if not time.get("start") or not time.get("end"):
                time["start"] = time.get("start") or default_start
                time["end"] = time.get("end") or default_end

        return self._normalize_plan(plan)

    def create_plan(self, user_query: str, memory_context: dict | None = None) -> dict:
        plans = self.create_plans(user_query, memory_context=memory_context)
        if plans:
            first = plans[0]
            if isinstance(first, dict) and first:
                return self._validate_and_rewrite_time(first, user_query)
            return first
        return {}

    @staticmethod
    def _contains_relative_to_today_hint(user_query: str) -> bool:
        q = user_query or ""
        return any(k in q for k in ["至今", "截至", "目前", "现在", "今日", "今天", "本月"])

    @staticmethod
    def _infer_time_window_type(user_query: str) -> str | None:
        q = (user_query or "").strip()
        if not q:
            return None
        if "昨天" in q or "昨日" in q:
            return "yesterday"
        if "本月" in q:
            return "this_month_to_today"
        if "上月" in q:
            return "last_month"
        if any(k in q for k in ["至今", "截至", "目前", "现在", "今日", "今天"]) and ("月" in q):
            return "month_to_today"
        if re.search(
            r"\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]?\s*(?:到|至|[-~—–－])\s*(?:\d{2,4}\s*年\s*)?(?:\d{1,2}\s*月\s*)?\d{1,2}\s*[日号]?",
            q,
        ):
            return "date_range"
        if re.search(r"\d{4}-\d{2}-\d{2}\s*(?:到|至|[-~—–－])\s*\d{4}-\d{2}-\d{2}", q):
            return "date_range"
        if re.search(r"\d{2,4}\s*年\s*\d{1,2}\s*月(?!\s*\d)", q) or re.search(r"(?:(?:\d{2,4}\s*年\s*)?)(?:正|十一|十二|[一二两三四五六七八九十])\s*月(?!\s*\d)", q):
            return "month"
        if re.search(r"\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]?", q):
            return "date"
        if re.search(r"\d{2,4}\s*年(?!\s*\d|\s*月)", q):
            return "year"
        if re.search(r"\d{4}-\d{2}-\d{2}", q):
            return "date"
        return None

    @staticmethod
    def _safe_parse_iso_date(value: object) -> datetime.date | None:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.date.fromisoformat(raw)
        except Exception:
            return None

    def _rewrite_time_with_llm(
        self,
        user_query: str,
        current_time: dict,
        rule_window: tuple[str, str],
        today: datetime.date,
        time_type: str,
    ) -> tuple[str, str] | None:
        if self.client is None:
            return rule_window
        messages = [
            {
                "role": "system",
                "content": (
                    "你是时间窗口修复器。"
                    "给定用户问题与当前 time（start/end），当发现不一致时你需要重写 time。"
                    "时间窗口必须是左闭右开：[start,end)。"
                    "start/end 必须是 YYYY-MM-DD。"
                    "如果时间类型为 this_month_to_today / month_to_today / yesterday，必须严格符合语义。"
                    "必须与规则解析窗口一致。"
                    "只输出工具调用 rewrite_time_window。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"今天日期: {today.isoformat()}\n"
                    f"用户问题: {user_query}\n"
                    f"时间类型: {time_type}\n"
                    f"当前 time: {json.dumps(current_time, ensure_ascii=False)}\n"
                    f"规则解析窗口(供参考): start={rule_window[0]} end={rule_window[1]}\n"
                    "请重写 time.start/time.end。"
                ),
            },
        ]
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_PLANNER_MODEL,
                messages=messages,
                tools=[TIME_REWRITE_TOOL_SCHEMA],
                tool_choice={"type": "function", "function": {"name": "rewrite_time_window"}},
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            for tool_call in tool_calls:
                if tool_call.function.name != "rewrite_time_window":
                    continue
                args = json.loads(tool_call.function.arguments or "{}")
                start = args.get("start")
                end = args.get("end")
                s = self._safe_parse_iso_date(start)
                e = self._safe_parse_iso_date(end)
                if not s or not e or e <= s:
                    return rule_window
                out = (s.isoformat(), e.isoformat())
                if time_type in {"this_month_to_today", "month_to_today"} and out[1] != today.isoformat():
                    return rule_window
                if time_type == "yesterday" and out[1] != today.isoformat():
                    return rule_window
                if out != rule_window:
                    return rule_window
                return out
        except Exception:
            return rule_window
        return rule_window

    def _validate_and_rewrite_time(self, plan: dict, user_query: str, today: datetime.date | None = None) -> dict:
        if not isinstance(plan, dict) or not plan:
            return plan
        time = plan.get("time")
        if not isinstance(time, dict):
            return plan
        today = today or datetime.date.today()
        rule_window = self._parse_time_window(user_query, today)
        if not rule_window:
            return plan
        time_type = self._infer_time_window_type(user_query)
        if not time_type or time_type not in self._TIME_REWRITE_WHITELIST:
            return plan
        if time.get("start") == rule_window[0] and time.get("end") == rule_window[1]:
            return plan
        rewritten = self._rewrite_time_with_llm(
            user_query=user_query,
            current_time=time,
            rule_window=rule_window,
            today=today,
            time_type=time_type,
        )
        if rewritten:
            time["start"], time["end"] = rewritten
            plan["time"] = time
        return plan


def _extract_json_content(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw


def plan_runtime_action(client: OpenAI, state: AgentRuntimeState) -> dict:
    if state.iteration == 0 and not state.history:
        return {
            "action": "run_dsl",
            "reason": "首次执行，先获取核心数据事实。",
            "query": state.goal,
            "analysis": "开始围绕用户目标进行首轮查询。",
        }

    history_payload = json.dumps(state.history[-3:], ensure_ascii=False)
    facts_payload = json.dumps(state.facts, ensure_ascii=False)
    working_payload = json.dumps(state.working_memory, ensure_ascii=False)
    messages = [
        {"role": "system", "content": LOOP_RUNTIME_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户目标:\n{state.goal}\n\n"
                f"已执行步数: {state.iteration}/{state.max_steps}\n"
                f"已有 facts:\n{facts_payload}\n\n"
                f"当前 working_memory:\n{working_payload}\n\n"
                f"历史:\n{history_payload}\n\n"
                "请输出下一步 JSON。"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(model=DEEPSEEK_PLANNER_MODEL, messages=messages)
        content = response.choices[0].message.content or ""
        parsed = json.loads(_extract_json_content(content))
        if not isinstance(parsed, dict):
            raise ValueError("runtime action is not an object")
    except Exception as e:
        return {
            "action": "finish",
            "reason": "loop action 解析失败，触发保底收敛。",
            "analysis": f"解析异常: {str(e)}",
        }

    action = str(parsed.get("action") or "").strip().lower()
    if action not in {"run_dsl", "finish"}:
        action = "finish"
    out = {
        "action": action,
        "reason": str(parsed.get("reason") or ""),
        "analysis": str(parsed.get("analysis") or ""),
    }
    if action == "run_dsl":
        query = str(parsed.get("query") or "").strip() or state.goal
        out["query"] = query
    return out
