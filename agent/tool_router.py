import json
import datetime
import re

from agent.planner import PlanningAgent
from operators import run_registered_operator
from tools import ComparisonTool, FastPathTool, QueryTool, StatisticsTool


def _execute_single_plan(
    plan: dict,
    user_query: str,
    query_tool: QueryTool,
    comparison_tool: ComparisonTool,
    statistics_tool: StatisticsTool,
    memory_context: dict | None = None,
) -> dict:
    dataset = plan.get("dataset")
    metric = plan.get("metric", {}) or {}
    time = plan.get("time", {}) or {}
    dimensions = plan.get("dimensions", []) or []
    filters = plan.get("filters", []) or []
    comparison = plan.get("comparison", {}) or {}
    statistics = plan.get("statistics", {}) or {}
    fast_path = plan.get("fast_path", {}) or {}

    time_field = time.get("field")
    time_start = time.get("start")
    time_end = time.get("end")

    filters_without_time = []
    for f in filters:
        if not isinstance(f, dict):
            continue
        if f.get("field") != time_field:
            filters_without_time.append(f)
            continue
        if f.get("op") in {">=", "<"} and str(f.get("value")) in {str(time_start), str(time_end)}:
            continue
        filters_without_time.append(f)

    comparison_type = comparison.get("type")
    stats_type = statistics.get("type") if isinstance(statistics, dict) else None

    execution_meta = {"engine": "dsl", "route": "query_tool"}
    tool_result = None
    comparison_df = None
    if isinstance(fast_path, dict) and fast_path.get("type"):
        execution_meta = {"engine": "fast_path", "route": f"fast_path.{str(fast_path.get('type'))}"}
        tool_result = FastPathTool().run(
            config=fast_path,
            user_query=user_query,
            memory_context=memory_context,
        )
    operator_result = run_registered_operator(plan=plan, user_query=user_query, query_tool=query_tool)
    if operator_result is not None and tool_result is None:
        execution_meta = {
            "engine": "operator",
            "route": f"operators.{str(operator_result.get('type') or 'unknown')}",
        }
        print(f"[Route] 使用固定算子: {execution_meta['route']}")
        tool_result = operator_result
    if comparison_type in {"yoy", "wow", "dod"}:
        if comparison_type == "wow" and stats_type == "weekly_decline_ratio":
            execution_meta = {"engine": "comparison", "route": "comparison.weekly_wow_series"}
            print("\n[Thinking] 执行共享周序列算子（Comparison → Weekly WoW Series）...")
            comparison_result = comparison_tool.build_weekly_wow_series(
                {
                    "dataset": dataset,
                    "filters": filters_without_time,
                    "time": {"field": time_field, "start": time_start, "end": time_end},
                    "statistics": statistics,
                }
            )
            if isinstance(comparison_result, str):
                tool_result = comparison_result
            else:
                comparison_df = comparison_result
        else:
            execution_meta = {"engine": "comparison", "route": f"comparison.{comparison_type}"}
            print(f"\n[Thinking] 执行派生指标对比计算: {comparison_type}")
            comparison_result = comparison_tool.perform_comparison_df(
                {
                    "dataset": dataset,
                    "metrics": [
                        {
                            "field": metric.get("field"),
                            "agg": metric.get("agg"),
                            "alias": metric.get("alias") or metric.get("business_name") or "value",
                        }
                    ],
                    "dimensions": dimensions,
                    "filters": filters_without_time,
                    "time": {"field": time_field, "start": time_start, "end": time_end},
                    "comparison": {"type": comparison_type},
                }
            )
            if isinstance(comparison_result, str):
                tool_result = comparison_result
            else:
                comparison_df = comparison_result
                if not stats_type:
                    tool_result = comparison_df.to_string(index=False)

    if stats_type == "weekly_decline_ratio" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.weekly_decline_ratio"}
        print("\n[Thinking] 执行统计型序列分析...")
        if comparison_df is not None:
            if comparison_type != "wow":
                tool_result = {
                    "type": "weekly_decline_ratio",
                    "error": "unsupported_pipeline_input",
                    "message": "weekly_decline_ratio 仅支持与 wow 周序列算子联动。",
                }
            else:
                stat_request = {
                    "type": "weekly_decline_ratio",
                    "series_input": True,
                    "window_weeks": statistics.get("window_weeks") or 10,
                    "weekdays": statistics.get("weekdays") or [4, 5],
                }
                try:
                    tool_result = statistics_tool.perform_statistics(stat_request, comparison_df)
                except Exception as e:
                    tool_result = {"type": "weekly_decline_ratio", "error": "statistics_execution_failed", "message": str(e)}
        else:
            numerator_metric = statistics.get("numerator_metric", {}) if isinstance(statistics, dict) else {}
            denominator_metric = statistics.get("denominator_metric", {}) if isinstance(statistics, dict) else {}
            window_weeks = statistics.get("window_weeks") or 10
            try:
                window_weeks = int(window_weeks)
            except Exception:
                window_weeks = 10
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=(max(int(window_weeks), 1) * 7 + 7))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": numerator_metric.get("field") or metric.get("field"),
                        "agg": numerator_metric.get("agg") or metric.get("agg") or "sum",
                        "alias": numerator_metric.get("alias") or "门店当日锁单数",
                    },
                    {
                        "field": denominator_metric.get("field") or "下发线索数 (门店)",
                        "agg": denominator_metric.get("agg") or "sum",
                        "alias": denominator_metric.get("alias") or "门店线索数",
                    },
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                numerator_alias = numerator_metric.get("alias") or "门店当日锁单数"
                denominator_alias = denominator_metric.get("alias") or "门店线索数"
                weekly_missing_cols = [c for c in [statistics.get("time_field") or time_field, numerator_alias, denominator_alias] if c not in raw_df.columns]
                if weekly_missing_cols:
                    print(f"  ⚠️  weekly_decline_ratio 输入列缺失，降级为基础查询输出: {weekly_missing_cols}")
                    tool_result = {
                        "type": "weekly_decline_ratio",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": weekly_missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "weekly_decline_ratio",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_weeks": statistics.get("window_weeks") or 10,
                        "weekdays": statistics.get("weekdays") or [4, 5],
                        "numerator_alias": numerator_alias,
                        "denominator_alias": denominator_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "weekly_decline_ratio", "error": "statistics_execution_failed", "message": str(e)}
    elif stats_type == "daily_threshold_count" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.daily_threshold_count"}
        print("\n[Thinking] 执行统计型阈值计数分析...")
        value_metric = statistics.get("value_metric", {}) if isinstance(statistics, dict) else {}
        metric_alias = value_metric.get("alias") or metric.get("alias") or "value"
        if comparison_df is not None:
            comparison_metric_alias = f"{metric_alias}_diff"
            stat_time_field = (
                statistics.get("time_field")
                or (time_field if time_field in comparison_df.columns else None)
                or (dimensions[0] if isinstance(dimensions, list) and dimensions else None)
            )
            stat_metric_alias = statistics.get("metric_alias") or comparison_metric_alias
            pipeline_missing_cols = [c for c in [stat_time_field, stat_metric_alias] if c and c not in comparison_df.columns]
            if not stat_time_field:
                tool_result = {
                    "type": "daily_threshold_count",
                    "error": "invalid_statistics_input_schema",
                    "missing_columns": ["time_field"],
                }
            elif pipeline_missing_cols:
                print(f"  ⚠️  comparison→statistics 输入列缺失，返回结构化错误: {pipeline_missing_cols}")
                tool_result = {
                    "type": "daily_threshold_count",
                    "error": "invalid_statistics_input_schema",
                    "missing_columns": pipeline_missing_cols,
                }
            else:
                stat_request = {
                    "type": "daily_threshold_count",
                    "time_field": stat_time_field,
                    "window_days": statistics.get("window_days") or 30,
                    "date_start": time_start,
                    "date_end": time_end,
                    "op": statistics.get("op") or ">",
                    "threshold": statistics.get("threshold") if isinstance(statistics, dict) else 0,
                    "metric_alias": stat_metric_alias,
                }
                try:
                    tool_result = statistics_tool.perform_statistics(stat_request, comparison_df)
                except Exception as e:
                    tool_result = {"type": "daily_threshold_count", "error": "statistics_execution_failed", "message": str(e)}
        else:
            window_days = statistics.get("window_days") or 30
            try:
                window_days = int(window_days)
            except Exception:
                window_days = 30
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=max(int(window_days), 1))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": value_metric.get("field") or metric.get("field"),
                        "agg": value_metric.get("agg") or metric.get("agg") or "count",
                        "alias": metric_alias,
                    }
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                daily_missing_cols = [c for c in [statistics.get("time_field") or time_field, metric_alias] if c not in raw_df.columns]
                if daily_missing_cols:
                    print(f"  ⚠️  daily_threshold_count 输入列缺失，返回结构化错误: {daily_missing_cols}")
                    tool_result = {
                        "type": "daily_threshold_count",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": daily_missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "daily_threshold_count",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_days": window_days,
                        "date_start": query_time_start,
                        "date_end": query_time_end,
                        "op": statistics.get("op") or ">",
                        "threshold": statistics.get("threshold") if isinstance(statistics, dict) else 0,
                        "metric_alias": metric_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "daily_threshold_count", "error": "statistics_execution_failed", "message": str(e)}
    elif stats_type == "daily_mean" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.daily_mean"}
        print("\n[Thinking] 执行统计型日均分析...")
        value_metric = statistics.get("value_metric", {}) if isinstance(statistics, dict) else {}
        metric_alias = value_metric.get("alias") or metric.get("alias") or "value"
        if comparison_df is not None:
            tool_result = {
                "type": "daily_mean",
                "error": "unsupported_pipeline_input",
                "message": "daily_mean 暂不支持 comparison 联动，请使用单窗口查询。",
            }
        else:
            window_days = statistics.get("window_days") or 30
            try:
                window_days = int(window_days)
            except Exception:
                window_days = 30
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=max(int(window_days), 1))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": value_metric.get("field") or metric.get("field"),
                        "agg": value_metric.get("agg") or metric.get("agg") or "count",
                        "alias": metric_alias,
                    }
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                daily_missing_cols = [c for c in [statistics.get("time_field") or time_field, metric_alias] if c not in raw_df.columns]
                if daily_missing_cols:
                    print(f"  ⚠️  daily_mean 输入列缺失，返回结构化错误: {daily_missing_cols}")
                    tool_result = {
                        "type": "daily_mean",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": daily_missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "daily_mean",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_days": window_days,
                        "date_start": query_time_start,
                        "date_end": query_time_end,
                        "metric_alias": metric_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "daily_mean", "error": "statistics_execution_failed", "message": str(e)}
    elif stats_type == "daily_percentile_rank" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.daily_percentile_rank"}
        print("\n[Thinking] 执行统计型分位分析...")
        value_metric = statistics.get("value_metric", {}) if isinstance(statistics, dict) else {}
        metric_alias = value_metric.get("alias") or metric.get("alias") or "value"
        if comparison_df is not None:
            tool_result = {
                "type": "daily_percentile_rank",
                "error": "unsupported_pipeline_input",
                "message": "daily_percentile_rank 暂不支持 comparison 联动，请使用单窗口查询。",
            }
        else:
            window_days = statistics.get("window_days") or 30
            try:
                window_days = int(window_days)
            except Exception:
                window_days = 30
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=max(int(window_days), 1))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": value_metric.get("field") or metric.get("field"),
                        "agg": value_metric.get("agg") or metric.get("agg") or "count",
                        "alias": metric_alias,
                    }
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                daily_missing_cols = [c for c in [statistics.get("time_field") or time_field, metric_alias] if c not in raw_df.columns]
                if daily_missing_cols:
                    print(f"  ⚠️  daily_percentile_rank 输入列缺失，返回结构化错误: {daily_missing_cols}")
                    tool_result = {
                        "type": "daily_percentile_rank",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": daily_missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "daily_percentile_rank",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_days": window_days,
                        "date_start": query_time_start,
                        "date_end": query_time_end,
                        "reference_date": statistics.get("reference_date"),
                        "metric_alias": metric_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "daily_percentile_rank", "error": "statistics_execution_failed", "message": str(e)}

    elif stats_type == "weekend_percentile_rank" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.weekend_percentile_rank"}
        print("\n[Thinking] 执行周末分位分析...")
        value_metric = statistics.get("value_metric", {}) if isinstance(statistics, dict) else {}
        metric_alias = value_metric.get("alias") or metric.get("alias") or "value"
        if comparison_df is not None:
            tool_result = {
                "type": "weekend_percentile_rank",
                "error": "unsupported_pipeline_input",
                "message": "weekend_percentile_rank 暂不支持 comparison 联动，请使用单窗口查询。",
            }
        else:
            window_weekends = statistics.get("window_weekends") or 10
            try:
                window_weekends = int(window_weekends)
            except Exception:
                window_weekends = 10
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=(max(int(window_weekends), 1) * 7 + 7))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": value_metric.get("field") or metric.get("field"),
                        "agg": value_metric.get("agg") or metric.get("agg") or "count",
                        "alias": metric_alias,
                    }
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                weekend_missing_cols = [c for c in [statistics.get("time_field") or time_field, metric_alias] if c not in raw_df.columns]
                if weekend_missing_cols:
                    print(f"  ⚠️  weekend_percentile_rank 输入列缺失，返回结构化错误: {weekend_missing_cols}")
                    tool_result = {
                        "type": "weekend_percentile_rank",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": weekend_missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "weekend_percentile_rank",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_weekends": window_weekends,
                        "reference_date": statistics.get("reference_date"),
                        "metric_alias": metric_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "weekend_percentile_rank", "error": "statistics_execution_failed", "message": str(e)}

    elif stats_type == "weekday_percentile_rank" and tool_result is None:
        execution_meta = {"engine": "statistics", "route": "statistics.weekday_percentile_rank"}
        print("\n[Thinking] 执行指定周内日分位分析...")
        value_metric = statistics.get("value_metric", {}) if isinstance(statistics, dict) else {}
        metric_alias = value_metric.get("alias") or metric.get("alias") or "value"
        if comparison_df is not None:
            tool_result = {
                "type": "weekday_percentile_rank",
                "error": "unsupported_pipeline_input",
                "message": "weekday_percentile_rank 暂不支持 comparison 联动，请使用单窗口查询。",
            }
        else:
            window_weeks = statistics.get("window_weeks") or 10
            try:
                window_weeks = int(window_weeks)
            except Exception:
                window_weeks = 10
            weekdays = statistics.get("weekdays") if isinstance(statistics, dict) else None
            query_time_start = time_start
            query_time_end = time_end
            try:
                if time_end:
                    end_day = datetime.date.fromisoformat(str(time_end)[:10])
                    start_day = end_day - datetime.timedelta(days=(max(int(window_weeks), 1) * 7 + 7))
                    query_time_start = start_day.isoformat()
                    query_time_end = end_day.isoformat()
            except Exception:
                pass
            query_plan = {
                "dataset": dataset,
                "metrics": [
                    {
                        "field": value_metric.get("field") or metric.get("field"),
                        "agg": value_metric.get("agg") or metric.get("agg") or "count",
                        "alias": metric_alias,
                    }
                ],
                "dimensions": ([time_field] if time_field else []),
                "filters": [
                    *filters_without_time,
                    {"field": time_field, "op": ">=", "value": query_time_start},
                    {"field": time_field, "op": "<", "value": query_time_end},
                ]
                if time_field and query_time_start and query_time_end
                else filters_without_time,
            }
            raw_df = query_tool.execute_analysis_df(query_plan)
            if isinstance(raw_df, str):
                tool_result = raw_df
            else:
                missing_cols = [c for c in [statistics.get("time_field") or time_field, metric_alias] if c not in raw_df.columns]
                if missing_cols:
                    print(f"  ⚠️  weekday_percentile_rank 输入列缺失，返回结构化错误: {missing_cols}")
                    tool_result = {
                        "type": "weekday_percentile_rank",
                        "error": "invalid_statistics_input_schema",
                        "missing_columns": missing_cols,
                    }
                else:
                    stat_request = {
                        "type": "weekday_percentile_rank",
                        "time_field": statistics.get("time_field") or time_field,
                        "window_weeks": window_weeks,
                        "weekdays": weekdays,
                        "date_start": query_time_start,
                        "date_end": query_time_end,
                        "reference_date": statistics.get("reference_date"),
                        "metric_alias": metric_alias,
                    }
                    try:
                        tool_result = statistics_tool.perform_statistics(stat_request, raw_df)
                    except Exception as e:
                        tool_result = {"type": "weekday_percentile_rank", "error": "statistics_execution_failed", "message": str(e)}

    if tool_result is None:
        execution_meta = {"engine": "dsl", "route": "query_tool"}
        print("\n[Thinking] 执行单次查询...")
        query_plan = {
            "dataset": dataset,
            "metrics": [
                {
                    "field": metric.get("field"),
                    "agg": metric.get("agg"),
                    "alias": metric.get("alias") or metric.get("business_name") or "value",
                }
            ],
            "dimensions": dimensions,
            "filters": [
                *filters_without_time,
                {"field": time_field, "op": ">=", "value": time_start},
                {"field": time_field, "op": "<", "value": time_end},
            ]
            if time_field and time_start and time_end
            else filters_without_time,
        }
        tool_result = query_tool.execute_analysis(query_plan)

    if isinstance(tool_result, str) and ("找不到数据集" in tool_result or "聚合计算失败" in tool_result):
        print("  ⚠️  执行异常，尝试回退到关键词匹配...")
        fallback_question = plan.get("question") or user_query
        fallback_result = query_tool.answer_question(fallback_question)
        tool_result = f"执行遇到问题: {tool_result}\n\n尝试关键词匹配结果:\n{fallback_result}"

    print(f"[Route] 规划路由完成: {execution_meta['engine']}::{execution_meta['route']}")
    sub_query = plan.get("question") or user_query
    tool_result_text = json.dumps(tool_result, ensure_ascii=False, indent=2) if isinstance(tool_result, dict) else str(tool_result)
    block = f"查询: {sub_query}\nDSL: {json.dumps(plan, ensure_ascii=False)}\n执行结果:\n{tool_result_text}"
    return {"block": block, "execution_meta": execution_meta}


def run_dsl_step(
    action_query: str,
    planning_agent: PlanningAgent,
    query_tool: QueryTool,
    comparison_tool: ComparisonTool,
    statistics_tool: StatisticsTool,
    memory_context: dict | None = None,
) -> dict:
    print("\n[Thinking] PlanningAgent 正在构建执行规划并路由...")
    plan = planning_agent.create_plan(action_query, memory_context=memory_context)
    if not isinstance(plan, dict) or not plan:
        return {"status": "error", "message": "未能生成有效的规划 DSL。"}

    goal_time_window = None
    if isinstance(memory_context, dict):
        goal_time_window = memory_context.get("goal_time_window")

    if (
        goal_time_window
        and isinstance(goal_time_window, (tuple, list))
        and len(goal_time_window) == 2
        and isinstance(plan.get("time"), dict)
    ):
        try:
            goal_start = datetime.date.fromisoformat(str(goal_time_window[0])[:10])
            goal_end = datetime.date.fromisoformat(str(goal_time_window[1])[:10])
            time = plan.get("time") or {}
            time_start = datetime.date.fromisoformat(str(time.get("start"))[:10])
            time_end = datetime.date.fromisoformat(str(time.get("end"))[:10])

            clamped_start = max(time_start, goal_start)
            clamped_end = min(time_end, goal_end)
            if clamped_end <= clamped_start:
                return {
                    "status": "error",
                    "message": f"规划的时间窗口超出目标范围，且裁剪后无有效区间: [{clamped_start.isoformat()}, {clamped_end.isoformat()})",
                }

            if clamped_start != time_start or clamped_end != time_end:
                time["start"] = clamped_start.isoformat()
                time["end"] = clamped_end.isoformat()
                plan["time"] = time

                time_field = time.get("field")
                filters = plan.get("filters")
                if isinstance(time_field, str) and isinstance(filters, list) and filters:
                    rewritten_filters: list[dict] = []
                    for f in filters:
                        if not isinstance(f, dict):
                            continue
                        if f.get("field") != time_field:
                            rewritten_filters.append(f)
                            continue
                        op = f.get("op")
                        value = f.get("value")
                        if op in {">=", "<"} and isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value.strip()):
                            continue
                        rewritten_filters.append(f)
                    plan["filters"] = rewritten_filters
        except Exception:
            pass

    if isinstance(plan.get("clarification"), dict):
        clarification = plan["clarification"]
        if clarification.get("need"):
            return {
                "status": "clarification",
                "clarification": clarification,
                "original_question": plan.get("question") or action_query,
            }

    execution = _execute_single_plan(
        plan=plan,
        user_query=action_query,
        query_tool=query_tool,
        comparison_tool=comparison_tool,
        statistics_tool=statistics_tool,
        memory_context=memory_context,
    )
    return {
        "status": "ok",
        "result_blocks": [execution["block"]],
        "execution_meta": execution.get("execution_meta") or {},
    }
