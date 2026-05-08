# 业务指标计算口径 (Business Operators)

本目录用于承接“强业务口径、不可用通用 DSL 可靠表达”的指标计算，并在 Agent 执行阶段通过算子注册中心做路由。

## 1. registry.py（算子注册/路由）

### 1.1 入口函数

- `run_registered_operator(plan: dict, user_query: str, query_tool) -> dict | None`
  - 返回 `None`：表示未命中任何固定算子，交回通用查询工具链路处理
  - 返回 `dict`：表示已命中算子，dict 为算子输出（或错误结构）

### 1.2 命中规则

- 在营门店（active_store）
  - 命中条件：`user_query` 或 `plan.metric` 文本中包含 “在营门店”
  - 数据依赖：`query_tool.datasets["order_data"]`（必须存在）
  - 时间依赖：`plan.time.start` / `plan.time.end`（必须存在）

- 留存小订（retained_intention）
  - 命中条件：`user_query` 或 `plan.metric` 文本中包含 “留存小订”
  - 数据依赖：`query_tool.datasets["order_data"]`（必须存在）
  - 时间依赖：`plan.time.start` / `plan.time.end`（必须存在）
  - 车系过滤：从 `plan.filters` 中提取 `field in ("series","series_group_logic")` 且 `op=="=="` 的 `value` 作为 `series`
  - 补充列：若 `order_data` 缺少 `series_group_logic`，则尝试读取 [schema/business_definition.json]并调用 `apply_series_group_logic(df, bdef)` 补齐

### 1.3 错误返回约定（registry 层）

- 缺少数据集：`{"type": "<operator>", "error": "dataset_not_found", "message": "..."}`
- 缺少时间窗口：`{"type": "<operator>", "error": "missing_time_window", "message": "..."}`

## 2. active_store.py（在营门店）

文件：[active_store.py]

### 2.1 业务定义

- 计算每天的“在营门店数”：对目标日期 `d`，若门店在 `d` 当天及往前 29 天（共 30 天窗口）存在订单活动，且门店开店日 `store_create_date <= d`，则计为在营。
- 输出为一个时间窗口内的日序列，以及窗口内最大/最小值信息。

### 2.2 输入要求

- `df`: `order_data`（订单明细），需要包含（或尽量包含）以下字段：
  - 门店：`store_name`
  - 开店日：`store_create_date`
  - 订单日期（二选一）：
    - `order_create_date`（可解析为日期）
    - `order_create_time`（datetime，将被 floor 到天）
- 时间窗口：`start` / `end`
  - `end` 按 `[start, end)`（左闭右开）解释：脚本内部会生成 `start_day ... end_day-1` 的日序列

### 2.3 主要函数

- `_to_day_series(df) -> pd.Series`：从 `order_create_date`/`order_create_time` 推导日粒度日期列
- `_calc_active_store_count(df, target_date) -> int`：计算某天 `target_date` 的在营门店数（30 天活动窗口 + 开店日约束）
- `run_active_store_operator(df, start, end) -> dict`：对窗口内每天计算，并返回摘要

### 2.4 输出结构

`run_active_store_operator` 返回：

- `type`: `"active_store"`
- `start`: 窗口起始日（YYYY-MM-DD）
- `end`: 窗口结束日（YYYY-MM-DD，开区间端点）
- `window_days`: 窗口天数
- `max_active_store_count` / `max_date`
- `min_active_store_count` / `min_date`
- `daily_rows`: `[{ "date": "YYYY-MM-DD", "active_store_count": int }, ...]`

异常：

- 时间解析失败：`{"type":"active_store","error":"invalid_time_range",...}`
- `end <= start`：同上

## 3. retained_intention.py（留存小订）

文件：[retained_intention.py]

### 3.1 业务定义

目标：在给定时间窗口内统计“留存小订单数（去重 order_number）”。

- 先定义窗口长度 `n_days`：
  - 若 `end` 是 `00:00:00` 的开区间端点，则业务截止日按 `end - 1 day` 解释
  - `n_days = (actual_end_day - start_day) + 1`（至少为 1）
  - 统计窗口：`[start_day, window_end_excl)`，其中 `window_end_excl = min(start_day + n_days, actual_end_day + 1 day)`
- 计入规则：
  - `intention_payment_time` 非空，且落在 `[start_day, window_end_excl)`
  - 且“未在窗口内退订”：`intention_refund_time` 为空，或 `intention_refund_time > window_end_excl`
- 若传入 `series`：
  - 优先用 `series_group_logic == series` 过滤
  - 否则用 `series == series` 过滤

### 3.2 输入要求

- `df`: `order_data`，需要字段：
  - `order_number`
  - `intention_payment_time`
  - `intention_refund_time`
  - （可选）`series_group_logic` 或 `series`（用于车系过滤）
- `start` / `end`: 时间字符串（可被 pandas 解析）

### 3.3 输出结构

返回：

- `type`: `"retained_intention"`
- `series`: 传入的车系（可能为 `None`）
- `start`: YYYY-MM-DD
- `end`: YYYY-MM-DD（实际业务截止日）
- `retained_count`: 去重后的留存小订数量（int）

异常：

- 数据为空：`{"error":"dataset_empty","message":"数据集为空"}`

## 4. series_group_logic.py（车系分组口径）

文件：[series_group_logic.py]

### 4.1 用途

为明细数据补充一列 `series_group_logic`，用于把 `product_name` 按业务定义映射成车系/车型组（如 LS8/LS9/CM2/其他）。

### 4.2 依赖数据

- 输入 df 需包含 `product_name`
- 规则来自 [schema/business_definition.json]的 `series_group_logic` 字典
  - key：输出车系名称（例如 `"LS8"`、`"其他"`）
  - value：类 SQL 的表达式字符串，支持 `LIKE` / `NOT LIKE`，以及 `OR`/`AND` 组合

### 4.3 行为约定

- 若 df 已存在 `series_group_logic`：直接返回
- 若缺少 `product_name` 或缺少/非法规则：生成 `series_group_logic = NA` 并返回
- 默认初始为 `"其他"`，按规则顺序覆盖命中的记录（key 为 `"其他"` 的规则会被跳过）

### 4.4 主要函数

- `_eval_series_group_logic_expr(product_name: pd.Series, expr: str) -> pd.Series[bool]`
- `apply_series_group_logic(df: pd.DataFrame, business_definition: dict) -> pd.DataFrame`
