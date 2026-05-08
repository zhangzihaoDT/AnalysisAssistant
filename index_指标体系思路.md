锁单指标体系梳理（目标→结构→驱动）

一、目标层（Target）

- metric: lock_orders（锁单数）
- 核心目标指标
  - 锁单数（ordernumber）
  - 锁单人数（md5/ownerphone）
  - 锁单率
    - 线索当日锁单率
    - 7日锁单率
    - 30日锁单率

二、维度层（Dimension）

- channel（渠道）
- store（门店）
- model（车型）
- city（城市）

三、结构层（Structure）
3.1 渠道结构（channel / attribution）

- 分渠道线索量
- 分渠道锁单量
- 分渠道锁单率（渠道锁单 / 渠道线索）
- 本渠道锁单构成
  - 直接锁单（Direct）
  - 归因锁单（Attributed）
- 辅助锁单构成
  - 首触辅助（First Touch）
  - 过程助攻（Middle）
  - 锁后触达（Post-Lock）
- 跨渠道助攻来源（assist channel Top）

  3.2 漏斗结构（funnel）

- 漏斗主链路：线索 → 试驾 → 锁单 → 开票
- 漏斗效率
  - 线索当日试驾率
  - 门店线索当日锁单率
  - 7日锁单率
  - 30日锁单率
  - 开票率（开票 / 锁单）

  3.3 用户路径结构（user journey / attribution）

- 平均触达次数
- 平均转化时长（天）
- 用户路径分类占比
  - One-Touch（Decisive）
  - Hesitant（Same Channel, Multiple Touches）
  - Cross-Channel（Comparison Shopper）
  - Long Consideration（>14 Days）
  - Repeat Lockers（Had Prior Locks）

  3.4 产品结构（product_mix）

- 车型结构
  - share_l6
  - share_ls6
  - share_ls9
- 动力结构
  - share_reev
  - share_ls6_reev

四、驱动层（Drivers）
4.1 价格驱动（price）

- ATP（平均成交价）
- 价格变化率（环比）
- 价格变化对 lock_orders 弹性（分车型/分城市）

  4.2 活动驱动（campaign）

- 活动期 vs 基准期（非活动期）对比
- 预售窗口 / 上市窗口对比（按 business_definition 活动窗口）
- 活动对漏斗环节的影响拆解（线索、试驾、锁单）

  4.3 门店驱动（store_performance）

- 在营门店数（门店规模）
- 门店锁单数、店均锁单数
- 门店线索数、门店线索占比
- 门店线索当日锁单率
- 门店集中度（CR5）
- 城市集中度（城市CR5）

五、2026-03-10 样例映射（来自 index_summary.py）
5.1 目标层样例

- 锁单数：148
- 锁单人数：148
- 开票数：170

  5.2 结构层样例

- 渠道结构
  - 锁单用户主要渠道Top：自然客流 41.0%、直接大定 9.7%、汽车之家车商汇 8.3%、新媒体-抖音 8.3%
  - 跨渠道助攻Top：自然客流、抖音-矩阵、APP行为激活、地推（门店）、APP
- 漏斗结构
  - 下发线索数：11317
  - 下发线索当日试驾率：6.8%
  - 下发（门店）线索当日锁单率：2.0%
  - 下发线索7日锁单率：1.0%
  - 下发线索30日锁单率：1.0%
  - 有效试驾数：1395
- 路径结构
  - 平均触达次数：2.27
  - 平均转化时长：5.93天
  - One-Touch：43.1%
  - Cross-Channel：50.0%
- 产品结构
  - share_l6：13.5%
  - share_ls6：58.1%
  - share_ls9：27.7%
  - share_reev：52.0%

  5.3 驱动层样例

- 价格：整体ATP（用户车）24.03万元
- 活动：需按活动窗口与基线做对比评估
- 门店：
  - 在营门店数：392
  - 店均锁单数：0.38
  - 门店CR5：7.4%
  - 城市CR5：23.6%
  - 店日均下发线索数：28.94

六、最终树形表达
锁单（lock_orders）
├─ 渠道（channel）
├─ 漏斗（funnel）
├─ 用户路径（attribution/journey）
├─ 产品结构（product_mix）
└─ 驱动因素（drivers）
├─ 价格（price）
├─ 活动（campaign）
└─ 门店（store_performance）
