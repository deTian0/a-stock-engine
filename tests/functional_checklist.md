# a-stock-engine 功能清单 (Functional Checklist)

> 生成日期：2026-08-26 | 用途：作为 `tests/` 测试套件的覆盖依据
> 范围：枚举核心公开方法的**功能 / 输入 / 输出 / 依赖 / 边界条件 / 异常分支**，并映射测试覆盖。
> 说明：工程为 Python（pytest），非 Java/JUnit；用户所述"JUnit 测试"在本工程以等价 pytest 实现。

---

## 1. 测试环境与约定

- 解释器：`D:/env/python3.12/python.exe`（含 pandas / numpy / pyyaml）
- 框架：pytest 9.x
- 网络函数（westock_helpers / akshare_provider / tushare_provider）**一律用假子进程 stdout 或 mock 替代**，保证离线确定性。
- 纯函数（risk_module.allocate_basket、lvrev_scorer.score_lvrev、daily_brief.generate_brief）直接构造桩数据测试。
- 提交纪律：每完成一个小变动即 `commit + push`，每次 commit 后 `git rev-parse HEAD` 核验 ref 真前进（防本机 packed-refs 陷阱）。

---

## 2. 核心已测模块详述

### 2.1 `risk_module.py`
| 方法 | 功能 | 输入 | 输出 | 边界/异常 |
|---|---|---|---|---|
| `allocate_basket(scores, budget, method="score_weighted", max_single=None)` | 按评分/等权分配仓位预算 | `scores: list[float]`, `budget: float`(0~1), `method: "score_weighted"\|"equal"`, `max_single: float\|None` | `list[float]` 各票权重，和为 budget | 空→`[]`；budget≤0→全 0；全零评分→退回等权；未知 method→退回等权；`max_single` 仅硬性截断**原超限项**，余量补给未超限项（不二次归一截断） |
| `enrich_risk_metrics(df, regime_cap=0.6, total_capital=1e6)` | 计算 ATR 止损/流动性标签/板块集中度警告 | df 含 `code,composite_score,sector,amount,close` | 增加 `stop_loss/atr14/liquidity_tag/sector_warning` 列 | 空 df→空；依赖 `batch_kline`（mock）；同板块≥3 只触发"同板块N只"警告 |

### 2.2 `lvrev_scorer.py`
| 方法 | 功能 | 输入 | 输出 | 边界/异常 |
|---|---|---|---|---|
| `score_lvrev(df, value_factor=False, ...)` | lvrev 内核打分（低波+反转+质量+成长） | 含 `vol20,reversal20,...` 的 df | 增加 `composite_score`(百分制) 列 | 缺列→依赖 df 预处理；空 df→空 |
| `apply_entry_gates(df, reversal_q=0.30)` | 入场闸门（低波+反转双重门控） | df + `composite_score` | `Series[bool]` entry_ok | 缺失项→False |

### 2.3 `daily_brief.py`
| 方法 | 功能 | 输入 | 输出 | 边界/异常 |
|---|---|---|---|---|
| `generate_brief(results, config)` | 生成盘前/盘后 Markdown 简报 | `results`(regime/categories/etf_picks/holding_prices/timestamp...), `config` | str(Markdown) | 持仓追踪优先用 `results["holding_prices"]` 实时价；缺失→回退选股候选 `close`；再缺失→回退 `config.account.holdings.cost_price`（旧 bug 已修）；无注入不崩溃 |
| `save_brief(content, config)` | 落盘简报到 `history/YYYY-MM-DD/` | Markdown 文本 | Path | IO 异常上抛 |
| `_premarket_window_ok(config)` | 盘前护栏（09:30 截止） | config(含时间) | `(bool, str)` | 越界→拦截 |
| `main()` | 编排：取数→评分→生成→落盘 | CLI args | 退出码 | 引擎失败→exit(1) |

**持仓追踪修复点（本任务关键 bug）**：旧逻辑 `cur_price = cur_price_map.get(code, cost)` 对不在选股候选的持仓 ETF 直接回退 `cost_price` 当"当日股价"，导致股价=成本、盈亏恒 +0.0%。修复：`main()` 用 `batch_quotes` 单独拉实时价注入 `holding_prices`，`generate_brief` 优先采用。

### 2.4 `westock_helpers.py`
| 方法 | 功能 | 输入 | 输出 | 边界/异常 |
|---|---|---|---|---|
| `batch_quotes(codes)` | 批量实时行情（经 `npx westock` 子进程） | `list[code]` | `{code: {close,...}}` | 子进程失败→异常；经 `_to_ws` 加前缀 |
| `_to_ws(code)` | 6 位码→`sh/sz/bj` 前缀（**已修 bug**） | `str` | `sh\|sz\|bj + code` | 沪 `5/6/9` 开头、深 `0/1/2/3` 开头、北交所 `4/8` 开头；**旧实现误将 `1xxxx`(深 ETF) 与 `5xxxx`(沪 ETF) 都归 `sz/bj`，导致持仓 live 价在生产失效**——已修正 |
| `batch_kline(code, days)` | 批量 K 线（mock 用） | code | K 线 dict | 子进程失败→异常 |

### 2.5 集成 `tests/test_integration.py`
- 串联 `score_lvrev` → `allocate_basket`：正常/边界/异常数据流。
- `generate_brief` 完整简报：注入 `holding_prices`、空头/多头 regime、盘前/盘后 session。
- 验证不崩溃、输出结构完整、关键文字存在。

### 2.6 三套数据校验 `tests/test_datasets.py`
| 数据集 | 特征 | 验证目标 |
|---|---|---|
| A 多头全候选 | regime 多头 + 完整候选列 | 正常路径稳定 |
| B 空头空候选 | regime 空头 + 候选为空 | 空数据不崩溃 |
| C 脏数据 | 缺 `composite_score/roe/sector/close` 等可选列 | `.get` 兜底/异常分支健壮 |

---

## 3. 全模块公开方法索引（测试映射）

| 模块 | 关键公开方法 | 用途 | 已单测 |
|---|---|---|---|
| risk_module | allocate_basket, enrich_risk_metrics | 仓位分配 / 风控标签 | ✅ |
| lvrev_scorer | score_lvrev, apply_entry_gates | lvrev 内核打分 / 入场闸门 | ✅(score_lvrev 集成覆盖) |
| daily_brief | generate_brief, save_brief, main, _premarket_window_ok | 简报生成 / 落盘 / 护栏 | ✅(generate_brief) |
| westock_helpers | batch_quotes, batch_kline, _to_ws | 行情/前缀 | ✅ |
| multifactor | MultiFactorEngine.assess_regime/filter_l2/... | 多因子引擎 | 部分(集成) |
| afternoon_review | review_sectors, generate_review_html, main | 盘后复盘 HTML | 待补 |
| backtest | run_backtest, generate_report | 回测引擎 | 待补 |
| verify_picks | (T+2 验证) | 命中验证 | 待补 |
| reward_attribution | (收益归因) | 归因 | 待补 |
| factor_engine | score_stocks, apply_risk_gates, pick_top_by_sector | 因子打分 | 待补 |
| data_enricher | enrich_l4_results, grade_signal | L4 富集 | 待补 |
| enrich_short | calc_rsi, calc_kdj, enrich | 短线指标 | 待补 |
| local_price_loader | get_price, calc_* | 本地价/指标 | 待补 |
| akshare_provider/tushare_provider | get_* | 数据源 | mock 预留 |

---

## 4. 测试覆盖矩阵

| 测试文件 | 用例数 | 覆盖 | 边界 | 异常 |
|---|---|---|---|---|
| test_risk_module.py | 13 | allocate 正常/equal/未知method | 空/零/负预算/全零评分 | max_single 截断再分配 |
| test_lvrev_scorer.py | 6 | score_lvrev 打分排序 | 空 df | 异常分支 |
| test_daily_brief.py | ~8 | generate_brief 各节/持仓修复 | 空候选/无注入 | 缺失字段兜底 |
| test_westock_helpers.py | ~8 | batch_quotes/mock 子进程/_to_ws 各前缀 | 空列表 | 子进程失败 |
| test_integration.py | 5 | 管线串联/盘前盘后/空头多头 | — | 脏数据 |
| test_datasets.py | 3 数据集 | 三套数据校验 | — | 缺列兜底 |
| **合计** | **45** | **全绿** | | |

---

## 5. 本轮测试发现的真实 Bug（已修）

1. **`westock_helpers._to_ws` 代码前缀映射错误**：将深市 ETF `1xxxx` 与沪市 ETF `5xxxx` 错判，导致持仓 live 价在生产环境查不到、回退成本（持仓追踪修复点失效）。已修正为按沪深北交易所规则映射。
2. **`daily_brief` 持仓追踪回退成本当股价**（前序任务）：已修，本报告 Task 1 已提交。

---

## 6. 后续测试扩展建议（非本次范围）

- afternoon_review / backtest / verify_picks / reward_attribution / factor_engine 的单元与集成测试。
- `guard` 磁盘/信号保护、`health_check` 各源连通性（mock 外部）。
- CI 集成：每次 push 跑 `pytest tests/ -q`。
