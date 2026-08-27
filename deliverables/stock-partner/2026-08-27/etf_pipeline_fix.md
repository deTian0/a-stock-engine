# ETF 日线数据管道修复报告（2026-08-27）

## 一、问题回顾
早盘「ETF 推荐」连续多天**逐字节相同**——同一 8 只 ETF、同一组动量值（如 6.4% / -3.0%），
20 日动量不可能 16 天纹丝不动。这是真 bug，不是显示问题。

## 二、根因（数据管道层）
`market.db.daily_price` 中 **ETF 数据冻结在 2026-03-13**，而同期股票数据已更新到 08 月。

唯一会刷新 `daily_price` 的每日任务是 `afternoon_review.main()` 里的 `_batch_preload_prices`，
它存在三个断点，导致 ETF 永不被纳入刷新：

| # | 断点 | 后果 |
|---|------|------|
| 1 | 刷新 universe 仅来自 `cli.get_stock_list()`（股票池），**不含 ETF** | ETF 根本不在待刷新列表 |
| 2 | 只对**缺失 code** 做补全；已存在的冻结 ETF 被跳过 | 03-13 旧行永远留着 |
| 3 | 后缀逻辑 `5/6→SH, 0/2/3→SZ, 其他→BJ`，把 ETF 错标 `.BJ` | 即使回退也写错 code |

而 `daily_brief`（早盘）自己从不刷新，完全依赖前一日下午的写入 → ETF 数据持续陈旧。

## 三、修复方案
1. **`database.py` 新增 `delete_prices_for_codes()`**：按 纯码 / `.SZ` / `.SH` 三态清空某 code 全部日线，
   避免新旧行合并成带巨大时间缺口的序列（否则 `_batch_calc_momentum` 动量算错）。

2. **`multifactor.py` 抽出 `WELL_KNOWN_ETFS` 模块常量 + 新增 `refresh_etf_daily_prices()`**：
   - 逐只拉取当日实时 K 线（`LocalPriceLoader` → KlineCache → westock CLI）；
   - **「取数成功才删旧 + 写新」**（网络抖动也不丢数据：失败项保留旧行，由新鲜度闸门回退实时）；
   - 用正确 `.SH` / `.SZ` 后缀 upsert，杜绝新旧行合并与 `.BJ` 错标。

3. **接线到两个每日任务**，让 ETF 日线纳入刷新：
   - `afternoon_review.main()`（每日刷新主入口）
   - `daily_brief.main()`（早盘自包含，不依赖前一日）

4. **`select_etfs` 复用 `WELL_KNOWN_ETFS` 常量**，并修正成交额查询兼容纯码形态（防御性）。

> 配套防御层（上一轮已落地）：`select_etfs` 已加 `as_of` 新鲜度闸门 + 实时回退，
> 即使刷新偶发失败也不会再输出冻结假值。

## 四、验证证据
- **真实刷新一次**：`refresh_etf_daily_prices()` → `{'refreshed': 14, 'failed': []}`
  - 清掉冻结旧行 **19,915 行**，重新写入 **14 只 ETF** 当日日线；
  - `daily_price` 中 ETF 最新日期：**2026-03-13 → 2026-08-27**。
- **早盘 ETF 推荐已解冻**（`select_etfs` 输出，数据日期 2026-08-27）：

  | code | name | 动量20日 | 动量60日 |
  |------|------|---------|---------|
  | 512480 | 半导体ETF | 9.79% | -2.41% |
  | 515050 | 5GETF | 18.86% | -17.64% |
  | 159845 | 中证1000 | 11.64% | -7.93% |
  | 512100 | 中证1000ETF | 11.46% | -7.94% |
  | 510500 | 中证500ETF | 8.13% | -4.49% |
  | 588000 | 科创50ETF | 6.29% | -2.47% |
  | 159995 | 芯片ETF | 8.33% | -5.65% |
  | 512880 | 证券ETF | -0.54% | 6.15% |

  （修复前这些数值恒为 6.4% / -3.0% 等冻结假值，逐日相同。）

- **单测**：`tests/test_multifactor.py` 新增 3 个用例（真实 SQL 删三态 / 刷新串联后缀 / 失败项记录），
  全量 **161 passed**，无回归。

## 五、影响范围
- 改动文件：`src/database.py`、`src/multifactor.py`、`src/afternoon_review.py`、`src/daily_brief.py`、`tests/test_multifactor.py`。
- 行为变化：每日早晚两次任务都会刷新 14 只主流 ETF 日线；ETF 推荐自此随真实行情每日变化。
- 风险：取数完全失败时单个 ETF 保留旧行（由新鲜度闸门标记过期并回退实时），**不会清空数据**。

## 六、后续建议
- 将 `WELL_KNOWN_ETFS` 与 `config.yaml` 打通，支持用户自定义 ETF 池。
- 监控每日 `refresh_etf_daily_prices` 返回的 `failed` 列表，连续失败需告警（CLI/网络侧）。
