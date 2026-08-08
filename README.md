# A股多因子选股系统 (a-stock-engine)

> 重装电脑后从历史对话记录重建。基于云端保存的7段对话摘要中的架构信息还原。

## 系统架构

```
a-stock-engine/
├── config/
│   └── config.yaml              # 配置文件（账户/环境/因子/输出）
├── src/
│   ├── westock_cli.py           # westock-data CLI 共享封装（已修复重复代码问题）
│   ├── kline_cache.py           # K线数据本地缓存
│   ├── local_price_loader.py    # 统一价格数据加载器
│   ├── multifactor.py           # 核心多因子选股引擎
│   ├── daily_brief.py           # 盘前简报生成器（自动化入口）
│   ├── verify_picks.py          # T+2推荐验证工具
│   ├── enrich_short.py          # 短线因子增强（RSI/KDJ/量比）
│   ├── sector_rotation_watchlist.py  # 板块轮动监控
│   └── reward_attribution.py    # 收益归因分析
├── briefs/                      # 简报输出目录（按日期归档）
├── data_cache/                  # 数据缓存目录
├── requirements.txt
└── README.md
```

## 分层过滤流程

```
全A股列表
  │
  ▼
L0 市场环境判断 ← 上证/深证/创业板 均线位置
  │  多头(80%) / 震荡(50%) / 空头(20%) 仓位上限
  ▼
L2 基础过滤 ← 排除ST/停牌/新股/小市值/异常PE/低成交额
  │
  ▼
L4 多因子评分 ← 价值(PE/PB) + 质量(ROE/毛利率/负债率)
  │              + 成长(营收增速/利润增速) + 动量(20日/60日)
  ▼
输出分类:
  ②A 质量榜   — 综合评分 Top 10
  ②B 短线榜   — 动量最强 + v5反弹引擎
  ③A 持仓     — 当前账户持仓
  ③B 操作建议 — 评分下降的持仓（建议减仓）
  ③C 观察名单 — 评分靠前的候选
```

## 快速开始

### 1. 安装依赖

```bash
cd a-stock-engine
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/config.yaml`：
- `account.holdings`: 填入当前持仓
- `account.available_cash`: 可用资金
- `factor_l4.factors`: 调整因子权重

### 3. 运行盘前选股

```bash
cd src
python daily_brief.py
```

输出: `briefs/YYYY-MM-DD/盘前选股简报.md`

### 4. T+2验证

```bash
# 验证近7天推荐
python verify_picks.py --range 7

# 验证指定日期
python verify_picks.py --date 2026-08-06

# 验证区间
python verify_picks.py --start 2026-07-27 --end 2026-07-29
```

### 5. 板块轮动监控（16:00运行）

```bash
python sector_rotation_watchlist.py
```

### 6. 收益归因分析

```bash
python reward_attribution.py
```

## 数据源

系统使用 **westock-data** CLI 获取行情和基本面数据。
westock-data 是 WorkBuddy 的内置技能，提供：
- A股全量股票列表
- K线数据（日/周/月线，前复权/后复权）
- 基本面数据（PE/PB/ROE/毛利率/营收增速等）
- 指数K线
- 板块行情和映射

数据缓存策略：本地缓存12小时，超时自动刷新。

## 已知问题 & 待办

> 以下问题在原系统代码审查中已识别，本次重建已修复部分：

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| 1 | bare except 泛滥 | ✅ 已修复 | 全部替换为具体异常类型 |
| 2 | run_westock/_kill_process_tree 重复 | ✅ 已修复 | 抽取到 westock_cli.py 统一管理 |
| 3 | sector_of() 重复 | ✅ 已修复 | 统一到 westock_cli.py |
| 4 | 研判文本stale bug | ✅ 已修复 | regime 直接从 assess_regime() 返回值取 |
| 5 | 三大指数接口返回NA | ⚠️ 待验证 | 添加了容错处理和数据不足时的默认逻辑 |
| 6 | CLI瞬时取数失败 | ⚠️ 待验证 | 添加了重试机制和缓存回退 |
| 7 | 单元测试 | ❌ 待实现 | 需要补充 |
| 8 | 因子参数需校准 | ❌ 待校准 | 原系统因子权重需要根据记忆重新调整 |

## 重建说明

本系统于 2026-08-08 从历史对话记录重建。原系统在 2026-07-17 至 2026-08-07 期间开发运行，
因电脑重装导致本地代码丢失。以下信息从云端对话历史中恢复：

- **架构设计**: L0/L2/L4分层过滤 + ②A/②B/③A/③C输出分类
- **文件结构**: 9个核心Python文件
- **数据接口**: westock-data CLI 封装
- **配置格式**: YAML配置，含账户/环境/因子/输出
- **历史绩效**: T+2验证15笔14正，平均+2.86%
- **自动化任务**: 09:00盘前 / 16:00板块 / 16:30追踪

**无法恢复的部分**（需要用户补充）：
- 具体的因子公式和参数（当前使用通用多因子模型）
- 持仓股票列表（config.yaml 中 holdings 为空）
- 可用资金金额
- 历史简报文件

## 自动化配置

原系统配置了3个定时任务，可在 WorkBuddy 中重新创建：

| 时间 | 任务 | 脚本 |
|------|------|------|
| 09:00 | 盘前选股 | `python src/daily_brief.py` |
| 16:00 | 板块轮动 | `python src/sector_rotation_watchlist.py` |
| 16:30 | 持仓追踪 | `python src/reward_attribution.py` |

## 技术栈

- Python 3.12+
- pandas / numpy（数据处理）
- PyYAML（配置文件）
- westock-data CLI（数据源）
