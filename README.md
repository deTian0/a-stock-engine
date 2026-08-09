# A股多因子选股系统 (a-stock-engine)

> 前后端分离 + Docker 部署 + CLI，基于本地数据库 + 新浪实时行情

## 项目结构

```
a-stock-engine/
├── src/                          # Python 源码
│   ├── data/                     # 数据层
│   │   ├── db.py                 # SQLite 数据库（market.db + selections.db）
│   │   ├── sina.py               # 新浪 API 数据源（默认，零依赖）
│   │   ├── akshare.py            # akshare 数据源（备用）
│   │   ├── westock.py            # westock-data CLI 封装
│   │   └── local.py              # 本地 Parquet 价格加载器
│   ├── engine/                   # 引擎层
│   │   ├── factors.py            # 多因子评分引擎
│   │   ├── selection.py          # 选股流水线（L0/L2/L4）
│   │   ├── backtest.py           # 回测引擎（T+N 验证）
│   │   ├── verify.py             # 选股验证
│   │   └── cache.py              # K线本地缓存
│   ├── tasks/                    # 任务层
│   │   ├── daily.py              # 盘前选股简报
│   │   ├── sector.py             # 板块轮动监控
│   │   └── reward.py             # 收益归因分析
│   └── utils/                    # 工具层
│       ├── rotation.py           # 轮动追踪
│       ├── enrich.py             # 短线因子增强
│       └── guard.py              # 运行守卫
├── api/                          # Flask 后端
│   ├── app.py                    # 主入口
│   ├── data_api.py               # 数据概况 API
│   ├── select_api.py             # 选股 API
│   ├── backtest_api.py           # 回测 API（异步）
│   └── services/                 # 业务逻辑层
├── front/                        # React 前端
│   └── src/pages/
│       ├── Dashboard.jsx         # 概览
│       ├── Selector.jsx          # 选股
│       └── Backtest.jsx          # 回测
├── docker/                       # 容器化部署
│   ├── docker-compose.yml        # 一键启动
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── config.yaml                   # 统一配置
├── local_backtest.py             # 回测入口（6年/20年）
└── import_local_data.py          # 数据导入
```

## 快速开始

### 开发模式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 后端
python api/app.py                    # → http://localhost:5000

# 3. 前端（新终端）
cd front && npm install && npm run dev  # → http://localhost:3000
```

### 命令行模式

```bash
# 选股（新浪 API，实时数据）
cd src
python tasks/daily.py

# 回测（本地数据库）
python local_backtest.py

# API 直接调用
curl -X POST http://localhost:5000/api/select/run \
  -H "Content-Type: application/json" \
  -d '{"top_n":20}'
```

### Docker 一键部署

```bash
docker-compose -f docker/docker-compose.yml up -d
# 访问 http://localhost
```

## 配置

`config.yaml` — 修改参数不动代码：

```yaml
data:
  source: sina                  # 数据源: sina | akshare | westock

portfolio:
  initial_capital: 50000        # 初始资金
  max_picks_per_day: 20         # 每日最多持仓

risk:
  stop_loss: 5.0                # 止损 (%)
  target_base: 3.0              # 基础目标收益 (%)

trade:
  stock_cost: 0.0013            # 股票成本 (万0.854+印花税)
  etf_cost: 0.00017             # ETF成本 (免印花税)
```

## 分层选股流程

```
全A股 (5190只) + ETF (1445只)
  │
  ▼
L0 市场环境 → MA60 择时（空头减仓/多头满仓）
  │
  ▼
L2 基础过滤 → 排除 ST/新股/小市值/异常 PE
  │
  ▼
L4 多因子评分 → 价值 + 质量 + 成长 + 动量 + 低波
  │
  ▼
输出: 精选 Top N（含目标价/止损价/持仓周期）
```

## 回测性能（v4.5, 2020-2026）

| 指标 | 数值 |
|------|------|
| 初始资金 | ¥50,000 |
| 最终资金 | ¥248,315 |
| 年化收益 | +30.3% |
| 夏普比率 | 1.45 |
| 最大回撤 | -14.7% |
| 交易笔数 | 15,626 |

## 数据源

| 数据源 | 类型 | 需要网络 | 说明 |
|--------|------|----------|------|
| 新浪 API | 实时行情 | 是 | 默认，5190只股票秒出，零依赖 |
| akshare | 实时行情 | 是 | 备用，需代理/反爬会失效 |
| westock-data | 实时行情 | 是 | WorkBuddy 内置 skill |
| 本地 DB | 历史数据 | 否 | Parquet + 后复权，2020-2026 |

通过 `config.yaml` → `data.source` 切换。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/data/summary | 数据概况 |
| GET | /api/data/market-regime | 市场环境（牛/熊） |
| POST | /api/select/run | 执行选股 |
| GET | /api/select/latest | 最近选股结果 |
| POST | /api/backtest/start | 启动回测（异步） |
| GET | /api/backtest/status/<id> | 查询回测进度 |
