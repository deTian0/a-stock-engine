"""
rotation_tracker.py — 板块轮动选股追踪器

功能:
  - 盘前/盘后双时段选股
  - 按板块分组，每板块 Top 5 股票 + Top 5 ETF
  - 两周滑动窗口追踪
  - 累计命中次数统计
  - 全部结果写入 SQLite

用法:
  python rotation_tracker.py --session pre_market   # 盘前选股
  python rotation_tracker.py --session post_market  # 盘后选股
  python rotation_tracker.py --report               # 查看累计命中报告

CPU 安全:
  - 所有获取操作串行执行（不使用线程池）
  - 每次 API 调用间隔 100ms
  - 单个板块处理失败不影响其他板块
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.westock import get_cli
from data.db import get_db
from utils.guard import setup_protection, teardown_protection, setup_logging
from engine.factors import score_stocks as factor_score, pick_top_by_sector, filter_candidates

logger = logging.getLogger(__name__)

# 每板块最大数量
MAX_PER_SECTOR_STOCK = 5
MAX_PER_SECTOR_ETF = 5
# 两周窗口
ROTATION_WINDOW_DAYS = 14
# CPU 保护：每次操作间隔
FETCH_INTERVAL_SEC = 0.1


class RotationTracker:
    """板块轮动选股追踪器——CPU 安全版。"""

    def __init__(self, session_type: str = "pre_market", config_path: str = None):
        self.session_type = session_type
        self.cli = get_cli()
        self.db = get_db()

        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config/config.yaml"
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        rot_cfg = self.config.get("rotation", {})
        self.max_per_sector = rot_cfg.get("max_per_sector_stock", MAX_PER_SECTOR_STOCK)
        self.max_etf = rot_cfg.get("max_per_sector_etf", MAX_PER_SECTOR_ETF)
        self.window_days = rot_cfg.get("window_days", ROTATION_WINDOW_DAYS)

        # 板块映射（懒加载）
        self._sector_map: Optional[dict] = None
        self._sector_list: Optional[pd.DataFrame] = None

    @property
    def sector_map(self) -> dict:
        if self._sector_map is None:
            try:
                self._sector_map = self.cli.get_sector_mapping()
                logger.info(f"板块映射加载: {len(self._sector_map)} 条")
            except Exception as e:
                logger.warning(f"板块映射加载失败: {e}")
                self._sector_map = {}
        return self._sector_map

    def _cpu_safe_wait(self):
        """CPU 保护：每次操作后短暂暂停。"""
        time.sleep(FETCH_INTERVAL_SEC)

    def get_stock_data(self) -> pd.DataFrame:
        """获取全市场股票数据（含板块信息）。"""
        logger.info("获取全市场股票数据...")
        try:
            df = self.cli.get_stock_list()
            if df is None or len(df) == 0:
                logger.error("获取股票列表为空")
                return pd.DataFrame()
            logger.info(f"获取到 {len(df)} 只股票")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

        # 附加板块信息
        smap = self.sector_map
        if smap and "code" in df.columns:
            df["sector"] = df["code"].map(smap).fillna("其他")

        return df

    def get_etf_data(self) -> pd.DataFrame:
        """获取 ETF 数据。使用 westock-tool label + ranking。"""
        logger.info("获取 ETF 数据...")
        try:
            import subprocess, json
            node = r"C:\Users\63516\.workbuddy\binaries\node\versions\22.22.2\node.exe"
            tool_js = (r"D:\soft\dev\WorkBuddy\resources\app.asar.unpacked"
                       r"\resources\builtin-skills\westock-tool\scripts\index.js")

            result = subprocess.run(
                [node, tool_js, "ranking", "size", "--asset", "etf",
                 "--limit", "100", "--raw"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"ETF 数据获取失败: {result.stderr[:200]}")
                return pd.DataFrame()

            data = json.loads(result.stdout)
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            # 标准化
            df = df.rename(columns={
                "代码": "code", "名称": "name", "最新价": "close",
                "涨跌幅": "change_pct", "成交额": "amount",
            })
            df["asset_type"] = "etf"
            df["sector"] = df.get("name", "").apply(
                lambda x: x[:2] if isinstance(x, str) and len(x) >= 2 else "ETF"
            )
            logger.info(f"获取到 {len(df)} 只 ETF")
            return df
        except Exception as e:
            logger.error(f"获取 ETF 数据失败: {e}")
            return pd.DataFrame()

    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """L2 过滤（委托 factor_engine）。"""
        return filter_candidates(df)

    def score_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """多因子评分（委托 factor_engine）。"""
        return factor_score(df)

    def pick_by_sector(self, df: pd.DataFrame, max_per_sector: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per_sector)

    def run(self) -> dict:
        """主流程：获取数据 → 过滤 → 评分 → 按板块选股 → 入库。"""
        window_end = datetime.now().strftime("%Y-%m-%d")
        window_start = (datetime.now() - timedelta(days=self.window_days)).strftime("%Y-%m-%d")

        logger.info(f"板块轮动追踪: {self.session_type}  |  窗口: {window_start} ~ {window_end}")

        # 1. 股票数据
        stocks = self.get_stock_data()
        self._cpu_safe_wait()

        stocks = self.filter_and_clean(stocks)
        stocks = self.score_stocks(stocks)

        # 2. ETF 数据
        etfs = self.get_etf_data()
        self._cpu_safe_wait()

        etfs = self.filter_and_clean(etfs)
        if len(etfs) > 0:
            etfs = self.score_stocks(etfs)

        # 3. 按板块选股
        stock_picks = self.pick_by_sector(stocks, self.max_per_sector)
        etf_picks = self.pick_by_sector(etfs, self.max_etf)
        all_picks = stock_picks + etf_picks

        logger.info(f"选中 {len(stock_picks)} 只股票 + {len(etf_picks)} 只 ETF")
        if not all_picks:
            return {"error": "无选中标的", "picks": []}

        # 4. 入库：轮动追踪
        self.db.save_rotation_picks(
            all_picks, window_start, window_end, self.session_type
        )

        # 5. 入库：累计命中
        seen = set()
        for p in all_picks:
            key = (p["code"], self.session_type)
            if key not in seen:
                seen.add(key)
                try:
                    self.db.upsert_pick_frequency(p["code"], self.session_type)
                except Exception as e:
                    logger.debug(f"更新命中频率失败 {p['code']}: {e}")

        # 6. 统计
        sector_counts = {}
        for p in all_picks:
            s = p["sector"]
            sector_counts[s] = sector_counts.get(s, 0) + 1

        results = {
            "session_type": self.session_type,
            "window": f"{window_start}~{window_end}",
            "stock_count": len(stock_picks),
            "etf_count": len(etf_picks),
            "total_count": len(all_picks),
            "sectors_covered": len(sector_counts),
            "sector_breakdown": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
            "picks": all_picks,
        }

        return results

    def generate_report(self, results: dict) -> str:
        """生成盘前/盘后简报。"""
        session_label = "盘前" if self.session_type == "pre_market" else "盘后"
        lines = [
            f"# {session_label}板块轮动选股\n",
            f"> 窗口: {results.get('window', '-')} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**选中**: {results.get('total_count', 0)} 只 "
            f"({results.get('stock_count', 0)} 股票 + {results.get('etf_count', 0)} ETF) | "
            f"覆盖 {results.get('sectors_covered', 0)} 个板块\n",
        ]

        # 板块分布
        breakdown = results.get("sector_breakdown", {})
        if breakdown:
            lines.append("## 板块分布\n")
            lines.append("| 板块 | 选中数 |")
            lines.append("|------|--------|")
            for sector, count in list(breakdown.items())[:15]:
                lines.append(f"| {sector} | {count} |")
            lines.append("")

        # 选中标的一览
        picks = results.get("picks", [])
        if picks:
            lines.append("## 选中标的\n")
            lines.append("| 板块 | 代码 | 名称 | 类型 | 评分 | 排名 |")
            lines.append("|------|------|------|------|------|------|")
            for p in picks[:50]:
                atype = "ETF" if p.get("asset_type") == "etf" else "股"
                lines.append(
                    f"| {p.get('sector', '-')} | {p['code']} | {p.get('name', '-')} | "
                    f"{atype} | {p.get('score', 0)} | #{p.get('rank', 0)} |"
                )

        # 累计命中 Top
        try:
            freq = self.db.get_pick_frequency(self.session_type, min_hits=1)
            if len(freq) > 0:
                lines.append(f"\n## 累计命中 Top 10 ({self.session_type})\n")
                lines.append("| 代码 | 命中次数 | 最近命中 |")
                lines.append("|------|---------|----------|")
                for _, row in freq.head(10).iterrows():
                    lines.append(
                        f"| {row['code']} | {row['total_hits']} | {row.get('last_hit_date', '-')} |"
                    )
        except Exception:
            pass

        lines.append(f"\n---\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="板块轮动选股追踪器")
    parser.add_argument("--session", default="pre_market",
                        choices=["pre_market", "post_market"],
                        help="选股时段 (默认 pre_market)")
    parser.add_argument("--report", action="store_true",
                        help="仅查看累计命中报告")
    args = parser.parse_args()

    setup_protection()
    setup_logging()

    tracker = RotationTracker(session_type=args.session)

    try:
        if args.report:
            results = tracker.db.get_pick_frequency(args.session)
            if len(results) == 0:
                print(f"暂无 {args.session} 命中记录")
                return
            print(f"\n{args.session} 累计命中 Top 20:")
            print(results[["code", "total_hits", "last_hit_date"]].head(20).to_string(index=False))
            return

        # 运行选股
        results = tracker.run()

        if "error" in results:
            print(f"ERROR: {results['error']}")
            return

        # 生成报告
        report = tracker.generate_report(results)
        session_tag = "盘前" if args.session == "pre_market" else "盘后"
        today = datetime.now().strftime("%Y-%m-%d")
        report_path = Path("briefs") / today / f"{session_tag}板块轮动选股.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        print(f"\n{session_tag}板块轮动选股完成:")
        print(f"  报告: {report_path}")
        print(f"  选中: {results['total_count']} 只 ({results['stock_count']} 股票 + {results['etf_count']} ETF)")
        print(f"  覆盖: {results['sectors_covered']} 个板块")

        # 板块分布摘要
        for sector, count in list(results.get("sector_breakdown", {}).items())[:10]:
            print(f"    {sector}: {count} 只")

    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        teardown_protection()


if __name__ == "__main__":
    main()
