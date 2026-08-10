r"""
verify_fixes.py — 验证所有修复是否生效的诊断脚本

用法:
    cd D:/workspace/github/a-stock-engine
    python verify_fixes.py

检查项目:
  1. config 因子名称是否正确（pe/pb 而非 pe_rank/pb_rank）
  2. market.db 中 daily_snapshot 是否可用（基本面数据源）
  3. daily_price 表是否有数据（批量动量计算）
  4. 因子评分链路是否通畅（模拟单只股票评分）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("verify")

def check_config():
    """1. 检查配置文件因子名称"""
    print("\n" + "=" * 60)
    print("1. 检查 config.yaml 因子名称")
    print("=" * 60)
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    factors = config["factor_l4"]["factors"]
    names = [f["name"] for f in factors]
    total_weight = sum(f["weight"] for f in factors)
    
    print(f"  因子数量: {len(factors)}")
    print(f"  总权重: {total_weight:.2f}")
    print(f"  因子列表: {names}")
    
    # 关键检查
    has_pe_rank = "pe_rank" in names
    has_pb_rank = "pb_rank" in names
    has_pe = "pe" in names
    has_pb = "pb" in names
    
    if has_pe_rank or has_pb_rank:
        print("  ❌ 仍然使用旧名称 pe_rank/pb_rank！")
    elif has_pe and has_pb:
        print("  ✅ 因子名称已修复为 pe/pb")
    else:
        print("  ⚠️  因子名称异常，请检查")
    
    # 市值单位
    min_cap = config["filter_l2"]["min_market_cap"]
    if min_cap > 1000000:
        print(f"  ✅ 市值过滤已改为元: {min_cap:,.0f} 元 ({min_cap/1e8:.0f} 亿)")
    else:
        print(f"  ⚠️  市值过滤值偏小: {min_cap}，可能仍在使用亿元单位")

def check_market_db():
    """2. 检查 market.db 数据"""
    print("\n" + "=" * 60)
    print("2. 检查 market.db 数据")
    print("=" * 60)
    
    from database import get_market_db
    mdb = get_market_db()
    
    # daily_snapshot
    row = mdb.conn.execute("""
        SELECT cache_key, rows_count, created_at 
        FROM market_data_cache 
        WHERE data_type='daily_snapshot' 
        ORDER BY cache_key DESC LIMIT 1
    """).fetchone()
    
    if row:
        print(f"  ✅ 本地快照: {row['cache_key']}")
        print(f"     {row['rows_count']} 行, 创建于 {row['created_at']}")
    else:
        print("  ⚠️  无 daily_snapshot 数据（需运行 import_local_data.py 导入）")
    
    # daily_price
    count = mdb.conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
    date_range = mdb.conn.execute("""
        SELECT MIN(date) as min_d, MAX(date) as max_d, COUNT(DISTINCT code) as stocks
        FROM daily_price
    """).fetchone()
    
    if count > 0:
        print(f"  ✅ daily_price 表: {count:,} 行, {date_range['stocks']} 只股票")
        print(f"     日期范围: {date_range['min_d']} ~ {date_range['max_d']}")
        
        # 检查是否有足够的交易日数据用于动量计算
        sample = mdb.conn.execute("""
            SELECT code, COUNT(*) as cnt FROM daily_price 
            GROUP BY code ORDER BY cnt DESC LIMIT 5
        """).fetchall()
        print(f"     Top 5 数据最全的股票:")
        for s in sample:
            print(f"       {s['code']}: {s['cnt']} 个交易日")
    else:
        print("  ⚠️  daily_price 表为空（需运行 import_local_data.py 导入 parquet 数据）")
    
    mdb.close()

def check_factor_pipeline():
    """3. 检查因子评分链路"""
    print("\n" + "=" * 60)
    print("3. 检查因子评分链路（模拟单只股票）")
    print("=" * 60)
    
    try:
        from multifactor import MultiFactorEngine
        import pandas as pd
        
        # 模拟数据
        mock_data = {
            "code": "000001",
            "name": "平安银行",
            "pe": 15.5,
            "pb": 1.2,
            "roe": 12.3,
            "gross_margin": 35.0,
            "debt_ratio": 45.0,
            "revenue_growth": 8.5,
            "profit_growth": 10.2,
            "market_cap": 3.5e11,
            "momentum_20d": 5.2,
            "momentum_60d": 12.8,
            "sector": "银行",
        }
        
        # 验证 _extract_fundamentals
        fund = MultiFactorEngine._extract_fundamentals(None, pd.Series(mock_data))
        print(f"  基本面提取结果: pe={fund.get('pe')}, pb={fund.get('pb')}, "
              f"roe={fund.get('roe')}, market_cap={fund.get('market_cap')}")
        
        # 验证因子名称匹配
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        factors_cfg = config["factor_l4"]["factors"]
        fund_keys = set(fund.keys())
        factor_names = set(f["name"] for f in factors_cfg)
        
        matched = factor_names & fund_keys
        missing = factor_names - fund_keys
        
        print(f"  因子匹配: {len(matched)}/{len(factor_names)} 命中")
        print(f"    命中: {sorted(matched)}")
        if missing:
            print(f"    缺失: {sorted(missing)}")
        
        if "pe_rank" not in factor_names and "pb_rank" not in factor_names:
            print("  ✅ pe/pb 因子名称正确（非 pe_rank/pb_rank）")
        
    except Exception as e:
        print(f"  ❌ 链路检查失败: {e}")

def main():
    print("A股多因子选股系统 — 修复验证")
    print("=" * 60)
    
    check_config()
    check_market_db()
    check_factor_pipeline()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print("\n运行选股引擎: cd D:\\workspace\\github\\a-stock-engine && python -m src.daily_brief")

if __name__ == "__main__":
    main()
