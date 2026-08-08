"""
迷你回测脚本 — 基于 westock 内置技能
跑 8/4 ~ 8/6 三天的评分 Top10，验证 T+1+3 收益
"""
import subprocess, json, sys
from datetime import datetime, timedelta
from pathlib import Path

NODE = r"C:\Users\63516\.workbuddy\binaries\node\versions\22.22.2\node.exe"
TOOL_JS = r"D:\soft\dev\WorkBuddy\resources\app.asar.unpacked\resources\builtin-skills\westock-tool\scripts\index.js"
DATA_JS = r"D:\soft\dev\WorkBuddy\resources\app.asar.unpacked\resources\builtin-skills\westock-data\scripts\index.js"
NODE_PATH = r"C:\Users\63516\.workbuddy\binaries\node\workspace\node_modules"

def run_node(script, args):
    cmd = [NODE, script] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           env={**dict(Path('.').iterdir()), 'NODE_PATH': NODE_PATH} if False else None)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None
    try:
        return json.loads(result.stdout)
    except:
        print(f"  PARSE ERROR: {result.stdout[:200]}")
        return None

def get_ranking(date_str, limit=10):
    print(f"\n{'='*60}")
    print(f"获取 {date_str} 评分 Top{limit}...")
    data = run_node(TOOL_JS, ['ranking', 'CompScore', '--date', date_str, f'--limit', str(limit), '--raw'])
    if not data:
        return []
    stocks = []
    for item in data:
        code = item['代码']  # 保持原格式 sh/sz
        stocks.append({
            'code': code,
            'name': item['名称'],
            'score': item['综合评分'],
            'fund_score': item.get('基本面评分', 0),
            'tech_score': item.get('技术评分', 0),
        })
        print(f"  {len(stocks)}. {code} {item['名称']} 评分:{item['综合评分']}")
    return stocks

def get_kline(code, start, end):
    """获取K线数据"""
    data = run_node(DATA_JS, ['kline', code, '--start', start, '--end', end, '--period', 'day', '--raw'])
    if not data or not isinstance(data, list) or len(data) < 2:
        return None
    return data

def verify_picks(picks, pick_date, periods=[1, 3]):
    """验证选股在 T+N 日的收益"""
    results = []
    end_date = (datetime.strptime(pick_date, '%Y-%m-%d') + timedelta(days=10)).strftime('%Y-%m-%d')
    
    for i, pick in enumerate(picks):
        code = pick['code']
        kline = get_kline(code, pick_date, end_date)
        if not kline or len(kline) < 2:
            results.append({**pick, 'status': 'no_data'})
            continue
        
        # K线按日期降序排列（最新在前），翻转成升序用于计算
        kline.reverse()
        
        # 找到 pick_date 当天或之后的第一个收盘价作为入场价
        entry = None
        entry_idx = -1
        for i, k in enumerate(kline):
            if k['date'] >= pick_date:
                entry = k
                entry_idx = i
                break
        if entry is None:
            results.append({**pick, 'status': 'no_entry'})
            continue
        
        entry_price = entry.get('last', entry.get('close', 0))
        
        perf = {**pick, 'entry_price': entry_price, 'entry_date': entry.get('date', pick_date)}
        for p in periods:
            if p < len(kline):
                exit_price = kline[p].get('last', kline[p].get('close', 0))
                if exit_price and entry_price:
                    ret = (exit_price / entry_price - 1) * 100
                    perf[f'T+{p}_price'] = round(exit_price, 2)
                    perf[f'T+{p}_return'] = round(ret, 2)
                else:
                    perf[f'T+{p}_return'] = None
            else:
                perf[f'T+{p}_return'] = None
        
        results.append(perf)
        if (i+1) % 3 == 0:
            print(f"  已验证 {i+1}/{len(picks)}")
    
    return results

# ===== 主流程 =====
dates = ['2026-08-04', '2026-08-05', '2026-08-06']
all_results = {}
all_returns = []

for d in dates:
    picks = get_ranking(d, 10)
    if not picks:
        print(f"  {d}: 无数据，跳过")
        continue
    
    print(f"\n验证 {d} 的 {len(picks)} 只推荐...")
    verified = verify_picks(picks, d)
    all_results[d] = verified
    
    for v in verified:
        for key in ['T+1_return', 'T+3_return']:
            if key in v and v[key] is not None:
                all_returns.append({'date': d, 'code': v['code'], 'name': v['name'],
                                   'score': v['score'], 'key': key, 'return': v[key]})

# ===== 统计 =====
print(f"\n{'='*60}")
print(f"回测总结")
print(f"{'='*60}")

t1_returns = [r['return'] for r in all_returns if r['key'] == 'T+1_return']
t3_returns = [r['return'] for r in all_returns if r['key'] == 'T+3_return']

if t1_returns:
    pos = [r for r in t1_returns if r > 0]
    print(f"\nT+1 收益统计:")
    print(f"  样本数: {len(t1_returns)}")
    print(f"  平均: {sum(t1_returns)/len(t1_returns):+.2f}%")
    print(f"  胜率: {len(pos)/len(t1_returns)*100:.1f}% ({len(pos)}/{len(t1_returns)})")
    print(f"  最大: {max(t1_returns):+.2f}%  最小: {min(t1_returns):+.2f}%")

if t3_returns:
    pos = [r for r in t3_returns if r > 0]
    print(f"\nT+3 收益统计:")
    print(f"  样本数: {len(t3_returns)}")
    print(f"  平均: {sum(t3_returns)/len(t3_returns):+.2f}%")
    print(f"  胜率: {len(pos)/len(t3_returns)*100:.1f}% ({len(pos)}/{len(t3_returns)})")
    print(f"  最大: {max(t3_returns):+.2f}%  最小: {min(t3_returns):+.2f}%")

# 存结果到  SQLite
print(f"\n保存结果到 SQLite...")
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from database import get_db
db = get_db(str(Path(__file__).parent / 'data_cache' / 'a-stock-engine.db'))

for d, picks in all_results.items():
    verifications = []
    for p in picks:
        v = {
            'code': p['code'],
            'name': p['name'],
            't0_close': p.get('entry_price'),
            'return_pct': p.get('T+1_return'),
            'status': 'success' if p.get('T+1_return') is not None else p.get('status', 'no_data'),
        }
        if p.get('T+1_price'):
            v['t2_close'] = p['T+1_price']
        verifications.append(v)
    db.save_t2_verification(d, verifications)

print("Done! 数据已存入 SQLite")
