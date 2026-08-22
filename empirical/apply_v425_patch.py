# -*- coding: utf-8 -*-
"""v4.25 补丁(REJECTED — 回测证伪, 勿应用): 砍 roe 负 alpha 权重 + 估值/质量硬闸门。

改动基于 #23 实证(ROE Rank-IC=-0.078/t=-2.94 显著负)与圆桌共识。
每处替换带断言(必须恰好命中1次), 未命中即报错退出, 避免静默部分替换。

=== 回测证伪结论(2026-08-21, 同一解释器 D:\env\python3.12 pandas3.0.5, 同 harness/数据) ===
  v4.24 未打补丁:   总 +0.59% / 夏普 0.07 / 回撤 26.9% / 393 笔
  v4.25 打此补丁:   总 -5.51% / 夏普 -0.01 / 回撤 19.1% / 382 笔   <- 回退 6.1pp, 夏普转负
结论: 截面 IC 为负 != 组合实盘改善。补丁使策略更防御(熊市少亏)但牺牲过多牛市收益,
      净效应为负, 不满足"无回归"护栏。未打 tag / 未 push。保留脚本仅供未来 ablation 复现。
"""
import sys, io

BASE = r"D:\workspace\github\a-stock-engine"

PATCHES = {
    r"src\lvrev_scorer.py": [
        # 1) 内核权重: 把 q(质量, 含负IC的roe) 0.12->0.05, 腾给 vol/rev(真alpha)
        ("W_DEFAULT = dict(vol=0.45, rev=0.35, value=0.0, q=0.12, g=0.08)",
         "W_DEFAULT = dict(vol=0.52, rev=0.43, value=0.0, q=0.05, g=0.08)  # #25: q(含负IC的roe) 0.12->0.05, 腾给vol/rev真alpha"),
        ("W_VALUE = dict(vol=0.38, rev=0.27, value=0.18, q=0.10, g=0.07)",
         "W_VALUE = dict(vol=0.43, rev=0.32, value=0.18, q=0.05, g=0.07)  # #25: 同步降q"),
        # 2) 质量分: 去掉 roe 正向排序(负IC), 仅保留低杠杆(debt_ratio)稳定器
        ('    # 3) 质量: ROE 越高越好, 负债率越低越好\n'
         '    s_q = pd.Series(0.5, index=d.index)\n'
         '    if "roe" in d.columns:\n'
         '        s_q = s_q + d["roe"].clip(-20, 40).fillna(0) / 40.0',
         '    # 3) 质量稳定器: 仅保留低杠杆(debt_ratio), 不再用 ROE 正向排序\n'
         '    #    (ROE Rank-IC=-0.078/t=-2.94 显著负, 见#23实证; 高ROE是负alpha, 改作门槛)\n'
         '    s_q = pd.Series(0.5, index=d.index)'),
    ],
    r"src\multifactor.py": [
        # 3) _score_l4_lvrev: 调 score_lvrev 前加估值/质量硬闸门(轻量绝对门槛)
        ('        df = pd.DataFrame(factor_list)\n'
         '        if len(df) == 0:\n'
         '            return df\n'
         '\n'
         '        # rev_chg 兜底(若批量特征缺失): 用 momentum_20d 等价替代',
         '        df = pd.DataFrame(factor_list)\n'
         '        if len(df) == 0:\n'
         '            return df\n'
         '\n'
         '        # 估值/质量硬闸门(轻量绝对门槛, #25 圆桌共识落地):\n'
         '        #   roe 不再作排序因子(截面IC=-0.078/t=-2.94显著负, 见#23实证), 仅作质量排雷门槛;\n'
         '        #   pb/pe 极端高估直接剔除(贵的不进榜)。\n'
         '        fg = self.config.get("factor_l4", {})\n'
         '        vg = fg.get("valuation_gate", {})\n'
         '        qg = fg.get("quality_gate", {})\n'
         '        max_pb = vg.get("max_pb", 0)\n'
         '        max_pe = vg.get("max_pe", 0)\n'
         '        roe_min = qg.get("roe_min", None)\n'
         '        debt_max = qg.get("debt_max", None)\n'
         '        if max_pb and "pb" in df.columns:\n'
         '            df = df[~(df["pb"] > max_pb)]\n'
         '        if max_pe and "pe" in df.columns:\n'
         '            df = df[~((df["pe"] > 0) & (df["pe"] > max_pe))]\n'
         '        if roe_min is not None and "roe" in df.columns:\n'
         '            df = df[df["roe"] >= roe_min]\n'
         '        if debt_max is not None and "debt_ratio" in df.columns:\n'
         '            df = df[df["debt_ratio"] <= debt_max]\n'
         '\n'
         '        # rev_chg 兜底(若批量特征缺失): 用 momentum_20d 等价替代'),
    ],
    r"config.yaml": [
        # 4) factor_l4.factors roe 权重 0.15->0 (旧内核死路径清理, 防误开仍追负alpha)
        ('    - name: "roe"\n'
         '      weight: 0.15\n'
         '      direction: "descending"     # ROE越高越好',
         '    - name: "roe"\n'
         '      weight: 0.0\n'
         '      direction: "descending"     # ROE越高越好 (权重置0: 仅旧内核路径用, IC显著负不追; 真alpha见lvrev内核)'),
        # 5) factor_l4 加估值/质量硬闸门配置
        ('    apply_entry_gate: false  # 默认仅标记 entry_ok(保证②A有候选); true=硬过滤只留到买点',
         '    apply_entry_gate: false  # 默认仅标记 entry_ok(保证②A有候选); true=硬过滤只留到买点\n'
         '    # 估值/质量硬闸门(轻量绝对门槛, #25 落地): roe改门槛(不再排序), 极端高估剔除\n'
         '    #   内核权重(vol/rev/q/g)在 src/lvrev_scorer.W_DEFAULT 硬编码, 改权重请调那。\n'
         '    valuation_gate:\n'
         '      max_pb: 20            # pb>20 剔除(极端高估/概念炒作)\n'
         '      max_pe: 100           # pe>100(且pe>0) 剔除(无盈利高估值)\n'
         '    quality_gate:\n'
         '      roe_min: 0            # roe<0(亏损) 剔除 — roe不作排序因子(IC显著负), 仅排雷\n'
         '      debt_max: 85          # 资产负债率>85% 剔除(高杠杆风险)'),
    ],
}

def main():
    ok = True
    for rel, repls in PATCHES.items():
        p = BASE + "\\" + rel
        with io.open(p, "r", encoding="utf-8") as f:
            src = f.read()
        for i, (old, new) in enumerate(repls):
            cnt = src.count(old)
            if cnt != 1:
                print("[FAIL] %s 替换#%d 命中 %d 次(期望1):\n---OLD---\n%r" % (rel, i, cnt, old))
                ok = False
                continue
            src = src.replace(old, new, 1)
            print("[OK]   %s 替换#%d 已应用" % (rel, i))
        if ok:
            with io.open(p, "w", encoding="utf-8") as f:
                f.write(src)
    if not ok:
        print("\n有替换未精确命中, 已中止写入。请检查 old 字符串。")
        sys.exit(1)
    print("\n全部补丁精确命中并已写入。")

if __name__ == "__main__":
    main()
