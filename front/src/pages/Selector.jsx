import { useState } from 'react'
import { Play, RefreshCw, Target, Shield } from 'lucide-react'

const API = '/api'

export default function Selector() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [topN, setTopN] = useState(20)
  const [maxPer, setMaxPer] = useState(5)

  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/select/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ top_n: topN, max_per_sector: maxPer })
      })
      const data = await r.json()
      setResult(data)
    } catch (e) {
      alert('请求失败: ' + e.message)
    }
    setLoading(false)
  }

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24, color: '#1e293b' }}>选股</h1>

      {/* 控制面板 */}
      <div style={styles.controls}>
        <label style={styles.label}>
          输出数量
          <input type="number" value={topN} onChange={e => setTopN(+e.target.value)}
            style={styles.input} min={5} max={100} />
        </label>
        <label style={styles.label}>
          每板块上限
          <input type="number" value={maxPer} onChange={e => setMaxPer(+e.target.value)}
            style={styles.input} min={1} max={20} />
        </label>
        <button onClick={run} disabled={loading} style={styles.btn}>
          {loading ? <RefreshCw size={16} className="spin" /> : <Play size={16} />}
          {loading ? '运行中...' : '执行选股'}
        </button>
      </div>

      {/* 市场状态 */}
      {result && (
        <div style={{ ...styles.badge, background: result.regime?.regime === 'bull' ? '#fef2f2' : '#f0fdf4' }}>
          <span style={{ fontSize: 20 }}>
            {result.regime?.regime === 'bull' ? '🐂' : '🐻'}
          </span>
          <span style={{ color: result.regime?.regime === 'bull' ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
            {result.regime?.regime === 'bull' ? '牛市' : '熊市'} 
          </span>
          <span style={{ color: '#64748b' }}>
            | 均价 {result.regime?.avg} | MA60 {result.regime?.ma60} | {result.date}
          </span>
          <span style={{ color: '#64748b', marginLeft: 16 }}>
            {result.stats?.universe?.toLocaleString()} → {result.stats?.candidates?.toLocaleString()} → {result.stats?.selected} 只
          </span>
        </div>
      )}

      {/* 选股结果 */}
      {result?.picks?.length > 0 && (
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th>#</th><th>代码</th><th>名称</th><th>板块</th><th>评分</th>
                <th>现价</th><th><Target size={14} /> 目标</th><th><Shield size={14} /> 止损</th>
              </tr>
            </thead>
            <tbody>
              {result.picks.map((p, i) => (
                <tr key={p.code}>
                  <td style={{ color: '#64748b' }}>{i + 1}</td>
                  <td style={styles.code}>{p.code}</td>
                  <td>{p.name}</td>
                  <td style={styles.tag}>{p.sector}</td>
                  <td style={{ color: p.score_norm > 70 ? '#ef4444' : '#64748b', fontWeight: 600 }}>
                    {p.score_norm}
                  </td>
                  <td>{p.close?.toFixed(2)}</td>
                  <td style={{ color: '#ef4444' }}>{p.target?.toFixed(2)}</td>
                  <td style={{ color: '#22c55e' }}>{p.stop?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result?.picks?.length === 0 && (
        <div style={styles.empty}>暂无符合条件的标的</div>
      )}
    </div>
  )
}

const styles = {
  controls: { display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 20, flexWrap: 'wrap' },
  label: { fontSize: 14, color: '#475569', display: 'flex', flexDirection: 'column', gap: 4 },
  input: { width: 80, padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: 8, fontSize: 14 },
  btn: { display: 'flex', alignItems: 'center', gap: 8, padding: '10px 24px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 15, fontWeight: 600 },
  badge: { display: 'flex', alignItems: 'center', gap: 10, padding: '12px 20px', borderRadius: 10, marginBottom: 20, fontSize: 14, flexWrap: 'wrap' },
  tableWrap: { background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', overflow: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse' },
  code: { fontFamily: 'monospace', fontWeight: 600 },
  tag: { background: '#f1f5f9', padding: '2px 8px', borderRadius: 6, fontSize: 13 },
  empty: { textAlign: 'center', padding: 60, color: '#94a3b8', fontSize: 16 },
}
