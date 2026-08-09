import { useState, useEffect, useCallback } from 'react'
import { Play, RefreshCw, TrendingUp, TrendingDown, Calendar } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

const API = '/api'

export default function Backtest() {
  const [loading, setLoading] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [result, setResult] = useState(null)
  const [capital, setCapital] = useState(50000)
  const [maxPicks, setMaxPicks] = useState(20)

  const start = async () => {
    setLoading(true)
    setResult(null)
    try {
      const r = await fetch(`${API}/backtest/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ capital, max_picks: maxPicks })
      })
      const { job_id } = await r.json()
      setJobId(job_id)
    } catch (e) {
      alert('启动失败: ' + e.message)
      setLoading(false)
    }
  }

  const poll = useCallback(async () => {
    if (!jobId) return
    const r = await fetch(`${API}/backtest/status/${jobId}`)
    const data = await r.json()
    if (data.status === 'done') {
      setResult(data.result)
      setLoading(false)
      setJobId(null)
    } else if (data.status === 'error') {
      alert('回测失败: ' + data.error)
      setLoading(false)
      setJobId(null)
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId || !loading) return
    const interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [jobId, loading, poll])

  const Card = ({ label, value, color, prefix = '', suffix = '' }) => (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={{ ...styles.cardValue, color: color || '#1e293b' }}>
        {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
      </div>
    </div>
  )

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24, color: '#1e293b' }}>回测</h1>

      <div style={styles.controls}>
        <label style={styles.label}>
          初始资金
          <input type="number" value={capital} onChange={e => setCapital(+e.target.value)}
            style={styles.input} min={10000} step={10000} />
        </label>
        <label style={styles.label}>
          每日持仓
          <input type="number" value={maxPicks} onChange={e => setMaxPicks(+e.target.value)}
            style={styles.input} min={5} max={50} />
        </label>
        <button onClick={start} disabled={loading} style={styles.btn}>
          {loading ? <RefreshCw size={16} className="spin" /> : <Play size={16} />}
          {loading ? '回测中... (约2分钟)' : '开始回测'}
        </button>
      </div>

      {result && (
        <>
          {/* 核心指标 */}
          <div style={styles.grid}>
            <Card label="初始资金" value={result.initial} color="#64748b" prefix="¥" />
            <Card label="最终资金" value={result.final} color={result.return_pct > 0 ? '#ef4444' : '#22c55e'} prefix="¥" />
            <Card label="总收益" value={result.return_pct} color={result.return_pct > 0 ? '#ef4444' : '#22c55e'} suffix="%" />
            <Card label="年化收益" value={result.cagr_pct} color={result.cagr_pct > 0 ? '#ef4444' : '#22c55e'} suffix="%" />
            <Card label="最大回撤" value={result.max_drawdown_pct} color="#22c55e" suffix="%" prefix="-" />
            <Card label="夏普比率" value={result.sharpe} color="#3b82f6" />
          </div>

          {/* 资产曲线 */}
          {result.portfolio?.length > 0 && (
            <div style={styles.section}>
              <h3 style={{ marginBottom: 16, color: '#1e293b' }}>资产曲线</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={result.portfolio.map(p => ({ date: p[0]?.slice(0, 10), value: p[1] }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={v => '¥' + (v / 10000).toFixed(0) + '万'} />
                  <Tooltip formatter={v => ['¥' + v.toLocaleString(), '资产']} />
                  <Area type="monotone" dataKey="value" stroke="#ef4444" fill="rgba(239,68,68,0.1)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* 逐年收益 */}
          {result.year_returns && (
            <div style={styles.section}>
              <h3 style={{ marginBottom: 16, color: '#1e293b' }}>逐年收益</h3>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {Object.entries(result.year_returns).map(([y, r]) => (
                  <div key={y} style={{
                    padding: '8px 16px', borderRadius: 8, background: r > 0 ? '#fef2f2' : '#f0fdf4',
                    border: `1px solid ${r > 0 ? '#fecaca' : '#bbf7d0'}`,
                  }}>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{y}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: r > 0 ? '#ef4444' : '#22c55e' }}>
                      {r > 0 ? '+' : ''}{r}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 交易统计 */}
          <div style={styles.section}>
            <h3 style={{ marginBottom: 16, color: '#1e293b' }}>交易统计</h3>
            <div style={styles.grid}>
              <Card label="总交易" value={result.trades} color="#64748b" suffix=" 笔" />
              <Card label="平均持有" value={result.avg_held} color="#64748b" suffix=" 天" />
              <Card label="平均收益" value={result.avg_return} color={result.avg_return > 0 ? '#ef4444' : '#22c55e'} suffix="%" />
            </div>
            {result.sell_reasons && (
              <div style={{ marginTop: 16, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                {Object.entries(result.sell_reasons).map(([k, v]) => (
                  <div key={k} style={{ background: '#f8fafc', padding: '8px 14px', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                    <span style={{ color: '#64748b', fontSize: 13 }}>{k}</span>
                    <span style={{ marginLeft: 8, fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

const styles = {
  controls: { display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 24, flexWrap: 'wrap' },
  label: { fontSize: 14, color: '#475569', display: 'flex', flexDirection: 'column', gap: 4 },
  input: { width: 100, padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: 8, fontSize: 14 },
  btn: { display: 'flex', alignItems: 'center', gap: 8, padding: '10px 24px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 15, fontWeight: 600 },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 24 },
  card: { background: '#fff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' },
  cardLabel: { fontSize: 12, color: '#94a3b8', marginBottom: 6 },
  cardValue: { fontSize: 22, fontWeight: 700, fontFamily: 'monospace' },
  section: { background: '#fff', borderRadius: 12, padding: 24, border: '1px solid #e2e8f0', marginBottom: 20 },
}
