import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, Database, Calendar } from 'lucide-react'

const API = '/api'

export default function Dashboard() {
  const [regime, setRegime] = useState(null)
  const [summary, setSummary] = useState(null)

  useEffect(() => {
    fetch(`${API}/data/market-regime`).then(r => r.json()).then(setRegime)
    fetch(`${API}/data/summary`).then(r => r.json()).then(setSummary)
  }, [])

  const Card = ({ icon: Icon, label, value, color }) => (
    <div style={styles.card}>
      <Icon size={24} style={{ color: color || '#64748b' }} />
      <div>
        <div style={styles.cardLabel}>{label}</div>
        <div style={styles.cardValue}>{value ?? '加载中...'}</div>
      </div>
    </div>
  )

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24, color: '#1e293b' }}>系统概览</h1>

      <div style={styles.grid}>
        <Card
          icon={regime?.regime === 'bull' ? TrendingUp : TrendingDown}
          label="市场环境"
          value={regime ? `${regime.regime === 'bull' ? '🐂 牛市' : '🐻 熊市'} (均价 ${regime.market_avg})` : '加载中...'}
          color={regime?.regime === 'bull' ? '#ef4444' : '#22c55e'}
        />
        <Card
          icon={Database}
          label="数据量"
          value={summary ? `${(summary.daily_price_rows / 1e6).toFixed(1)}M 条` : '加载中...'}
          color="#3b82f6"
        />
        <Card
          icon={Calendar}
          label="数据区间"
          value={summary?.daily_price_range || '加载中...'}
          color="#8b5cf6"
        />
        <Card
          icon={Database}
          label="基本面天数"
          value={summary ? `${summary.fundamental_days} 天` : '加载中...'}
          color="#f59e0b"
        />
      </div>

      <div style={{ marginTop: 32, padding: 24, background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0' }}>
        <h2 style={{ fontSize: 18, marginBottom: 16, color: '#1e293b' }}>快速操作</h2>
        <p style={{ color: '#64748b', lineHeight: 1.8 }}>
          点击左侧 <strong>选股</strong> 执行当日多因子选股。<br />
          点击左侧 <strong>回测</strong> 运行历史资金模拟。
        </p>
      </div>
    </div>
  )
}

const styles = {
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 },
  card: { background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', gap: 16 },
  cardLabel: { fontSize: 13, color: '#64748b', marginBottom: 4 },
  cardValue: { fontSize: 18, fontWeight: 600, color: '#1e293b' },
}
