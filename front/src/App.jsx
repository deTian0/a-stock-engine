import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import Selector from './pages/Selector'
import Backtest from './pages/Backtest'
import { BarChart3, ListFilter, Settings } from 'lucide-react'

const TABS = [
  { id: 'dashboard', label: '概览', icon: BarChart3 },
  { id: 'selector', label: '选股', icon: ListFilter },
  { id: 'backtest', label: '回测', icon: Settings },
]

export default function App() {
  const [tab, setTab] = useState('dashboard')

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <nav style={styles.sidebar}>
        <div style={styles.logo}>A股选股</div>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              ...styles.tabBtn,
              ...(tab === t.id ? styles.tabActive : {})
            }}
          >
            <t.icon size={18} />
            <span>{t.label}</span>
          </button>
        ))}
      </nav>

      {/* Main */}
      <main style={styles.main}>
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'selector' && <Selector />}
        {tab === 'backtest' && <Backtest />}
      </main>
    </div>
  )
}

const styles = {
  container: { display: 'flex', minHeight: '100vh', background: '#f1f5f9', fontFamily: '-apple-system, sans-serif' },
  sidebar: { width: 200, background: '#1e293b', color: '#e2e8f0', padding: '24px 12px', display: 'flex', flexDirection: 'column', gap: 8 },
  logo: { fontSize: 20, fontWeight: 700, color: '#38bdf8', marginBottom: 24, padding: '0 12px' },
  tabBtn: { display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none', color: '#94a3b8', padding: '10px 12px', borderRadius: 8, cursor: 'pointer', fontSize: 15, textAlign: 'left', width: '100%' },
  tabActive: { background: '#334155', color: '#38bdf8' },
  main: { flex: 1, padding: 32, overflow: 'auto' },
}
