/**
 * Elite Quant System - Real-time Trading Dashboard
 * Modern React frontend with WebSocket streaming and D3/Recharts visualization
 */

import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, 
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Activity, AlertTriangle, 
  Zap, Shield, Brain, BarChart3, ArrowUpRight, ArrowDownRight
} from 'lucide-react';
import { create } from 'zustand';

// =============================================================================
// TYPES
// =============================================================================

interface Signal {
  symbol: string;
  direction: number;
  confidence: number;
  uncertainty: number;
  regimeId: number;
  timestamp: number;
}

interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

interface RiskMetrics {
  sharpe: number;
  maxDrawdown: number;
  dailyPnl: number;
  positionUtilization: number;
  var95: number;
  regime: 'LOW_VOL' | 'NORMAL' | 'HIGH_VOL' | 'CRISIS';
}

interface PnLDataPoint {
  timestamp: number;
  value: number;
  cumulative: number;
}

// =============================================================================
// ZUSTAND STORE
// =============================================================================

interface TradingStore {
  signals: Signal[];
  positions: Position[];
  riskMetrics: RiskMetrics;
  pnlHistory: PnLDataPoint[];
  connected: boolean;
  addSignal: (signal: Signal) => void;
  setPositions: (positions: Position[]) => void;
  setRiskMetrics: (metrics: RiskMetrics) => void;
  addPnLPoint: (point: PnLDataPoint) => void;
  setConnected: (connected: boolean) => void;
}

const useTradingStore = create<TradingStore>((set) => ({
  signals: [],
  positions: [],
  riskMetrics: {
    sharpe: 0,
    maxDrawdown: 0,
    dailyPnl: 0,
    positionUtilization: 0,
    var95: 0,
    regime: 'NORMAL',
  },
  pnlHistory: [],
  connected: false,
  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 100),
    })),
  setPositions: (positions) => set({ positions }),
  setRiskMetrics: (riskMetrics) => set({ riskMetrics }),
  addPnLPoint: (point) =>
    set((state) => ({
      pnlHistory: [...state.pnlHistory, point].slice(-500),
    })),
  setConnected: (connected) => set({ connected }),
}));

// =============================================================================
// WEBSOCKET HOOK
// =============================================================================

const useWebSocket = (url: string) => {
  const { addSignal, setPositions, setRiskMetrics, addPnLPoint, setConnected } =
    useTradingStore();

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'signal':
          addSignal(data.payload);
          break;
        case 'positions':
          setPositions(data.payload);
          break;
        case 'risk':
          setRiskMetrics(data.payload);
          break;
        case 'pnl':
          addPnLPoint(data.payload);
          break;
      }
    };

    return () => ws.close();
  }, [url]);
};

// =============================================================================
// COMPONENTS
// =============================================================================

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color?: 'green' | 'red' | 'yellow' | 'blue';
}> = ({ title, value, change, icon, color = 'blue' }) => {
  const colorClasses = {
    green: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30',
    red: 'from-rose-500/20 to-rose-600/5 border-rose-500/30',
    yellow: 'from-amber-500/20 to-amber-600/5 border-amber-500/30',
    blue: 'from-cyan-500/20 to-cyan-600/5 border-cyan-500/30',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-gradient-to-br ${colorClasses[color]} border rounded-xl p-5 backdrop-blur-sm`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-zinc-400 text-sm font-medium">{title}</span>
        <div className="text-zinc-500">{icon}</div>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-white font-mono">{value}</span>
        {change !== undefined && (
          <span
            className={`text-sm flex items-center ${
              change >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {change >= 0 ? (
              <ArrowUpRight size={14} />
            ) : (
              <ArrowDownRight size={14} />
            )}
            {Math.abs(change).toFixed(2)}%
          </span>
        )}
      </div>
    </motion.div>
  );
};

const RegimeBadge: React.FC<{ regime: string }> = ({ regime }) => {
  const colors: Record<string, string> = {
    LOW_VOL: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    NORMAL: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
    HIGH_VOL: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    CRISIS: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  };

  return (
    <motion.span
      initial={{ scale: 0.8 }}
      animate={{ scale: 1 }}
      className={`px-3 py-1 rounded-full text-xs font-medium border ${
        colors[regime] || colors.NORMAL
      }`}
    >
      {regime.replace('_', ' ')}
    </motion.span>
  );
};

const SignalFeed: React.FC = () => {
  const { signals } = useTradingStore();

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5 h-80 overflow-hidden">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="text-violet-400" size={18} />
        <h3 className="text-white font-semibold">AI Signals</h3>
      </div>
      <div className="space-y-2 overflow-y-auto h-60 scrollbar-thin">
        <AnimatePresence mode="popLayout">
          {signals.slice(0, 10).map((signal, idx) => (
            <motion.div
              key={signal.timestamp}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: idx * 0.05 }}
              className={`flex items-center justify-between p-3 rounded-lg border ${
                signal.direction > 0
                  ? 'bg-emerald-500/10 border-emerald-500/20'
                  : signal.direction < 0
                  ? 'bg-rose-500/10 border-rose-500/20'
                  : 'bg-zinc-800/50 border-zinc-700'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="text-white font-mono font-medium">
                  {signal.symbol}
                </span>
                {signal.direction > 0 ? (
                  <TrendingUp className="text-emerald-400" size={16} />
                ) : signal.direction < 0 ? (
                  <TrendingDown className="text-rose-400" size={16} />
                ) : (
                  <Activity className="text-zinc-400" size={16} />
                )}
              </div>
              <div className="flex items-center gap-3">
                <span className="text-zinc-400 text-sm">
                  {(signal.confidence * 100).toFixed(0)}%
                </span>
                <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      signal.direction > 0
                        ? 'bg-emerald-500'
                        : signal.direction < 0
                        ? 'bg-rose-500'
                        : 'bg-zinc-600'
                    }`}
                    style={{ width: `${signal.confidence * 100}%` }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};

const PositionsTable: React.FC = () => {
  const { positions } = useTradingStore();

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="text-cyan-400" size={18} />
        <h3 className="text-white font-semibold">Active Positions</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-zinc-500 text-xs uppercase">
              <th className="text-left pb-3">Symbol</th>
              <th className="text-right pb-3">Qty</th>
              <th className="text-right pb-3">Entry</th>
              <th className="text-right pb-3">Current</th>
              <th className="text-right pb-3">P&L</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {positions.map((pos) => (
              <motion.tr
                key={pos.symbol}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-t border-zinc-800"
              >
                <td className="py-3 text-white font-mono">{pos.symbol}</td>
                <td
                  className={`py-3 text-right ${
                    pos.quantity > 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {pos.quantity > 0 ? '+' : ''}
                  {pos.quantity}
                </td>
                <td className="py-3 text-right text-zinc-400 font-mono">
                  ${pos.entryPrice.toFixed(2)}
                </td>
                <td className="py-3 text-right text-white font-mono">
                  ${pos.currentPrice.toFixed(2)}
                </td>
                <td
                  className={`py-3 text-right font-mono font-medium ${
                    pos.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}
                  <span className="text-xs ml-1">
                    ({pos.pnlPercent >= 0 ? '+' : ''}
                    {pos.pnlPercent.toFixed(1)}%)
                  </span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const PnLChart: React.FC = () => {
  const { pnlHistory } = useTradingStore();

  const formatData = pnlHistory.map((p) => ({
    time: new Date(p.timestamp).toLocaleTimeString(),
    value: p.cumulative,
  }));

  const isPositive =
    formatData.length > 0 && formatData[formatData.length - 1].value >= 0;

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="text-violet-400" size={18} />
          <h3 className="text-white font-semibold">Cumulative P&L</h3>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={formatData}>
          <defs>
            <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isPositive ? '#10b981' : '#f43f5e'}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isPositive ? '#10b981' : '#f43f5e'}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="time"
            stroke="#71717a"
            tick={{ fill: '#71717a', fontSize: 11 }}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fill: '#71717a', fontSize: 11 }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #3f3f46',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#a1a1aa' }}
            formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
          />
          <ReferenceLine y={0} stroke="#52525b" strokeDasharray="3 3" />
          <Area
            type="monotone"
            dataKey="value"
            stroke={isPositive ? '#10b981' : '#f43f5e'}
            strokeWidth={2}
            fill="url(#pnlGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

const ConnectionStatus: React.FC = () => {
  const { connected } = useTradingStore();

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          connected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
        }`}
      />
      <span className="text-zinc-500 text-sm">
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
};

// =============================================================================
// MAIN APP
// =============================================================================

const App: React.FC = () => {
  const { riskMetrics } = useTradingStore();

  // Connect to WebSocket
  useWebSocket('ws://localhost:8080/ws/market-data');

  // Simulate data for demo
  useEffect(() => {
    const store = useTradingStore.getState();
    store.setConnected(true);

    // Demo data
    const symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NVDA'];
    const interval = setInterval(() => {
      // Add random signal
      store.addSignal({
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        direction: Math.random() > 0.5 ? 1 : -1,
        confidence: 0.5 + Math.random() * 0.5,
        uncertainty: Math.random() * 0.3,
        regimeId: Math.floor(Math.random() * 3),
        timestamp: Date.now(),
      });

      // Update P&L
      const lastPnl =
        store.pnlHistory.length > 0
          ? store.pnlHistory[store.pnlHistory.length - 1].cumulative
          : 0;
      store.addPnLPoint({
        timestamp: Date.now(),
        value: (Math.random() - 0.48) * 10000,
        cumulative: lastPnl + (Math.random() - 0.48) * 5000,
      });

      // Update risk metrics
      store.setRiskMetrics({
        sharpe: 1.2 + Math.random() * 0.8,
        maxDrawdown: -0.02 - Math.random() * 0.03,
        dailyPnl: lastPnl,
        positionUtilization: 0.4 + Math.random() * 0.3,
        var95: -50000 - Math.random() * 20000,
        regime: ['LOW_VOL', 'NORMAL', 'HIGH_VOL', 'CRISIS'][
          Math.floor(Math.random() * 4)
        ] as RiskMetrics['regime'],
      });

      // Update positions
      store.setPositions(
        symbols.slice(0, 4).map((sym) => ({
          symbol: sym,
          quantity: Math.floor(Math.random() * 2000) - 1000,
          entryPrice: 100 + Math.random() * 100,
          currentPrice: 100 + Math.random() * 100,
          pnl: (Math.random() - 0.5) * 10000,
          pnlPercent: (Math.random() - 0.5) * 10,
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Zap className="text-violet-500" size={24} />
              <span className="text-xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                Elite Quant
              </span>
            </div>
            <RegimeBadge regime={riskMetrics.regime} />
          </div>
          <ConnectionStatus />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Sharpe Ratio"
            value={riskMetrics.sharpe.toFixed(2)}
            icon={<TrendingUp size={18} />}
            color={riskMetrics.sharpe > 1 ? 'green' : 'yellow'}
          />
          <MetricCard
            title="Max Drawdown"
            value={`${(riskMetrics.maxDrawdown * 100).toFixed(2)}%`}
            icon={<AlertTriangle size={18} />}
            color={riskMetrics.maxDrawdown > -0.05 ? 'green' : 'red'}
          />
          <MetricCard
            title="Daily P&L"
            value={`$${(riskMetrics.dailyPnl / 1000).toFixed(1)}k`}
            change={riskMetrics.dailyPnl / 1000}
            icon={<Activity size={18} />}
            color={riskMetrics.dailyPnl >= 0 ? 'green' : 'red'}
          />
          <MetricCard
            title="Position Util"
            value={`${(riskMetrics.positionUtilization * 100).toFixed(0)}%`}
            icon={<Shield size={18} />}
            color={riskMetrics.positionUtilization < 0.7 ? 'green' : 'yellow'}
          />
        </div>

        {/* Charts and Signals */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <PnLChart />
          </div>
          <SignalFeed />
        </div>

        {/* Positions */}
        <PositionsTable />
      </main>
    </div>
  );
};

export default App;

