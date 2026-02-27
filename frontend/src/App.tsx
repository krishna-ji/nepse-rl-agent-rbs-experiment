import { useEffect, useState, useCallback } from "react";
import {
  BarChart3,
  Activity,
  TrendingDown,
  Percent,
  Target,
  ChevronRight,
  Loader2,
  Crosshair,
  Shield,
} from "lucide-react";
import TradingChart from "./components/TradingChart";
import {
  fetchSystem,
  fetchTickers,
  fetchMetrics,
  fetchLedger,
  fetchOHLCV,
  type SystemTearSheet,
  type TearSheet,
  type LedgerRow,
  type OHLCVRow,
} from "./api";

// ─── KPI Card ─────────────────────────────────────────────────────────────────
function KpiCard({
  label,
  value,
  icon: Icon,
  colorCode,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  colorCode?: boolean;
}) {
  const numVal = parseFloat(value);
  let textColor = "text-gray-100";
  if (colorCode && !isNaN(numVal)) {
    textColor = numVal >= 0 ? "text-emerald-400" : "text-red-400";
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-[#161924] p-4 flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs text-gray-400 uppercase tracking-wide">
        <Icon size={14} />
        {label}
      </div>
      <div className={`text-xl font-semibold ${textColor}`}>{value}</div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [system, setSystem] = useState<SystemTearSheet | null>(null);
  const [tickers, setTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<TearSheet | null>(null);
  const [ledger, setLedger] = useState<LedgerRow[]>([]);
  const [ohlcv, setOhlcv] = useState<OHLCVRow[]>([]);
  const [loading, setLoading] = useState(false);

  // Load system tear sheet + tickers on mount
  useEffect(() => {
    fetchSystem().then(setSystem).catch(console.error);
    fetchTickers().then(setTickers).catch(console.error);
  }, []);

  // Load data when ticker changes
  const loadTickerData = useCallback(async (ticker: string) => {
    setLoading(true);
    try {
      const [m, l, o] = await Promise.all([
        fetchMetrics(ticker),
        fetchLedger(ticker),
        fetchOHLCV(ticker),
      ]);
      setMetrics(m);
      setLedger(l);
      setOhlcv(o);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedTicker) {
      loadTickerData(selectedTicker);
    }
  }, [selectedTicker, loadTickerData]);

  // ─── Derived data ──────────────────────────────────────────────────
  const fmt = (v: number | null | undefined, pct = false) => {
    if (v == null || isNaN(v)) return "—";
    return pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(4);
  };

  const recentTrades = ledger
    .filter(
      (r) =>
        r.transition === "BUY (0->1)" ||
        r.transition === "SELL (1->0)" ||
        r.transition === "FORCED_EXIT" ||
        r.forced_liquidation
    )
    .slice(-10)
    .reverse();

  // ─── Render ────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-gray-800 bg-[#111318] flex flex-col overflow-y-auto">
        <div className="px-4 py-5 text-lg font-bold tracking-tight text-gray-100 border-b border-gray-800">
          <span className="text-emerald-400">NEPSE</span> RL Dashboard
        </div>

        {/* System KPIs */}
        {system && (
          <div className="px-3 py-3 border-b border-gray-800 space-y-2">
            <div className="text-[10px] font-semibold uppercase tracking-widest text-gray-500 px-1">
              System Portfolio
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-[#161924] rounded-lg p-2">
                <div className="text-gray-500">Win Rate</div>
                <div className="text-emerald-400 font-semibold">
                  {(system.win_rate * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-[#161924] rounded-lg p-2">
                <div className="text-gray-500">Trades</div>
                <div className="text-gray-100 font-semibold">
                  {system.total_system_trades}
                </div>
              </div>
              <div className="bg-[#161924] rounded-lg p-2">
                <div className="text-gray-500">Expectancy</div>
                <div
                  className={`font-semibold ${system.system_expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`}
                >
                  {system.system_expectancy.toFixed(4)}
                </div>
              </div>
              <div className="bg-[#161924] rounded-lg p-2">
                <div className="text-gray-500">Profit Factor</div>
                <div className="text-gray-100 font-semibold">
                  {system.profit_factor.toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Ticker list */}
        <div className="px-3 py-3">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-gray-500 mb-2 px-1">
            Tickers
          </div>
          {tickers.length === 0 && (
            <div className="text-xs text-gray-600 px-1">No eval data found</div>
          )}
          {tickers.map((t) => (
            <button
              key={t}
              onClick={() => setSelectedTicker(t)}
              className={`w-full text-left text-xs px-2 py-1.5 rounded mb-0.5 flex items-center gap-1 transition-colors cursor-pointer ${selectedTicker === t
                  ? "bg-emerald-500/10 text-emerald-400"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`}
            >
              <Activity size={12} />
              {t}
            </button>
          ))}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {!selectedTicker && (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Select a ticker to begin analysis
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <Loader2 className="animate-spin mr-2" size={20} />
            Loading…
          </div>
        )}

        {selectedTicker && !loading && (
          <div className="space-y-6">
            {/* Header */}
            <div>
              <h1 className="text-2xl font-bold text-gray-100">
                {selectedTicker}
              </h1>
              <p className="text-xs text-gray-500">latest_eval</p>
            </div>

            {/* KPI Cards */}
            {metrics && (
              <div className="grid grid-cols-5 gap-3">
                <KpiCard
                  label="Total Return"
                  value={fmt(metrics.total_return_agent, true)}
                  icon={BarChart3}
                  colorCode
                />
                <KpiCard
                  label="Buy & Hold"
                  value={fmt(metrics.total_return_baseline, true)}
                  icon={TrendingDown}
                />
                <KpiCard
                  label="Max Drawdown"
                  value={fmt(metrics.max_drawdown_agent, true)}
                  icon={TrendingDown}
                />
                <KpiCard
                  label="Sharpe Ratio"
                  value={fmt(metrics.annualized_sharpe)}
                  icon={Percent}
                  colorCode
                />
                <KpiCard
                  label="Win Rate"
                  value={fmt(metrics.win_rate, true)}
                  icon={Target}
                />
              </div>
            )}

            {/* TradingView Chart */}
            <TradingChart ohlcv={ohlcv} ledger={ledger} />

            {/* Trade Ledger Table */}
            {recentTrades.length > 0 && (
              <div className="rounded-xl border border-gray-800 bg-[#161924] overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-800 text-xs font-semibold uppercase tracking-widest text-gray-500">
                  Recent Executions
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                      <th className="px-4 py-2">Date</th>
                      <th className="px-4 py-2">Action</th>
                      <th className="px-4 py-2 text-right">Price</th>
                      <th className="px-4 py-2 text-right">Portfolio Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTrades.map((row, i) => {
                      let action = row.transition;
                      let actionColor = "text-gray-300";
                      if (row.forced_liquidation) {
                        action = "STOP OUT";
                        actionColor = "text-purple-400";
                      } else if (row.transition === "BUY (0->1)") {
                        action = "BUY";
                        actionColor = "text-emerald-400";
                      } else if (row.transition === "SELL (1->0)") {
                        action = "SELL";
                        actionColor = "text-red-400";
                      } else if (row.transition === "FORCED_EXIT") {
                        action = "STOP OUT";
                        actionColor = "text-purple-400";
                      }

                      return (
                        <tr
                          key={i}
                          className="border-b border-gray-800/50 hover:bg-gray-800/30"
                        >
                          <td className="px-4 py-2 text-gray-300">
                            {row.date?.slice(0, 10)}
                          </td>
                          <td className={`px-4 py-2 font-medium ${actionColor}`}>
                            {action}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {row.close != null ? row.close.toFixed(2) : "—"}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {row.portfolio_value != null
                              ? row.portfolio_value.toFixed(4)
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
