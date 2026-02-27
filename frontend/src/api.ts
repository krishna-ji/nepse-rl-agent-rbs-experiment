const API_BASE = "http://localhost:8000";

// ── System-level macro tear sheet ──────────────────────────────────────────
export interface SystemTearSheet {
    total_system_trades: number;
    total_closed: number;
    total_wins: number;
    win_rate: number;
    profit_factor: number;
    system_expectancy: number;
    gross_profit: number;
    gross_loss: number;
    forced_liquidations: number;
    tickers_evaluated: string[];
}

export async function fetchSystem(): Promise<SystemTearSheet> {
    const res = await fetch(`${API_BASE}/api/system`);
    if (!res.ok) throw new Error("Failed to fetch system tear sheet");
    return res.json();
}

// ── Tickers (flat, no run_id) ─────────────────────────────────────────────
export async function fetchTickers(): Promise<string[]> {
    const res = await fetch(`${API_BASE}/api/tickers`);
    if (!res.ok) throw new Error("Failed to fetch tickers");
    return res.json();
}

// ── Per-ticker tear sheet ─────────────────────────────────────────────────
export interface TearSheet {
    ticker: string;
    total_return_agent: number | null;
    total_return_baseline: number | null;
    excess_return: number | null;
    annualized_sharpe: number | null;
    max_drawdown_agent: number | null;
    max_drawdown_baseline: number | null;
    num_trades: number | null;
    win_rate: number | null;
    avg_win: number | null;
    avg_loss: number | null;
    profit_factor: number | null;
    exposure_pct: number | null;
    forced_liquidations: number | null;
    episode_length: number | null;
    final_portfolio_value: number | null;
}

export async function fetchMetrics(ticker: string): Promise<TearSheet> {
    const res = await fetch(`${API_BASE}/api/metrics/${ticker}`);
    if (!res.ok) throw new Error("Failed to fetch metrics");
    return res.json();
}

// ── Trade ledger ──────────────────────────────────────────────────────────
export interface LedgerRow {
    ticker: string;
    date: string;
    close: number | null;
    action: number | null;
    tsl_level: number | null;
    portfolio_value: number | null;
    position: number | null;
    forced_liquidation: boolean;
    transition: string;
    exit_type: string;
    trade_id: number | null;
}

export async function fetchLedger(ticker: string): Promise<LedgerRow[]> {
    const res = await fetch(`${API_BASE}/api/ledger/${ticker}`);
    if (!res.ok) throw new Error("Failed to fetch ledger");
    return res.json();
}

// ── Raw OHLCV ─────────────────────────────────────────────────────────────
export interface OHLCVRow {
    time: string;
    open: number | null;
    high: number | null;
    low: number | null;
    close: number | null;
    volume: number | null;
}

export async function fetchOHLCV(ticker: string): Promise<OHLCVRow[]> {
    const res = await fetch(`${API_BASE}/api/data/${ticker}`);
    if (!res.ok) throw new Error("Failed to fetch OHLCV data");
    return res.json();
}
