import { useEffect, useRef } from "react";
import {
    createChart,
    createSeriesMarkers,
    CandlestickSeries,
    LineSeries,
    type IChartApi,
    type ISeriesApi,
    type CandlestickData,
    type LineData,
    type SeriesMarker,
    type Time,
    type ISeriesMarkersPluginApi,
    ColorType,
} from "lightweight-charts";
import type { OHLCVRow, LedgerRow } from "../api";

interface Props {
    ohlcv: OHLCVRow[];
    ledger: LedgerRow[];
}

export default function TradingChart({ ohlcv, ledger }: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const equityRef = useRef<ISeriesApi<"Line"> | null>(null);
    const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

    // ── Create chart once ──────────────────────────────────────────────
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: 500,
            layout: {
                background: { type: ColorType.Solid, color: "#0f1117" },
                textColor: "#9ca3af",
            },
            grid: {
                vertLines: { color: "#1e2030" },
                horzLines: { color: "#1e2030" },
            },
            crosshair: { mode: 0 },
            rightPriceScale: { borderColor: "#2a2e39" },
            timeScale: { borderColor: "#2a2e39" },
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: "#26a69a",
            downColor: "#ef5350",
            borderDownColor: "#ef5350",
            borderUpColor: "#26a69a",
            wickDownColor: "#ef5350",
            wickUpColor: "#26a69a",
        });

        const equitySeries = chart.addSeries(LineSeries, {
            color: "#42a5f5",
            lineWidth: 2,
            priceScaleId: "left",
        });

        chart.priceScale("left").applyOptions({
            borderColor: "#2a2e39",
            scaleMargins: { top: 0.1, bottom: 0.1 },
        });

        chartRef.current = chart;
        candleRef.current = candleSeries;
        equityRef.current = equitySeries;
        // Initialize markers plugin (empty)
        markersRef.current = createSeriesMarkers(candleSeries, []);

        // Resize handler
        const handleResize = () => {
            if (containerRef.current) {
                chart.applyOptions({ width: containerRef.current.clientWidth });
            }
        };
        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.remove();
            chartRef.current = null;
            candleRef.current = null;
            equityRef.current = null;
            markersRef.current = null;
        };
    }, []);

    // ── Update candlestick data ────────────────────────────────────────
    useEffect(() => {
        if (!candleRef.current || ohlcv.length === 0) return;

        const candles: CandlestickData[] = ohlcv
            .filter((r) => r.open != null && r.high != null && r.low != null && r.close != null)
            .map((r) => ({
                time: r.time as Time,
                open: r.open!,
                high: r.high!,
                low: r.low!,
                close: r.close!,
            }));

        candleRef.current.setData(candles);
        chartRef.current?.timeScale().fitContent();
    }, [ohlcv]);

    // ── Update markers + equity curve from ledger ──────────────────────
    useEffect(() => {
        if (!candleRef.current || !equityRef.current || !markersRef.current || ledger.length === 0) return;

        // Equity curve
        const equityData: LineData[] = ledger
            .filter((r) => r.portfolio_value != null && r.date)
            .map((r) => ({
                time: r.date.slice(0, 10) as Time,
                value: r.portfolio_value!,
            }));
        equityRef.current.setData(equityData);

        // Execution markers
        const markers: SeriesMarker<Time>[] = [];

        for (const row of ledger) {
            if (!row.date) continue;
            const time = row.date.slice(0, 10) as Time;

            if (row.forced_liquidation) {
                markers.push({
                    time,
                    position: "aboveBar",
                    color: "#ab47bc",
                    shape: "arrowDown",
                    text: "STOP OUT",
                });
            } else if (row.transition === "BUY (0->1)") {
                // 0 → 1 transition
                markers.push({
                    time,
                    position: "belowBar",
                    color: "#26a69a",
                    shape: "arrowUp",
                    text: "BUY",
                });
            } else if (row.transition === "SELL (1->0)") {
                // 1 → 0 transition (natural sell)
                markers.push({
                    time,
                    position: "aboveBar",
                    color: "#ef5350",
                    shape: "arrowDown",
                    text: "SELL",
                });
            } else if (row.transition === "FORCED_EXIT") {
                markers.push({
                    time,
                    position: "aboveBar",
                    color: "#ab47bc",
                    shape: "arrowDown",
                    text: "STOP OUT",
                });
            }
        }

        // lightweight-charts requires markers sorted by time
        markers.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));

        markersRef.current.setMarkers(markers);
    }, [ledger]);

    return (
        <div
            ref={containerRef}
            className="w-full rounded-lg border border-gray-800"
        />
    );
}
