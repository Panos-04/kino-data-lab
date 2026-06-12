import { useEffect, useMemo, useState } from "react";
import {
    fetchAIResults,
} from "../api/kino";
import type { AIAuditExample, AIProfitSummary, AIResultsResponse } from "../api/kino";
import usePageTitle from "../hooks/usePageTitle";

function formatMoney(value?: number | null) {
    if (value === undefined || value === null) return "—";

    return `€${value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPercent(value?: number | null) {
    if (value === undefined || value === null) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function getProfitClass(value?: number | null) {
    if (value === undefined || value === null) return "";
    if (value > 0) return "good";
    if (value < 0) return "bad";
    return "";
}

function getDecisionClass(decision?: string) {
    if (decision === "PLAY") return "good";
    if (decision === "SKIP") return "bad";
    return "neutral";
}

function ProfitCard({
    title,
    summary,
    highlight,
}: {
    title: string;
    summary?: AIProfitSummary;
    highlight?: boolean;
}) {
    if (!summary) {
        return (
            <div className="ai-profit-card">
                <h3>{title}</h3>
                <p>No data</p>
            </div>
        );
    }

    return (
        <div className={`ai-profit-card ${highlight ? "highlight" : ""}`}>
            <h3>{title}</h3>

            <div className="ai-profit-grid">
                <span>Cost</span>
                <strong>{formatMoney(summary.total_cost)}</strong>

                <span>Return</span>
                <strong>{formatMoney(summary.total_return)}</strong>

                <span>Profit</span>
                <strong className={getProfitClass(summary.total_profit)}>
                    {formatMoney(summary.total_profit)}
                </strong>

                <span>ROI</span>
                <strong className={getProfitClass(summary.roi)}>
                    {formatPercent(summary.roi)}
                </strong>

                <span>Paying rounds</span>
                <strong>{summary.paying_rounds ?? "—"}</strong>

                <span>Paying rate</span>
                <strong>{formatPercent(summary.paying_round_rate)}</strong>

                <span>Dead-zone rate</span>
                <strong className="bad">{formatPercent(summary.dead_zone_rate)}</strong>
            </div>

            <div className="ai-hit-distribution">
                {Object.entries(summary.hit_distribution ?? {}).map(([hits, count]) => (
                    <span key={hits}>
                        {hits}: {count}
                    </span>
                ))}
            </div>
        </div>
    );
}

function ConfidenceTables({
    result,
}: {
    result: AIResultsResponse;
}) {
    const thresholds = result.data?.confidence_summary?.thresholds ?? {};
    const buckets = result.data?.confidence_summary?.buckets ?? {};

    return (
        <div className="ai-panel">
            <h2>Confidence gate results</h2>
            <p>
                This shows what would happen if we only played predictions above a
                confidence threshold.
            </p>

            <div className="ai-table-wrap">
                <table className="ai-table">
                    <thead>
                        <tr>
                            <th>Threshold</th>
                            <th>Played</th>
                            <th>Skipped</th>
                            <th>ROI</th>
                            <th>Profit</th>
                            <th>Paying rate</th>
                            <th>Dead-zone</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(thresholds).map(([threshold, value]) => (
                            <tr key={threshold}>
                                <td>{threshold}+</td>
                                <td>{value.played_decisions}</td>
                                <td>{value.skipped_decisions}</td>
                                <td className={getProfitClass(value.roi)}>
                                    {formatPercent(value.roi)}
                                </td>
                                <td className={getProfitClass(value.profit)}>
                                    {formatMoney(value.profit)}
                                </td>
                                <td>{formatPercent(value.paying_round_rate)}</td>
                                <td className="bad">{formatPercent(value.dead_zone_rate)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <h3>Confidence buckets</h3>

            <div className="ai-table-wrap">
                <table className="ai-table">
                    <thead>
                        <tr>
                            <th>Bucket</th>
                            <th>Decisions</th>
                            <th>ROI</th>
                            <th>Profit</th>
                            <th>Paying rate</th>
                            <th>Dead-zone</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(buckets).map(([bucket, value]) => (
                            <tr key={bucket}>
                                <td>{bucket.replace("_", "–")}</td>
                                <td>{value.decisions}</td>
                                <td className={getProfitClass(value.roi)}>
                                    {formatPercent(value.roi)}
                                </td>
                                <td className={getProfitClass(value.profit)}>
                                    {formatMoney(value.profit)}
                                </td>
                                <td>{formatPercent(value.paying_round_rate)}</td>
                                <td className="bad">{formatPercent(value.dead_zone_rate)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function AuditExampleCard({ item }: { item: AIAuditExample }) {
    const bestRound = [...(item.round_details ?? [])].sort(
        (a, b) => b.hit_count - a.hit_count
    )[0];

    const worstRound = [...(item.round_details ?? [])].sort(
        (a, b) => a.hit_count - b.hit_count
    )[0];

    return (
        <div className="ai-audit-card">
            <div className="ai-audit-card-head">
                <div>
                    <strong>Draw {item.draw_id}</strong>
                    <p>
                        {item.operation} · {item.regime} · {item.zone} · mode{" "}
                        {item.selected_mode}
                    </p>
                </div>

                <div className="ai-audit-score">
                    <span>Score</span>
                    <strong>{item.confidence_score.toFixed(1)}</strong>
                </div>
            </div>

            <div className="ai-audit-numbers">
                {item.selected_numbers.map((number) => (
                    <span key={number}>{number}</span>
                ))}
            </div>

            <div className="ai-profit-grid compact">
                <span>Cost</span>
                <strong>{formatMoney(item.cost)}</strong>

                <span>Return</span>
                <strong>{formatMoney(item.return)}</strong>

                <span>Profit</span>
                <strong className={getProfitClass(item.profit)}>
                    {formatMoney(item.profit)}
                </strong>

                <span>ROI</span>
                <strong className={getProfitClass(item.roi)}>
                    {formatPercent(item.roi)}
                </strong>
            </div>

            <div className="ai-mini-details">
                <span>
                    Best round:{" "}
                    <strong>{bestRound ? `${bestRound.hit_count} hits` : "—"}</strong>
                </span>
                <span>
                    Worst round:{" "}
                    <strong>{worstRound ? `${worstRound.hit_count} hits` : "—"}</strong>
                </span>
            </div>

            <div className="ai-hit-distribution">
                {Object.entries(item.hit_distribution ?? {}).map(([hits, count]) => (
                    <span key={hits}>
                        {hits}: {count}
                    </span>
                ))}
            </div>

            <details className="ai-details">
                <summary>Why this decision?</summary>
                <ul>
                    {item.confidence_reasons.map((reason) => (
                        <li key={reason}>{reason}</li>
                    ))}
                </ul>
            </details>

            <details className="ai-details">
                <summary>Number explanations</summary>
                <div className="ai-number-explanations">
                    {(item.number_explanations ?? []).map((entry) => (
                        <div key={entry.number}>
                            <strong>{entry.number}</strong>
                            <span>{entry.components.join(", ")}</span>
                        </div>
                    ))}
                </div>
            </details>
        </div>
    );
}

function AuditSection({
    title,
    items,
}: {
    title: string;
    items?: AIAuditExample[];
}) {
    return (
        <div className="ai-panel">
            <h2>{title}</h2>

            {!items?.length ? (
                <p>No examples found for this category.</p>
            ) : (
                <div className="ai-audit-grid">
                    {items.map((item) => (
                        <AuditExampleCard key={`${item.draw_id}-${item.profit}`} item={item} />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function AIResultsPage() {
    usePageTitle("AI Results | KINO Data Lab");

    const [result, setResult] = useState<AIResultsResponse | null>(null);
    const [loading, setLoading] = useState(true);

    async function load() {
        setLoading(true);

        try {
            const data = await fetchAIResults();
            setResult(data);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        load();
    }, []);

    const topScores = result?.data?.top_pick_latest_scores ?? [];
    const featureImportance = result?.data?.feature_importance ?? [];
    const audit = result?.data?.audit_examples_by_quality;

    const bestProfitMode = useMemo(() => {
        if (!result?.data) return null;

        const entries = [
            ["raw", result.data.raw_profit_summary],
            ["spread", result.data.spread_profit_summary],
            ["hybrid", result.data.hybrid_profit_summary],
            ["relaxed_hybrid", result.data.relaxed_hybrid_profit_summary],
            ["miss", result.data.miss_profit_summary],
            ["regime", result.data.regime_profit_summary],
            ["rescue_1", result.data.rescue_1_profit_summary],
            ["smart_rescue_1", result.data.smart_rescue_1_profit_summary],
            ["safe_smart_rescue_1", result.data.safe_smart_rescue_1_profit_summary],
            ["random", result.data.random_profit_summary],
            ["rescue_2", result.data.rescue_2_profit_summary],
            ["swap_model_1", result.data.swap_model_1_profit_summary],
        ] as const;

        return entries
            .filter(([, summary]) => summary)
            .sort(
                ([, a], [, b]) =>
                    (b?.roi ?? Number.NEGATIVE_INFINITY) -
                    (a?.roi ?? Number.NEGATIVE_INFINITY)
            )[0]?.[0];
    }, [result]);

    if (loading) {
        return (
            <main className="ai-page">
                <h1>KINO AI Results</h1>
                <p>Loading...</p>
            </main>
        );
    }

    if (!result) {
        return (
            <main className="ai-page">
                <h1>KINO AI Results</h1>
                <p>No AI result found.</p>
            </main>
        );
    }

    return (
        <main className="ai-page">
            <div className="ai-page-head">
                <div>
                    <h1>KINO AI Results</h1>
                    <p>
                        Confidence-aware prediction audit, ROI testing, and decision
                        examples.
                    </p>
                </div>

                <button onClick={load}>Reload</button>
            </div>

            <section className="ai-summary-grid">
                <div className="ai-summary-card">
                    <span>Model avg hits</span>
                    <strong>{result.model_top20_hits?.toFixed(3)}</strong>
                    <small>baseline {result.baseline_top20_hits?.toFixed(3)}</small>
                </div>

                <div className="ai-summary-card">
                    <span>Lift</span>
                    <strong className={getProfitClass(result.lift)}>
                        {result.lift >= 0 ? "+" : ""}
                        {result.lift?.toFixed(3)}
                    </strong>
                    <small>hits above baseline</small>
                </div>

                <div className="ai-summary-card">
                    <span>Latest decision</span>
                    <strong
                        className={getDecisionClass(result.data?.latest_confidence?.decision)}
                    >
                        {result.data?.latest_confidence?.decision ?? "—"}
                    </strong>
                    <small>
                        score {result.data?.latest_confidence?.confidence_score ?? "—"}
                    </small>
                </div>

                <div className="ai-summary-card">
                    <span>Best ROI mode</span>
                    <strong>{bestProfitMode ?? "—"}</strong>
                    <small>based on saved backtest</small>
                </div>

                <div className="ai-summary-card">
                    <span>Latest operation</span>
                    <strong>{result.data?.latest_operation ?? "—"}</strong>
                    <small>
                        {result.data?.latest_zone ?? "—"} · streak{" "}
                        {result.data?.latest_operation_streak_length ?? "—"}
                    </small>
                </div>

                <div className="ai-summary-card">
                    <span>Cost / decision</span>
                    <strong>{formatMoney(result.data?.cost_per_combo_decision)}</strong>
                    <small>
                        {result.data?.horizon ?? "—"} rounds · stake €
                        {result.data?.stake ?? "—"}
                    </small>
                </div>
            </section>

            <section className="ai-profit-mode-grid">
                <ProfitCard
                    title="Raw"
                    summary={result.data?.raw_profit_summary}
                    highlight={bestProfitMode === "raw"}
                />
                <ProfitCard
                    title="Spread"
                    summary={result.data?.spread_profit_summary}
                    highlight={bestProfitMode === "spread"}
                />
                <ProfitCard
                    title="Hybrid"
                    summary={result.data?.hybrid_profit_summary}
                    highlight={bestProfitMode === "hybrid"}
                />
                <ProfitCard
                    title="Miss / Low probability"
                    summary={result.data?.miss_profit_summary}
                    highlight={bestProfitMode === "miss"}
                />
                <ProfitCard
                    title="Regime-aware"
                    summary={result.data?.regime_profit_summary}
                    highlight={bestProfitMode === "regime"}
                />
                <ProfitCard
                    title="Rescue 1"
                    summary={result.data?.rescue_1_profit_summary}
                    highlight={bestProfitMode === "rescue_1"}
                />

                <ProfitCard
                    title="Smart Rescue 1"
                    summary={result.data?.smart_rescue_1_profit_summary}
                    highlight={bestProfitMode === "smart_rescue_1"}
                />

                <ProfitCard
                    title="Random"
                    summary={result.data?.random_profit_summary}
                    highlight={bestProfitMode === "random"}
                />
                <ProfitCard
                    title="Safe Smart Rescue 1"
                    summary={result.data?.safe_smart_rescue_1_profit_summary}
                    highlight={bestProfitMode === "safe_smart_rescue_1"}
                />

                <ProfitCard
                    title="Rescue 2"
                    summary={result.data?.rescue_2_profit_summary}
                    highlight={bestProfitMode === "rescue_2"}
                />
                <ProfitCard
                    title="V8 Swap Model 1"
                    summary={result.data?.swap_model_1_profit_summary}
                    highlight={bestProfitMode === "swap_model_1"}
                />
            </section>

            <section className="ai-two-column">
                <div className="ai-panel">
                    <h2>Latest confidence-aware picks</h2>

                    <div className="ai-pick-list">
                        {topScores.map((score) => (
                            <div key={score.number} className="ai-pick-row">
                                <strong>
                                    #{score.rank} — {score.number}
                                </strong>
                                <div className="ai-pick-bar">
                                    <span style={{ width: `${Math.min(score.probability_percent * 8, 100)}%` }} />
                                </div>
                                <span>{score.probability_percent?.toFixed(4)}%</span>
                            </div>
                        ))}
                    </div>

                    <details className="ai-details" open>
                        <summary>Latest confidence reasons</summary>
                        <ul>
                            {(result.data?.latest_confidence?.reasons ?? []).map((reason) => (
                                <li key={reason}>{reason}</li>
                            ))}
                        </ul>
                    </details>
                </div>

                <div className="ai-panel">
                    <h2>Strongest model features</h2>

                    <div className="ai-feature-grid">
                        {featureImportance.slice(0, 12).map((feature) => (
                            <div key={feature.feature} className="ai-feature-card">
                                <strong>{feature.feature}</strong>
                                <span>coef: {feature.coefficient}</span>
                                <span>strength: {feature.absolute_strength}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <ConfidenceTables result={result} />

            <AuditSection
                title="6 high-confidence wins"
                items={audit?.high_confidence_wins}
            />

            <AuditSection
                title="6 high-confidence losses"
                items={audit?.high_confidence_losses}
            />

            <AuditSection
                title="6 low-confidence wins"
                items={audit?.low_confidence_wins}
            />

            <AuditSection
                title="6 low-confidence losses"
                items={audit?.low_confidence_losses}
            />
        </main>
    );
}