import { useEffect, useMemo, useState } from "react";
import { fetchComboTest } from "../api/kino";
import type { ComboTestResult } from "../api/kino";
import usePageTitle from "../hooks/usePageTitle";
type Strategy = "cold" | "hot" | "middle";

function ComboTestingPage() {
    const [strategy, setStrategy] = useState<Strategy>("cold");
    const [windowSize, setWindowSize] = useState(20);
    const [stepSize, setStepSize] = useState(10);
    const [pickCount, setPickCount] = useState(5);
    const [futureSize, setFutureSize] = useState(1);

    const [result, setResult] = useState<ComboTestResult | null>(null);
    const [loading, setLoading] = useState(false);
    usePageTitle("Combo Testing");
    async function runTest() {
        setLoading(true);

        try {
            const data = await fetchComboTest({
                strategy,
                window: windowSize,
                step: stepSize,
                pick: pickCount,
                future: futureSize,
            });

            setResult(data);
        } catch (error) {
            console.error("Failed to run combo test:", error);
            setResult(null);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        runTest();
    }, []);

    const maxCount = useMemo(() => {
        if (!result) return 1;

        return Math.max(...result.distribution.map((row) => row.count), 1);
    }, [result]);

    return (
        <main className="page combo-page">
            <header className="topbar">
                <div>
                    <h1>KINO Combo Testing</h1>
                    <p>
                        Test cold, hot, and middle number combinations against future draws.
                    </p>
                </div>
            </header>

            <section className="combo-controls">
                <div className="control-group">
                    <label>Strategy</label>
                    <select
                        value={strategy}
                        onChange={(event) => setStrategy(event.target.value as Strategy)}
                    >
                        <option value="cold">Cold numbers</option>
                        <option value="hot">Hot numbers</option>
                        <option value="middle">Middle numbers</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Window</label>
                    <select
                        value={windowSize}
                        onChange={(event) => {
                            const value = Number(event.target.value);
                            setWindowSize(value);
                            setStepSize(value === 20 ? 10 : 5);
                        }}
                    >
                        <option value={20}>20 games</option>
                        <option value={10}>10 games</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Pick</label>
                    <select
                        value={pickCount}
                        onChange={(event) => setPickCount(Number(event.target.value))}
                    >
                        <option value={3}>3 numbers</option>
                        <option value={4}>4 numbers</option>
                        <option value={5}>5 numbers</option>
                        <option value={6}>6 numbers</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Future draws</label>
                    <select
                        value={futureSize}
                        onChange={(event) => setFutureSize(Number(event.target.value))}
                    >
                        <option value={1}>Next 1 draw</option>
                        <option value={5}>Next 5 draws</option>
                        <option value={10}>Next 10 draws</option>
                    </select>
                </div>

                <button className="run-test-button" onClick={runTest} disabled={loading}>
                    {loading ? "Running..." : "Run test"}
                </button>
            </section>

            {!result && !loading && (
                <section className="relation-empty">
                    <h2>No result yet</h2>
                    <p>Choose settings and run a combo test.</p>
                </section>
            )}

            {result && (
                <>
                    <section className="combo-summary">
                        <div className="summary-card">
                            <span>Tested draws</span>
                            <strong>{result.tested_draws}</strong>
                        </div>

                        <div className="summary-card">
                            <span>Skipped windows</span>
                            <strong>{result.skipped_windows}</strong>
                        </div>

                        <div className="summary-card">
                            <span>4+ hits</span>
                            <strong>{result.four_plus.count}</strong>
                        </div>

                        <div
                            className={`summary-card ${result.four_plus.difference >= 0 ? "positive" : "negative"
                                }`}
                        >
                            <span>4+ rate vs baseline</span>
                            <strong>
                                {result.four_plus.percentage}% /{" "}
                                {result.four_plus.baseline_percentage}%
                            </strong>
                            <small>
                                {result.four_plus.difference >= 0 ? "+" : ""}
                                {result.four_plus.difference}%
                            </small>
                        </div>
                    </section>

                    <section className="combo-chart-panel">
                        <h2>Hit distribution</h2>

                        <div className="combo-chart">
                            {result.distribution.map((row) => (
                                <div className="combo-bar-row" key={row.hits}>
                                    <div className="combo-bar-label">
                                        {row.hits}/{result.pick_count}
                                    </div>

                                    <div className="combo-bar-track">
                                        <div
                                            className="combo-bar-fill"
                                            style={{
                                                width: `${(row.count / maxCount) * 100}%`,
                                            }}
                                        />
                                    </div>

                                    <div className="combo-bar-value">
                                        <strong>{row.count}</strong>
                                        <span>{row.percentage}%</span>
                                    </div>

                                    <div
                                        className={`combo-diff ${row.difference >= 0 ? "positive" : "negative"
                                            }`}
                                    >
                                        {row.difference >= 0 ? "+" : ""}
                                        {row.difference}%
                                    </div>
                                </div>
                            ))}
                        </div>

                        <p className="combo-note">
                            Difference compares your strategy result against the mathematical
                            random baseline for the same combo size.
                        </p>
                    </section>

                    <section className="best-results-panel">
                        <h2>Best 4+/5 results</h2>

                        {result.best_results.length === 0 && (
                            <p>No 4+ result found for this test.</p>
                        )}

                        <div className="best-results-grid">
                            {result.best_results.map((item) => (
                                <article
                                    className="best-result-card"
                                    key={`${item.window_id}-${item.draw_id}`}
                                >
                                    <header>
                                        <strong>{item.hit_count}/{result.pick_count}</strong>
                                        <span>Draw {item.draw_id}</span>
                                    </header>

                                    <div>
                                        <p>Combo</p>
                                        <div className="number-pills">
                                            {item.combo.map((number) => (
                                                <span
                                                    key={number}
                                                    className={
                                                        item.hit_numbers.includes(number)
                                                            ? "number-pill hit"
                                                            : "number-pill"
                                                    }
                                                >
                                                    {number}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    <div>
                                        <p>Hit numbers</p>
                                        <div className="number-pills">
                                            {item.hit_numbers.map((number) => (
                                                <span key={number} className="number-pill hit">
                                                    {number}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </article>
                            ))}
                        </div>
                    </section>
                </>
            )}
        </main>
    );
}

export default ComboTestingPage;