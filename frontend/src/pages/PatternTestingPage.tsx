import { useEffect, useState } from "react";
import { fetchPatternTest } from "../api/kino";
import type { PatternExample, PatternTestResult } from "../api/kino";
import type { PatternRepeatSummary } from "../api/kino";
import usePageTitle from "../hooks/usePageTitle";
const BOARD_NUMBERS = Array.from({ length: 80 }, (_, index) => index + 1);

function RepeatRateTable({
    title,
    items,
}: {
    title: string;
    items: PatternRepeatSummary[];
}) {
    const windows = [1, 5, 10, 20, 50, 100];

    return (
        <section className="pattern-panel repeat-rate-panel">
            <h2>{title}</h2>

            <div className="repeat-table">
                <div className="repeat-table-header">
                    <span>Pattern</span>
                    <span>Events</span>
                    {windows.map((window) => (
                        <span key={window}>≤{window}</span>
                    ))}
                </div>

                {items.map((item) => (
                    <div
                        className="repeat-table-row"
                        key={`${item.type}-${item.group}`}
                    >
                        <strong>
                            {item.type === "row" ? "Row" : "Column"} {item.group}
                        </strong>

                        <span>{item.events}</span>

                        {windows.map((window) => {
                            const rate = item.repeat_rates.find(
                                (entry) => entry.within_games === window
                            );

                            return (
                                <span key={window} className="repeat-rate-cell">
                                    {rate ? `${rate.repeat_rate}%` : "-"}
                                </span>
                            );
                        })}
                    </div>
                ))}
            </div>
        </section>
    );
}

function MiniPatternBoard({ pattern }: { pattern: PatternExample }) {
    const drawSet = new Set(pattern.draw_numbers);
    const hitSet = new Set(pattern.hit_numbers);

    return (
        <div className="mini-board-card">
            <header>
                <strong>
                    {pattern.type === "row" ? "Row" : "Column"} {pattern.group}
                </strong>
                <span>Draw {pattern.draw_id}</span>
            </header>

            <p>
                {pattern.hit_count} hits: {pattern.hit_numbers.join(", ")}
            </p>

            <div className="mini-kino-board">
                {BOARD_NUMBERS.map((number) => {
                    const isDrawn = drawSet.has(number);
                    const isPatternHit = hitSet.has(number);

                    return (
                        <div
                            key={number}
                            className={
                                isPatternHit
                                    ? "mini-number pattern-hit"
                                    : isDrawn
                                        ? "mini-number drawn"
                                        : "mini-number"
                            }
                        >
                            {number}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function PatternTestingPage() {
    const [rowThreshold, setRowThreshold] = useState(6);
    const [columnThreshold, setColumnThreshold] = useState(5);
    const [limit, setLimit] = useState(12);
    usePageTitle("Pattern Testing");
    const [result, setResult] = useState<PatternTestResult | null>(null);
    const [loading, setLoading] = useState(false);

    async function runTest() {
        setLoading(true);

        try {
            const data = await fetchPatternTest({
                row_threshold: rowThreshold,
                column_threshold: columnThreshold,
                limit,
            });

            setResult(data);
        } catch (error) {
            console.error("Failed to fetch pattern test:", error);
            setResult(null);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        runTest();
    }, []);

    return (
        <main className="page pattern-page">
            <header className="topbar">
                <div>
                    <h1>KINO Pattern Testing</h1>
                    <p>
                        Search row and column patterns across imported KINO draw history.
                    </p>
                </div>
            </header>

            <section className="combo-controls">
                <div className="control-group">
                    <label>Row threshold</label>
                    <select
                        value={rowThreshold}
                        onChange={(event) => setRowThreshold(Number(event.target.value))}
                    >
                        <option value={5}>5+ hits in row</option>
                        <option value={6}>6+ hits in row</option>
                        <option value={7}>7+ hits in row</option>
                        <option value={8}>8+ hits in row</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Column threshold</label>
                    <select
                        value={columnThreshold}
                        onChange={(event) => setColumnThreshold(Number(event.target.value))}
                    >
                        <option value={4}>4+ hits in column</option>
                        <option value={5}>5+ hits in column</option>
                        <option value={6}>6+ hits in column</option>
                        <option value={7}>7+ hits in column</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Examples</label>
                    <select
                        value={limit}
                        onChange={(event) => setLimit(Number(event.target.value))}
                    >
                        <option value={6}>6 examples</option>
                        <option value={12}>12 examples</option>
                        <option value={24}>24 examples</option>
                    </select>
                </div>

                <button className="run-test-button" onClick={runTest} disabled={loading}>
                    {loading ? "Running..." : "Run test"}
                </button>
            </section>

            {result && (
                <>
                    <section className="combo-summary">
                        <div className="summary-card">
                            <span>Total draws</span>
                            <strong>{result.total_draws}</strong>
                        </div>

                        <div className="summary-card">
                            <span>Row events</span>
                            <strong>{result.row_pattern_count}</strong>
                            <small>{result.row_pattern_percentage}%</small>
                        </div>

                        <div className="summary-card">
                            <span>Column events</span>
                            <strong>{result.column_pattern_count}</strong>
                            <small>{result.column_pattern_percentage}%</small>
                        </div>

                        <div className="summary-card">
                            <span>Best streak</span>
                            <strong>{result.streaks[0]?.streak ?? 0}</strong>
                            <small>
                                {result.streaks[0]
                                    ? `${result.streaks[0].type} ${result.streaks[0].group}`
                                    : "None"}
                            </small>
                        </div>
                    </section>

                    <section className="pattern-grid">
                        <div className="pattern-panel">
                            <h2>Rows with most hits</h2>

                            {result.row_summary.map((row) => (
                                <div className="pattern-summary-row" key={row.group}>
                                    <span>Row {row.group}</span>
                                    <strong>{row.count}</strong>
                                    <small>{row.percentage}%</small>
                                </div>
                            ))}
                        </div>

                        <div className="pattern-panel">
                            <h2>Columns with most hits</h2>

                            {result.column_summary.map((column) => (
                                <div className="pattern-summary-row" key={column.group}>
                                    <span>Column {column.group}</span>
                                    <strong>{column.count}</strong>
                                    <small>{column.percentage}%</small>
                                </div>
                            ))}
                        </div>

                        <div className="pattern-panel">
                            <h2>Best continuation streaks</h2>

                            {result.streaks.map((item) => (
                                <div
                                    className="pattern-summary-row"
                                    key={`${item.type}-${item.group}`}
                                >
                                    <span>
                                        {item.type === "row" ? "Row" : "Column"} {item.group}
                                    </span>
                                    <strong>{item.streak}</strong>
                                    <small>draws</small>
                                </div>
                            ))}
                        </div>
                    </section>
                    <section className="pattern-gap-grid">
                        <div className="pattern-panel">
                            <h2>How many games apart — Rows</h2>

                            {result.row_gap_summary.slice(0, 8).map((item) => (
                                <div
                                    className="gap-summary-row"
                                    key={`${item.type}-${item.group}`}
                                >
                                    <div>
                                        <strong>Row {item.group}</strong>
                                        <span>{item.events} events</span>
                                    </div>

                                    <div>
                                        <small>Avg gap</small>
                                        <strong>{item.avg_gap ?? "-"}</strong>
                                    </div>

                                    <div>
                                        <small>Min</small>
                                        <strong>{item.min_gap ?? "-"}</strong>
                                    </div>

                                    <div>
                                        <small>Max</small>
                                        <strong>{item.max_gap ?? "-"}</strong>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="pattern-panel">
                            <h2>How many games apart — Columns</h2>

                            {result.column_gap_summary.slice(0, 10).map((item) => (
                                <div
                                    className="gap-summary-row"
                                    key={`${item.type}-${item.group}`}
                                >
                                    <div>
                                        <strong>Column {item.group}</strong>
                                        <span>{item.events} events</span>
                                    </div>

                                    <div>
                                        <small>Avg gap</small>
                                        <strong>{item.avg_gap ?? "-"}</strong>
                                    </div>

                                    <div>
                                        <small>Min</small>
                                        <strong>{item.min_gap ?? "-"}</strong>
                                    </div>

                                    <div>
                                        <small>Max</small>
                                        <strong>{item.max_gap ?? "-"}</strong>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>

                    <section className="repeat-rate-grid">
                        <RepeatRateTable
                            title="Repeat rate — Rows"
                            items={result.row_repeat_summary}
                        />

                        <RepeatRateTable
                            title="Repeat rate — Columns"
                            items={result.column_repeat_summary}
                        />
                    </section>

                    <section className="pattern-examples-section">
                        <h2>Example row patterns</h2>

                        <div className="mini-board-grid">
                            {result.row_patterns.map((pattern) => (
                                <MiniPatternBoard
                                    key={`${pattern.draw_id}-${pattern.type}-${pattern.group}`}
                                    pattern={pattern}
                                />
                            ))}
                        </div>
                    </section>

                    <section className="pattern-examples-section">
                        <h2>Example column patterns</h2>

                        <div className="mini-board-grid">
                            {result.column_patterns.map((pattern) => (
                                <MiniPatternBoard
                                    key={`${pattern.draw_id}-${pattern.type}-${pattern.group}`}
                                    pattern={pattern}
                                />
                            ))}
                        </div>
                    </section>
                </>
            )}
        </main>
    );
}

export default PatternTestingPage;