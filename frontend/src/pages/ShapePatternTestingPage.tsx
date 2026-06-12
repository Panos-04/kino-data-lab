import { useEffect, useState } from "react";
import { fetchShapePatternTest } from "../api/kino";
import type { ShapePatternExample, ShapePatternTestResult } from "../api/kino";
import usePageTitle from "../hooks/usePageTitle";
const BOARD_NUMBERS = Array.from({ length: 80 }, (_, index) => index + 1);

function formatShapeName(shape: string) {
    return shape
        .replaceAll("_", " ")
        .replace("2x2", "2×2")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function ShapeMiniBoard({ example }: { example: ShapePatternExample }) {
    const drawSet = new Set(example.draw_numbers);
    const shapeSet = new Set(example.shape_numbers);
    const hitSet = new Set(example.hit_numbers);

    return (
        <article className="mini-board-card">
            <header>
                <strong>{formatShapeName(example.shape)}</strong>
                <span>Draw {example.draw_id}</span>
            </header>

            <p>
                Center {example.center_number} · {example.hit_count}/{example.shape_size} hits
            </p>

            <div className="mini-kino-board">
                {BOARD_NUMBERS.map((number) => {
                    const isHit = hitSet.has(number);
                    const isShape = shapeSet.has(number);
                    const isDrawn = drawSet.has(number);

                    return (
                        <div
                            key={number}
                            className={
                                isHit
                                    ? "mini-number pattern-hit"
                                    : isShape
                                        ? "mini-number shape-cell"
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
        </article>
    );
}

function ShapePatternTestingPage() {
    const [shape, setShape] = useState("all");
    const [minHits, setMinHits] = useState(4);
    const [limit, setLimit] = useState(12);
    usePageTitle("Shape Pattern Testing");
    const [result, setResult] = useState<ShapePatternTestResult | null>(null);
    const [loading, setLoading] = useState(false);

    async function runTest() {
        setLoading(true);

        try {
            const data = await fetchShapePatternTest({
                shape,
                min_hits: shape === "all" ? undefined : minHits,
                limit,
            });

            setResult(data);
        } catch (error) {
            console.error("Failed to fetch shape patterns:", error);
            setResult(null);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        runTest();
    }, []);

    return (
        <main className="page shape-page">
            <header className="topbar">
                <div>
                    <h1>KINO Shape Pattern Testing</h1>
                    <p>
                        Detect crosses, boxes, lines, diagonals, and L-shapes inside KINO draws.
                    </p>
                </div>
            </header>

            <section className="combo-controls">
                <div className="control-group">
                    <label>Shape</label>
                    <select value={shape} onChange={(event) => setShape(event.target.value)}>
                        <option value="all">All shapes</option>
                        <option value="cross">Cross</option>
                        <option value="box_2x2">2×2 Box</option>
                        <option value="l_shape">L-shape</option>
                        <option value="vertical_4">Vertical 4</option>
                        <option value="horizontal_4">Horizontal 4</option>
                        <option value="diagonal_down_4">Diagonal down 4</option>
                        <option value="diagonal_up_4">Diagonal up 4</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Min hits</label>
                    <select
                        value={minHits}
                        onChange={(event) => setMinHits(Number(event.target.value))}
                        disabled={shape === "all"}
                    >
                        <option value={3}>3+</option>
                        <option value={4}>4+</option>
                        <option value={5}>5+</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Examples</label>
                    <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
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
                            <span>Total shape events</span>
                            <strong>{result.total_events}</strong>
                        </div>

                        <div className="summary-card">
                            <span>Draws with shape</span>
                            <strong>{result.draws_with_any_shape}</strong>
                            <small>{result.draws_with_any_shape_percentage}%</small>
                        </div>

                        <div className="summary-card">
                            <span>Avg events per draw</span>
                            <strong>{result.avg_events_per_draw}</strong>
                            <small>Max {result.most_events_in_one_draw}</small>
                        </div>
                    </section>

                    <section className="pattern-grid">
                        <div className="pattern-panel">
                            <h2>Shape summary</h2>

                            {result.shape_summary.map((item) => (
                                <div className="pattern-summary-row" key={item.shape}>
                                    <span>{formatShapeName(item.shape)}</span>
                                    <strong>{item.events}</strong>
                                    <small>{item.draw_percentage}% draws</small>
                                </div>
                            ))}
                        </div>

                        <div className="pattern-panel">
                            <h2>Most common centers</h2>

                            {result.center_summary.slice(0, 12).map((item) => (
                                <div
                                    className="pattern-summary-row"
                                    key={`${item.shape}-${item.center_number}`}
                                >
                                    <span>
                                        {formatShapeName(item.shape)} {item.center_number}
                                    </span>
                                    <strong>{item.events}</strong>
                                    <small>events</small>
                                </div>
                            ))}
                        </div>

                        <div className="pattern-panel">
                            <h2>Hit count split</h2>

                            {Object.entries(result.hit_count_summary).map(([shapeName, rows]) => (
                                <div key={shapeName} className="shape-hit-group">
                                    <h3>{formatShapeName(shapeName)}</h3>

                                    {rows.map((row) => (
                                        <div
                                            className="pattern-summary-row"
                                            key={`${shapeName}-${row.hit_count}`}
                                        >
                                            <span>{row.hit_count} hits</span>
                                            <strong>{row.events}</strong>
                                            <small>events</small>
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </section>

                    <section className="pattern-examples-section">
                        <h2>Shape examples</h2>

                        <div className="mini-board-grid">
                            {result.examples.map((example) => (
                                <ShapeMiniBoard
                                    key={`${example.draw_id}-${example.shape}-${example.center_number}`}
                                    example={example}
                                />
                            ))}
                        </div>
                    </section>
                </>
            )}
        </main>
    );
}

export default ShapePatternTestingPage;