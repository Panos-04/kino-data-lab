import { useEffect, useMemo, useState } from "react";
import { fetchShapeMovements } from "../api/kino";
import type { ShapeMovementsResult } from "../api/kino";
import usePageTitle from "../hooks/usePageTitle";

function formatShapeName(shape: string) {
  return shape
    .replaceAll("_", " ")
    .replace("2x2", "2×2")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function ShapeMovementsPage() {
  usePageTitle("Shape Movements");

  const [shape, setShape] = useState("cross");
  const [mode, setMode] = useState("one-to-one");
  const [minHits, setMinHits] = useState(4);
  const [future, setFuture] = useState(10);
  const [limit, setLimit] = useState(30);

  const [result, setResult] = useState<ShapeMovementsResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function runTest() {
    setLoading(true);

    try {
      const data = await fetchShapeMovements({
        shape,
        mode,
        min_hits: minHits,
        future,
        limit,
      });

      setResult(data);
    } catch (error) {
      console.error("Failed to fetch shape movements:", error);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    runTest();
  }, []);

  const maxVectorCount = useMemo(() => {
    if (!result) return 1;
    return Math.max(...result.vector_summary.map((item) => item.count), 1);
  }, [result]);

  const maxGapCount = useMemo(() => {
    if (!result) return 1;
    return Math.max(...result.gap_summary.map((item) => item.count), 1);
  }, [result]);

  return (
    <main className="page shape-movements-page">
      <header className="topbar">
        <div>
          <h1>KINO Shape Movements</h1>
          <p>
            Stored movement vectors between detected shape events.
          </p>
        </div>
      </header>

      <section className="combo-controls">
        <div className="control-group">
          <label>Shape</label>
          <select value={shape} onChange={(event) => setShape(event.target.value)}>
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
          <label>Mode</label>
          <select value={mode} onChange={(event) => setMode(event.target.value)}>
            <option value="one-to-one">One-to-one</option>
            <option value="nearest">Nearest</option>
            <option value="best-overlap">Best overlap</option>
            <option value="all">All links</option>
          </select>
        </div>

        <div className="control-group">
          <label>Min hits</label>
          <select
            value={minHits}
            onChange={(event) => setMinHits(Number(event.target.value))}
          >
            <option value={3}>3+</option>
            <option value={4}>4+</option>
            <option value={5}>5+</option>
          </select>
        </div>

        <div className="control-group">
          <label>Future window</label>
          <select
            value={future}
            onChange={(event) => setFuture(Number(event.target.value))}
          >
            <option value={5}>Next 5 games</option>
            <option value={10}>Next 10 games</option>
            <option value={20}>Next 20 games</option>
            <option value={50}>Next 50 games</option>
          </select>
        </div>

        <div className="control-group">
          <label>Limit</label>
          <select
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
          >
            <option value={20}>20 rows</option>
            <option value={30}>30 rows</option>
            <option value={50}>50 rows</option>
          </select>
        </div>

        <button className="run-test-button" onClick={runTest} disabled={loading}>
          {loading ? "Loading..." : "Load movements"}
        </button>
      </section>

      {result && (
        <>
          <section className="combo-summary">
            <div className="summary-card">
              <span>Shape</span>
              <strong>{formatShapeName(result.shape)}</strong>
            </div>

            <div className="summary-card">
              <span>Total movements</span>
              <strong>{result.total_movements}</strong>
            </div>

            <div className="summary-card">
              <span>Mode</span>
              <strong>{result.mode}</strong>
            </div>

            <div className="summary-card">
              <span>Future window</span>
              <strong>{result.future_window}</strong>
              <small>games</small>
            </div>
          </section>

          <section className="movement-grid">
            <div className="movement-panel">
              <h2>Most common movement vectors</h2>

              <div className="movement-list">
                {result.vector_summary.map((item) => (
                  <div
                    className="movement-row"
                    key={`${item.delta_row}-${item.delta_col}`}
                  >
                    <div className="movement-vector">
                      Δr {item.delta_row >= 0 ? "+" : ""}
                      {item.delta_row}, Δc {item.delta_col >= 0 ? "+" : ""}
                      {item.delta_col}
                    </div>

                    <div className="movement-bar-track">
                      <div
                        className="movement-bar-fill"
                        style={{
                          width: `${(item.count / maxVectorCount) * 100}%`,
                        }}
                      />
                    </div>

                    <strong>{item.count}</strong>
                    <span>{item.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="movement-panel">
              <h2>Gap distribution</h2>

              <div className="movement-list">
                {result.gap_summary.map((item) => (
                  <div className="movement-row" key={item.gap}>
                    <div className="movement-vector">
                      {item.gap} game{item.gap === 1 ? "" : "s"}
                    </div>

                    <div className="movement-bar-track">
                      <div
                        className="movement-bar-fill"
                        style={{
                          width: `${(item.count / maxGapCount) * 100}%`,
                        }}
                      />
                    </div>

                    <strong>{item.count}</strong>
                    <span>{item.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="movement-panel">
            <h2>Most common center-to-center moves</h2>

            <div className="center-move-grid">
              {result.center_summary.map((item) => (
                <article
                  className="center-move-card"
                  key={`${item.from_center}-${item.to_center}`}
                >
                  <strong>
                    {item.from_center} → {item.to_center}
                  </strong>
                  <span>{item.count} times</span>
                  <small>{item.percentage}%</small>
                </article>
              ))}
            </div>
          </section>

          <section className="movement-panel">
            <h2>Recent movement examples</h2>

            <div className="movement-examples">
              {result.examples.map((item) => (
                <article className="movement-example-card" key={item.id}>
                  <header>
                    <strong>
                      {item.from_center} → {item.to_center}
                    </strong>
                    <span>
                      Δr {item.delta_row >= 0 ? "+" : ""}
                      {item.delta_row}, Δc {item.delta_col >= 0 ? "+" : ""}
                      {item.delta_col}
                    </span>
                  </header>

                  <p>
                    Draw {item.from_draw_id} → {item.to_draw_id}
                  </p>

                  <div className="movement-example-meta">
                    <span>Gap: {item.gap}</span>
                    <span>Overlap: {item.overlap_score}</span>
                    <span>Distance: {item.distance_score}</span>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}

      {!result && !loading && (
        <section className="relation-empty">
          <h2>No stored movements found</h2>
          <p>
            Build movements first with the Django command, then reload this page.
          </p>
        </section>
      )}
    </main>
  );
}

export default ShapeMovementsPage;