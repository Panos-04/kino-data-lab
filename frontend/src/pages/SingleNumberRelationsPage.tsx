import { useEffect, useState } from "react";
import {
    getRelationSelection,
    getRelationStorageKey,
    type RelationSelection,
} from "../utils/relationSelection";
import { fetchNumberRelations } from "../api/kino";
import type { NumberRelations } from "../api/kino";

function SingleNumberRelationsPage() {
    const [selection, setSelection] = useState<RelationSelection | null>(() =>
        getRelationSelection()
    );
    const [relations, setRelations] = useState<NumberRelations | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        function handleStorage(event: StorageEvent) {
            if (event.key !== getRelationStorageKey()) return;
            setSelection(getRelationSelection());
        }

        function handleCustomEvent() {
            setSelection(getRelationSelection());
        }

        window.addEventListener("storage", handleStorage);
        window.addEventListener("kino-relation-selection", handleCustomEvent);

        return () => {
            window.removeEventListener("storage", handleStorage);
            window.removeEventListener("kino-relation-selection", handleCustomEvent);
        };
    }, []);

    useEffect(() => {
        async function loadRelations() {
            if (!selection) return;

            setLoading(true);

            try {
                const data = await fetchNumberRelations(
                    selection.windowId,
                    selection.number,
                    15
                );
                setRelations(data);
            } catch (error) {
                console.error("Failed to fetch number relations:", error);
                setRelations(null);
            } finally {
                setLoading(false);
            }
        }

        loadRelations();
    }, [selection]);

    return (
        <main className="page relations-page">
            <header className="topbar">
                <div>
                    <h1>KINO Single Number Relations</h1>
                    <p>Live relation analysis for the selected number and frame.</p>
                </div>
            </header>

            {!selection && (
                <section className="relation-empty">
                    <h2>No number selected yet</h2>
                    <p>Open the trend board and click a number.</p>
                </section>
            )}

            {selection && (
                <section className="relation-panel">
                    <h2>Selected number: {selection.number}</h2>

                    <div className="relation-meta">
                        <p>Window ID: {selection.windowId}</p>
                        <p>Frame: {selection.frameIndex + 1}</p>
                        <p>
                            Mode: {selection.windowSize} games / step {selection.stepSize}
                        </p>
                    </div>

                    {loading && <p>Loading relation data...</p>}

                    {!loading && relations && (
                        <>
                            <div className="anchor-summary">
                                <h3>Anchor appearances</h3>
                                <p>
                                    Total: <strong>{relations.anchor_appearances}</strong>
                                </p>
                                <p>
                                    Split:{" "}
                                    <strong>{relations.first_half_anchor_appearances}</strong>
                                    {" | "}
                                    <strong>{relations.second_half_anchor_appearances}</strong>
                                </p>
                            </div>

                            <div className="relations-list">
                                <h3>Top connected numbers</h3>

                                {relations.related_numbers.map((item) => (
                                    <div className="relation-row" key={item.number}>
                                        <div className="relation-number">{item.number}</div>

                                        <div className="relation-info">
                                            <strong>{item.total_count} connections</strong>
                                            <span>
                                                Split: {item.first_half_count} |{" "}
                                                {item.second_half_count}
                                            </span>
                                        </div>

                                        <div
                                            className={
                                                item.change > 0
                                                    ? "relation-change positive"
                                                    : item.change < 0
                                                        ? "relation-change negative"
                                                        : "relation-change neutral"
                                            }
                                        >
                                            {item.change > 0 ? `+${item.change}` : item.change}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                </section>
            )}
        </main>
    );
}

export default SingleNumberRelationsPage;