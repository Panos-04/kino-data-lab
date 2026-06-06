import { useEffect, useState } from "react";
import {
    getFrameSelection,
    getFrameStorageKey,
    type FrameSelection,
} from "../utils/frameSelection";
import { fetchGeneralRelations } from "../api/kino";
import type { GeneralRelations, GeneralRelationAnchor } from "../api/kino";

function getAnchorLabel(type: string) {
    if (type === "hot") return "Hot";
    if (type === "cold") return "Cold";
    return "Middle";
}

function GeneralRelationsPage() {
    const [selection, setSelection] = useState<FrameSelection | null>(() =>
        getFrameSelection()
    );
    const [relations, setRelations] = useState<GeneralRelations | null>(null);
    const [selectedAnchor, setSelectedAnchor] =
        useState<GeneralRelationAnchor | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        function refreshSelection() {
            setSelection(getFrameSelection());
        }

        function handleStorage(event: StorageEvent) {
            if (event.key !== getFrameStorageKey()) return;
            refreshSelection();
        }

        window.addEventListener("storage", handleStorage);
        window.addEventListener("kino-frame-selection", refreshSelection);

        return () => {
            window.removeEventListener("storage", handleStorage);
            window.removeEventListener("kino-frame-selection", refreshSelection);
        };
    }, []);

    useEffect(() => {
        async function loadGeneralRelations() {
            if (!selection) return;

            setLoading(true);

            try {
                const data = await fetchGeneralRelations(selection.windowId, 20, 20);
                setRelations(data);
                setSelectedAnchor(data.anchors[0] ?? null);
            } catch (error) {
                console.error("Failed to fetch general relations:", error);
                setRelations(null);
                setSelectedAnchor(null);
            } finally {
                setLoading(false);
            }
        }

        loadGeneralRelations();
    }, [selection]);

    return (
        <main className="page relations-page">
            <header className="topbar">
                <div>
                    <h1>KINO General Relations</h1>
                    <p>
                        Auto-selected hot, cold, and middle numbers for the current frame.
                    </p>
                </div>
            </header>

            {!selection && (
                <section className="relation-empty">
                    <h2>No frame selected yet</h2>
                    <p>Open the trend board and move to a frame.</p>
                </section>
            )}

            {selection && (
                <section className="relation-panel">
                    <h2>Frame {selection.frameIndex + 1}</h2>

                    <div className="relation-meta">
                        <p>Window ID: {selection.windowId}</p>
                        <p>
                            Mode: {selection.windowSize} games / step {selection.stepSize}
                        </p>
                    </div>

                    {loading && <p>Loading general relations...</p>}

                    {!loading && relations && (
                        <div className="general-relations-layout">
                            <aside className="anchor-list">
                                <h3>Selected anchors</h3>

                                {relations.anchors.map((anchor) => (
                                    <button
                                        type="button"
                                        key={anchor.anchor_number}
                                        className={`anchor-card ${selectedAnchor?.anchor_number === anchor.anchor_number
                                            ? "active"
                                            : ""
                                            } ${anchor.anchor_type}`}
                                        onClick={() => setSelectedAnchor(anchor)}
                                    >
                                        <span className="anchor-number">
                                            {anchor.anchor_number}
                                        </span>
                                        <span>{getAnchorLabel(anchor.anchor_type)}</span>
                                        <small>Heat {anchor.anchor_heat}</small>
                                    </button>
                                ))}
                            </aside>

                            {selectedAnchor && (
                                <section className="anchor-detail">
                                    <header className="anchor-detail-header">
                                        <div>
                                            <h3>Number {selectedAnchor.anchor_number}</h3>
                                            <p>
                                                {getAnchorLabel(selectedAnchor.anchor_type)} anchor ·
                                                heat {selectedAnchor.anchor_heat}
                                            </p>
                                        </div>

                                        <div className="anchor-summary-mini">
                                            <strong>{selectedAnchor.anchor_appearances}</strong>
                                            <span>
                                                Split{" "}
                                                {selectedAnchor.first_half_anchor_appearances} |{" "}
                                                {selectedAnchor.second_half_anchor_appearances}
                                            </span>
                                        </div>
                                    </header>

                                    <div className="connection-columns">
                                        <div>
                                            <h4>20 strongest connections</h4>

                                            {selectedAnchor.strongest_connections.map((item) => (
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

                                        <div>
                                            <h4>20 weakest connections</h4>

                                            {selectedAnchor.weakest_connections.map((item) => (
                                                <div className="relation-row" key={item.number}>
                                                    <div className="relation-number weak">
                                                        {item.number}
                                                    </div>

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
                                    </div>
                                </section>
                            )}
                        </div>
                    )}
                </section>
            )}
        </main>
    );
}

export default GeneralRelationsPage;