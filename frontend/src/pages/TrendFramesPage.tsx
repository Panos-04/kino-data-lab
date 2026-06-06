import { useEffect, useState } from "react";
import { fetchWindows } from "../api/kino";
import type { WindowAnalysis } from "../api/kino";

import ModeSwitch from "../components/ModeSwitch";
import FrameControls from "../components/FrameControls";
import HeatLegend from "../components/HeatLegend";
import KinoBoard from "../components/KinoBoard";

function TrendFramesPage() {
    const [windows, setWindows] = useState<WindowAnalysis[]>([]);
    const [windowSize, setWindowSize] = useState(20);
    const [stepSize, setStepSize] = useState(10);
    const [loading, setLoading] = useState(false);
    const [currentIndex, setCurrentIndex] = useState(0);

    useEffect(() => {
        async function loadWindows() {
            setLoading(true);

            try {
                const data = await fetchWindows(windowSize, stepSize, 500, 10);
                setWindows(data);
                setCurrentIndex(0);
            } catch (error) {
                console.error("Failed to fetch windows:", error);
            } finally {
                setLoading(false);
            }
        }

        loadWindows();
    }, [windowSize, stepSize]);

    const currentWindow = windows[currentIndex];

    function handleChangeMode(newWindowSize: number, newStepSize: number) {
        setWindowSize(newWindowSize);
        setStepSize(newStepSize);
    }

    function goPrevious() {
        setCurrentIndex((prev) => Math.max(prev - 1, 0));
    }

    function goNext() {
        setCurrentIndex((prev) => Math.min(prev + 1, windows.length - 1));
    }

    return (
        <main className="page">
            <header className="topbar">
                <div>
                    <h1>KINO Trend Frames</h1>
                    <p>
                    Frame-by-frame heat tracking for KINO number trends.
                    </p>
                </div>

                <ModeSwitch windowSize={windowSize} onChangeMode={handleChangeMode} />
            </header>

            {loading && <p className="status-text">Loading frames...</p>}

            {!loading && currentWindow && (
                <>
                    <FrameControls
                        currentIndex={currentIndex}
                        totalFrames={windows.length}
                        startDrawId={currentWindow.start_draw_id}
                        endDrawId={currentWindow.end_draw_id}
                        windowSize={currentWindow.window_size}
                        stepSize={currentWindow.step_size}
                        onPrevious={goPrevious}
                        onNext={goNext}
                    />

                    <HeatLegend />

                    <section className="boards-grid">
                        <KinoBoard
                            title={`${currentWindow.window_size}-game base frame`}
                            subtitle={`Draws ${currentWindow.start_draw_id} → ${currentWindow.end_draw_id}`}
                            numbersData={currentWindow.numbers}
                            maxHeat={11}
                            showSplitOnClick={true}
                        />

                    </section>
                </>
            )}

            {!loading && !currentWindow && (
                <p className="status-text">No frame data found.</p>
            )}
        </main>
    );
}

export default TrendFramesPage;