type FrameControlsProps = {
    currentIndex: number;
    totalFrames: number;
    startDrawId?: number;
    endDrawId?: number;
    windowSize?: number;
    stepSize?: number;
    onPrevious: () => void;
    onNext: () => void;
};

function FrameControls({
    currentIndex,
    totalFrames,
    startDrawId,
    endDrawId,
    windowSize,
    stepSize,
    onPrevious,
    onNext,
}: FrameControlsProps) {
    return (
        <section className="frame-controls">
            <button onClick={onPrevious} disabled={currentIndex === 0}>
                ← Previous
            </button>

            <div className="frame-meta">
                <h2>
                    Frame {currentIndex + 1} / {totalFrames}
                </h2>

                {startDrawId && endDrawId && (
                    <p>
                        Draws {startDrawId} → {endDrawId}
                    </p>
                )}

                {windowSize && stepSize && (
                    <p>
                        Window: {windowSize} games | Step: {stepSize}
                    </p>
                )}
            </div>

            <button onClick={onNext} disabled={currentIndex === totalFrames - 1}>
                Next →
            </button>
        </section>
    );
}

export default FrameControls;