type ModeSwitchProps = {
    windowSize: number;
    onChangeMode: (windowSize: number, stepSize: number) => void;
};

function ModeSwitch({ windowSize, onChangeMode }: ModeSwitchProps) {
    return (
        <div className="mode-switch">
            <button
                className={windowSize === 20 ? "active" : ""}
                onClick={() => onChangeMode(20, 10)}
            >
                20 games / move 10
            </button>

            <button
                className={windowSize === 10 ? "active" : ""}
                onClick={() => onChangeMode(10, 5)}
            >
                10 games / move 5
            </button>
        </div>
    );
}

export default ModeSwitch;