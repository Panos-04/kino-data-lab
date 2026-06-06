import { getBallStyle } from "../styles/heatPalette";
import type { WindowNumber } from "../api/kino";

type KinoBallProps = {
    number: number;
    data?: WindowNumber;
    maxHeat?: number;
    showSplitOnClick?: boolean;
    isSelected?: boolean;
    onClick?: (number: number) => void;
};

function KinoBall({
    number,
    data,
    maxHeat = 11,
    showSplitOnClick = false,
    isSelected = false,
    onClick,
}: KinoBallProps) {
    const count = data?.count ?? 0;
    const firstHalf = data?.first_half_count ?? 0;
    const secondHalf = data?.second_half_count ?? 0;

    return (
        <button
            type="button"
            className={`ball ${isSelected ? "ball-selected" : ""}`}
            style={getBallStyle(count, maxHeat)}
            title={`Number ${number} | Total ${count} | Split ${firstHalf}-${secondHalf}`}
            onClick={() => onClick?.(number)}
        >
            <div className="ball-number">{number}</div>

            {showSplitOnClick && isSelected ? (
                <div className="ball-split">
                    <span>{firstHalf}</span>
                    <span className="split-divider">|</span>
                    <span>{secondHalf}</span>
                </div>
            ) : (
                <div className="ball-heat">{count}</div>
            )}
        </button>
    );
}

export default KinoBall;