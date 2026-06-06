import { useMemo, useState } from "react";
import KinoBall from "./KinoBall";
import type { WindowNumber } from "../api/kino";

type KinoBoardProps = {
    title: string;
    subtitle: string;
    numbersData: WindowNumber[];
    maxHeat?: number;
    showSplitOnClick?: boolean;
    onNumberClick?: (number: number) => void;
};

function KinoBoard({
    title,
    subtitle,
    numbersData,
    maxHeat = 11,
    showSplitOnClick = false,
    onNumberClick,
  }: KinoBoardProps) {
    const [selectedNumber, setSelectedNumber] = useState<number | null>(null);

    const numberMap = useMemo(() => {
        const map = new Map<number, WindowNumber>();

        numbersData.forEach((item) => {
            map.set(item.number, item);
        });

        return map;
    }, [numbersData]);

    const numbers = Array.from({ length: 80 }, (_, index) => index + 1);

    function handleBallClick(number: number) {
        if (showSplitOnClick) {
            setSelectedNumber((current) => {
                if (current === number) return null;
                return number;
            });
        }

        onNumberClick?.(number);
    }

    return (
        <section className="board-panel">
            <div className="board-panel-header">
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>

            <div className="board-wrapper">
                <div className="board">
                    {numbers.map((num) => (
                        <KinoBall
                            key={num}
                            number={num}
                            data={numberMap.get(num)}
                            maxHeat={maxHeat}
                            showSplitOnClick={showSplitOnClick}
                            isSelected={selectedNumber === num}
                            onClick={handleBallClick}
                        />
                    ))}
                </div>
            </div>
        </section>
    );
}

export default KinoBoard;