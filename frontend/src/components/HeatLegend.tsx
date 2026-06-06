import { HEAT_PALETTE } from "../styles/heatPalette";

const HEAT_LEGEND = [
    { value: 0, label: "0" },
    { value: 1, label: "1" },
    { value: 2, label: "2" },
    { value: 3, label: "3" },
    { value: 4, label: "4" },
    { value: 5, label: "5" },
    { value: 6, label: "6" },
    { value: 7, label: "7" },
    { value: 8, label: "8" },
    { value: 9, label: "9" },
    { value: 10, label: "10" },
    { value: 11, label: "11+" },
];

function HeatLegend() {
    return (
        <section className="legend">
            <h3 className="legend-title">Heat scale</h3>

            <div className="legend-scale">
                {HEAT_LEGEND.map((item) => {
                    const palette = HEAT_PALETTE[item.value];

                    return (
                        <span
                            key={item.label}
                            className="legend-box"
                            style={{
                                background: `linear-gradient(180deg, ${palette.top} 0%, ${palette.bottom} 100%)`,
                                color: palette.text,
                            }}
                        >
                            {item.label}
                        </span>
                    );
                })}
            </div>

            <p className="legend-note">
                Lower heat on the left, hotter trend on the right.
            </p>
        </section>
    );
}

export default HeatLegend;