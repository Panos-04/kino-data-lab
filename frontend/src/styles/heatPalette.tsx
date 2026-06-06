export const HEAT_PALETTE: Record<
  number,
  { top: string; bottom: string; text: string }
> = {
  0: { top: "#0b2468", bottom: "#07184c", text: "#7fa7df" },
  1: { top: "#12337a", bottom: "#0b2460", text: "#a9c6ff" },

  2: { top: "#fff97a", bottom: "#f4eb1a", text: "#10204f" },
  3: { top: "#fff15e", bottom: "#eedc00", text: "#10204f" },

  4: { top: "#ffd95c", bottom: "#ffca28", text: "#10204f" },
  5: { top: "#ffc34d", bottom: "#ffb300", text: "#10204f" },
  6: { top: "#ffb347", bottom: "#ff9800", text: "#10204f" },

  7: { top: "#ff9a4d", bottom: "#ff7f11", text: "#ffffff" },
  8: { top: "#ff875f", bottom: "#ff7043", text: "#ffffff" },
  9: { top: "#ff7468", bottom: "#f4511e", text: "#ffffff" },

  10: { top: "#ff625d", bottom: "#e53935", text: "#ffffff" },
  11: { top: "#ff6b6b", bottom: "#c62828", text: "#ffffff" },
};

export function getBallStyle(count: number, maxHeat = 11) {
  const clamped = Math.max(0, Math.min(count, maxHeat));
  const palette = HEAT_PALETTE[clamped];

  return {
    background: `linear-gradient(180deg, ${palette.top} 0%, ${palette.bottom} 100%)`,
    color: palette.text,
    boxShadow:
      clamped >= 10
        ? "0 0 16px rgba(255, 90, 90, 0.35)"
        : clamped >= 7
        ? "0 0 12px rgba(255, 140, 70, 0.22)"
        : clamped >= 4
        ? "0 0 10px rgba(255, 200, 50, 0.14)"
        : "none",
  };
}