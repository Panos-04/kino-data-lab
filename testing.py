import requests
from datetime import date

GAME_ID = 1100

def fetch_kino_draws(day: str):
    url = f"https://api.opap.gr/draws/v3.0/{GAME_ID}/draw-date/{day}/{day}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()

def parse_draws(data):
    draws = []

    for item in data.get("content", []):
        draws.append({
            "draw_id": item.get("drawId"),
            "draw_time": item.get("drawTime"),
            "numbers": item.get("winningNumbers", {}).get("list", []),
        })

    return draws

if __name__ == "__main__":
    today = date.today().isoformat()
    data = fetch_kino_draws(today)
    draws = parse_draws(data)

    print(f"Found {len(draws)} draws")
    print(draws[:3])