import requests
from django.core.management.base import BaseCommand
from kino.models import KinoDraw

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


class Command(BaseCommand):
    help = "Import KINO draws for one specific day"

    def add_arguments(self, parser):
        parser.add_argument("day", type=str, help="Date format: YYYY-MM-DD")

    def handle(self, *args, **options):
        day = options["day"]

        self.stdout.write(f"Fetching KINO draws for {day}...")

        data = fetch_kino_draws(day)
        draws = parse_draws(data)

        created_count = 0
        updated_count = 0

        for draw in draws:
            _, created = KinoDraw.objects.update_or_create(
                draw_id=draw["draw_id"],
                defaults={
                    "draw_time": draw["draw_time"],
                    "numbers": draw["numbers"],
                }
            )

            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Created: {created_count}, Updated: {updated_count}"
            )
        )