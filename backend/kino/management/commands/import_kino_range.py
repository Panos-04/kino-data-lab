import time
import requests
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from kino.models import KinoDraw


GAME_ID = 1100
PAGE_SIZE = 200


def fetch_kino_draws_page(day: str, page: int = 0):
    url = f"https://api.opap.gr/draws/v3.0/{GAME_ID}/draw-date/{day}/{day}"

    params = {
        "page": page,
        "size": PAGE_SIZE,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_all_kino_draws_for_day(day: str):
    all_draws = []
    page = 0

    while True:
        data = fetch_kino_draws_page(day, page)
        content = data.get("content", [])

        all_draws.extend(content)

        total_pages = data.get("totalPages", 1)
        current_page = data.get("number", page)

        if current_page + 1 >= total_pages:
            break

        page += 1
        time.sleep(0.15)

    return all_draws


def parse_draws(items):
    draws = []

    for item in items:
        draws.append({
            "draw_id": item.get("drawId"),
            "draw_time": item.get("drawTime"),
            "numbers": item.get("winningNumbers", {}).get("list", []),
        })

    return draws


def date_range(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


class Command(BaseCommand):
    help = "Import KINO draws for a date range"

    def add_arguments(self, parser):
        parser.add_argument("start_date", type=str, help="Start date: YYYY-MM-DD")
        parser.add_argument("end_date", type=str, help="End date: YYYY-MM-DD")

    def handle(self, *args, **options):
        try:
            start_date = datetime.strptime(options["start_date"], "%Y-%m-%d").date()
            end_date = datetime.strptime(options["end_date"], "%Y-%m-%d").date()
        except ValueError:
            raise CommandError("Dates must be in YYYY-MM-DD format.")

        if start_date > end_date:
            raise CommandError("start_date cannot be after end_date.")

        total_created = 0
        total_updated = 0
        total_failed = 0

        for day in date_range(start_date, end_date):
            day_str = day.isoformat()
            self.stdout.write(f"Fetching {day_str}...")

            try:
                raw_draws = fetch_all_kino_draws_for_day(day_str)
                draws = parse_draws(raw_draws)

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

                total_created += created_count
                total_updated += updated_count

                self.stdout.write(
                    self.style.SUCCESS(
                        f"{day_str}: Found {len(draws)}, Created {created_count}, Updated {updated_count}"
                    )
                )

                time.sleep(0.25)

            except requests.RequestException as e:
                total_failed += 1
                self.stdout.write(
                    self.style.ERROR(f"{day_str}: Failed API request - {e}")
                )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Import finished."))
        self.stdout.write(f"Total created: {total_created}")
        self.stdout.write(f"Total updated: {total_updated}")
        self.stdout.write(f"Failed days: {total_failed}")