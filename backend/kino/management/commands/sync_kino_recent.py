import time
import requests
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from kino.models import KinoDraw


GAME_ID = 1100
PAGE_SIZE = 200


def fetch_kino_draws_page(day: str, page: int = 0):
    url = f"https://api.opap.gr/draws/v3.0/{GAME_ID}/draw-date/{day}/{day}"

    response = requests.get(
        url,
        params={
            "page": page,
            "size": PAGE_SIZE,
        },
        timeout=15,
    )

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
        time.sleep(0.1)

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


class Command(BaseCommand):
    help = "Sync recent KINO draws and keep only a rolling number of days"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=10,
            help="How many recent days to keep"
        )

    def handle(self, *args, **options):
        days = options["days"]

        today = timezone.localdate()
        start_day = today - timedelta(days=days - 1)

        total_created = 0
        total_updated = 0

        self.stdout.write(
            self.style.WARNING(
                f"Syncing KINO draws from {start_day} to {today}"
            )
        )

        current_day = start_day

        while current_day <= today:
            day_str = current_day.isoformat()
            self.stdout.write(f"Fetching {day_str}...")

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

            current_day += timedelta(days=1)
            time.sleep(0.2)

        cutoff_day = today - timedelta(days=days)
        cutoff_timestamp = int(
            timezone.datetime.combine(
                cutoff_day,
                timezone.datetime.min.time()
            ).timestamp() * 1000
        )

        deleted_count, _ = KinoDraw.objects.filter(
            draw_time__lt=cutoff_timestamp
        ).delete()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Recent sync finished."))
        self.stdout.write(f"Total created: {total_created}")
        self.stdout.write(f"Total updated: {total_updated}")
        self.stdout.write(f"Deleted old draws: {deleted_count}")
        self.stdout.write(f"Current draw count: {KinoDraw.objects.count()}")