import time
import requests
from datetime import datetime, timedelta, timezone as dt_timezone

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
        timeout=20,
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


def parse_draw(item):
    return {
        "draw_id": item.get("drawId"),
        "draw_time": item.get("drawTime"),
        "numbers": item.get("winningNumbers", {}).get("list", []),
    }


class Command(BaseCommand):
    help = "Append only: sync missing KINO draws from the latest stored date to today"

    def add_arguments(self, parser):
        parser.add_argument(
            "--lookback-days",
            type=int,
            default=2,
            help="Fetch a small safety window before today to avoid missing late draws.",
        )

    def handle(self, *args, **options):
        lookback_days = options["lookback_days"]

        latest_draw = KinoDraw.objects.order_by("-draw_time").first()
        today = timezone.localdate()

        if latest_draw:
            latest_date = datetime.fromtimestamp(
                latest_draw.draw_time / 1000,
                tz=dt_timezone.utc,
            ).date()

            start_day = min(
                latest_date,
                today - timedelta(days=lookback_days - 1),
            )
        else:
            start_day = today - timedelta(days=lookback_days - 1)

        total_created = 0
        total_updated = 0
        total_found = 0

        self.stdout.write("")
        self.stdout.write(
            self.style.WARNING(
                f"Syncing KINO draws append-only from {start_day} to {today}"
            )
        )

        current_day = start_day

        while current_day <= today:
            day_str = current_day.isoformat()
            self.stdout.write(f"Fetching {day_str}...")

            raw_items = fetch_all_kino_draws_for_day(day_str)
            total_found += len(raw_items)

            created_count = 0
            updated_count = 0

            for item in raw_items:
                draw = parse_draw(item)

                if not draw["draw_id"] or not draw["numbers"]:
                    continue

                _, created = KinoDraw.objects.update_or_create(
                    draw_id=draw["draw_id"],
                    defaults={
                        "draw_time": draw["draw_time"],
                        "numbers": draw["numbers"],
                    },
                )

                if created:
                    created_count += 1
                else:
                    updated_count += 1

            total_created += created_count
            total_updated += updated_count

            self.stdout.write(
                self.style.SUCCESS(
                    f"{day_str}: Found {len(raw_items)}, Created {created_count}, Updated {updated_count}"
                )
            )

            current_day += timedelta(days=1)
            time.sleep(0.2)

        latest_after = KinoDraw.objects.order_by("-draw_id").first()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Latest sync finished."))
        self.stdout.write(f"Total found: {total_found}")
        self.stdout.write(f"Total created: {total_created}")
        self.stdout.write(f"Total updated: {total_updated}")
        self.stdout.write(f"Total draws stored: {KinoDraw.objects.count()}")

        if latest_after:
            self.stdout.write(f"Latest draw ID: {latest_after.draw_id}")