from collections import Counter

from django.core.management.base import BaseCommand
from django.db import transaction

from kino.models import KinoDraw, KinoBoardPatternEvent
from kino.services.board_pattern_detector import detect_board_patterns


class Command(BaseCommand):
    help = "Build and save row/column board pattern events"

    def add_arguments(self, parser):
        parser.add_argument(
            "--row-threshold",
            type=int,
            default=6,
            help="Minimum hits inside a row of 10 numbers",
        )

        parser.add_argument(
            "--column-threshold",
            type=int,
            default=5,
            help="Minimum hits inside a column of 8 numbers",
        )

        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Delete existing board pattern events for these thresholds before rebuilding",
        )

    def handle(self, *args, **options):
        row_threshold = options["row_threshold"]
        column_threshold = options["column_threshold"]
        rebuild = options["rebuild"]

        if rebuild:
            deleted_count, _ = KinoBoardPatternEvent.objects.filter(
                threshold__in=[row_threshold, column_threshold]
            ).delete()

            self.stdout.write(
                self.style.WARNING(
                    f"Deleted {deleted_count} existing board pattern events."
                )
            )

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        created_count = 0
        updated_count = 0
        row_event_count = 0
        column_event_count = 0

        pattern_counter = Counter()

        self.stdout.write("")
        self.stdout.write(
            self.style.WARNING(
                f"Building board pattern events: rows {row_threshold}+, columns {column_threshold}+"
            )
        )

        for draw in draws:
            events = detect_board_patterns(
                draw_numbers=draw.numbers,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
            )

            if not events:
                continue

            with transaction.atomic():
                for event in events:
                    _, created = KinoBoardPatternEvent.objects.update_or_create(
                        draw=draw,
                        pattern_type=event["pattern_type"],
                        group_number=event["group_number"],
                        threshold=event["threshold"],
                        defaults={
                            "group_numbers": event["group_numbers"],
                            "hit_numbers": event["hit_numbers"],
                            "hit_count": event["hit_count"],
                        },
                    )

                    if created:
                        created_count += 1
                    else:
                        updated_count += 1

                    if event["pattern_type"] == "row":
                        row_event_count += 1
                    elif event["pattern_type"] == "column":
                        column_event_count += 1

                    pattern_counter[
                        (
                            event["pattern_type"],
                            event["group_number"],
                        )
                    ] += 1

        total_stored = KinoBoardPatternEvent.objects.count()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Board pattern events build finished."))
        self.stdout.write(f"Draws tested: {len(draws)}")
        self.stdout.write(f"Created: {created_count}")
        self.stdout.write(f"Updated: {updated_count}")
        self.stdout.write(f"Row events seen in this build: {row_event_count}")
        self.stdout.write(f"Column events seen in this build: {column_event_count}")
        self.stdout.write(f"Total stored board pattern events: {total_stored}")

        self.stdout.write("")
        self.stdout.write("Most common row/column pattern groups:")

        for (pattern_type, group_number), count in pattern_counter.most_common(20):
            self.stdout.write(
                f"{pattern_type} {group_number}: {count} events"
            )