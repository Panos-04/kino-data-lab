from collections import Counter

from django.core.management.base import BaseCommand
from kino.models import KinoDraw


def build_windows(draws, window_size, step_size):
    windows = []

    for start_index in range(0, len(draws) - window_size + 1, step_size):
        end_index = start_index + window_size
        window_draws = draws[start_index:end_index]

        counter = Counter()

        for draw in window_draws:
            counter.update(draw.numbers)

        numbers = []

        for number in range(1, 81):
            numbers.append({
                "number": number,
                "count": counter.get(number, 0),
                "percentage": round((counter.get(number, 0) / window_size) * 100, 2),
            })

        windows.append({
            "window_number": len(windows) + 1,
            "start_draw_id": window_draws[0].draw_id,
            "end_draw_id": window_draws[-1].draw_id,
            "start_time": window_draws[0].draw_time,
            "end_time": window_draws[-1].draw_time,
            "window_size": window_size,
            "step_size": step_size,
            "numbers": numbers,
            "top_numbers": sorted(
                numbers,
                key=lambda item: item["count"],
                reverse=True
            )[:10],
        })

    return windows


class Command(BaseCommand):
    help = "Analyze KINO sliding windows"

    def add_arguments(self, parser):
        parser.add_argument(
            "--window",
            type=int,
            default=20,
            help="Number of draws per window"
        )

        parser.add_argument(
            "--step",
            type=int,
            default=10,
            help="How many draws to move forward per window"
        )
        parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many windows to print. Use 0 for all."
        )

    def handle(self, *args, **options):
        window_size = options["window"]
        step_size = options["step"]

        draws = list(KinoDraw.objects.order_by("draw_time"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found. Import data first."))
            return

        if len(draws) < window_size:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Found {len(draws)}, need at least {window_size}."
                )
            )
            return

        windows = build_windows(draws, window_size, step_size)
        limit = options["limit"]
        windows_to_print = windows if limit == 0 else windows[:limit]

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Created {len(windows)} windows using window={window_size}, step={step_size}"
            )
        )

        self.stdout.write("")

        for window in windows_to_print:
            self.stdout.write(
                f"Window {window['window_number']}: "
                f"{window['start_draw_id']} → {window['end_draw_id']}"
            )

            top_text = ", ".join(
                [
                    f"{item['number']}({item['count']})"
                    for item in window["top_numbers"][:5]
                ]
            )

            self.stdout.write(f"Top 5: {top_text}")
            self.stdout.write("")