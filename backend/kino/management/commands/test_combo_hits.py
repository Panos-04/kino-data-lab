from collections import Counter

from django.core.management.base import BaseCommand

from kino.models import KinoDraw, KinoWindowAnalysis


class Command(BaseCommand):
    help = "Backtest combo hit rates: cold/hot/middle numbers against future draws"

    def add_arguments(self, parser):
        parser.add_argument("--window", type=int, default=20)
        parser.add_argument("--step", type=int, default=10)
        parser.add_argument("--pick", type=int, default=5)
        parser.add_argument("--future", type=int, default=1)
        parser.add_argument(
            "--strategy",
            type=str,
            default="cold",
            choices=["cold", "hot", "middle"],
        )

    def handle(self, *args, **options):
        window_size = options["window"]
        step_size = options["step"]
        pick_count = options["pick"]
        future_size = options["future"]
        strategy = options["strategy"]

        windows = (
            KinoWindowAnalysis.objects
            .filter(window_size=window_size, step_size=step_size)
            .prefetch_related("numbers")
            .order_by("start_time")
        )

        hit_distribution = Counter()
        tested_draws = 0
        skipped_windows = 0

        best_results = []

        for window in windows:
            numbers = list(window.numbers.all())
            expected = window.window_size * 0.25

            if strategy == "cold":
                selected = sorted(
                    numbers,
                    key=lambda item: (item.count, item.number)
                )[:pick_count]

            elif strategy == "hot":
                selected = sorted(
                    numbers,
                    key=lambda item: (-item.count, item.number)
                )[:pick_count]

            else:
                selected = sorted(
                    numbers,
                    key=lambda item: (abs(item.count - expected), item.number)
                )[:pick_count]

            combo = [item.number for item in selected]

            future_draws = list(
                KinoDraw.objects
                .filter(draw_time__gt=window.end_time)
                .order_by("draw_time")[:future_size]
            )

            if len(future_draws) < future_size:
                skipped_windows += 1
                continue

            for draw in future_draws:
                hit_numbers = sorted(set(combo).intersection(draw.numbers))
                hit_count = len(hit_numbers)

                hit_distribution[hit_count] += 1
                tested_draws += 1

                if hit_count >= 4:
                    best_results.append({
                        "window_id": window.id,
                        "draw_id": draw.draw_id,
                        "combo": combo,
                        "draw_numbers": draw.numbers,
                        "hit_count": hit_count,
                        "hit_numbers": hit_numbers,
                    })

        if tested_draws == 0:
            self.stdout.write(self.style.WARNING("No tests completed."))
            return

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Combo hit test finished"))
        self.stdout.write(f"Strategy: {strategy}")
        self.stdout.write(f"Combo size: {pick_count}")
        self.stdout.write(f"Future draws per frame: {future_size}")
        self.stdout.write(f"Tested future draws: {tested_draws}")
        self.stdout.write(f"Skipped windows: {skipped_windows}")

        self.stdout.write("")
        self.stdout.write("Hit distribution:")
        for hits in range(0, pick_count + 1):
            count = hit_distribution[hits]
            percentage = (count / tested_draws) * 100

            self.stdout.write(
                f"{hits}/{pick_count}: {count} times ({percentage:.3f}%)"
            )

        four_plus = sum(
            hit_distribution[hits]
            for hits in range(4, pick_count + 1)
        )

        self.stdout.write("")
        self.stdout.write(
            f"4+/5 hits: {four_plus} times ({(four_plus / tested_draws) * 100:.3f}%)"
        )

        self.stdout.write("")
        self.stdout.write("Best 4+/5 results:")
        for result in best_results[:10]:
            self.stdout.write(str(result))