from collections import Counter

from django.core.management.base import BaseCommand

from kino.models import KinoDraw, KinoWindowAnalysis


class Command(BaseCommand):
    help = "Backtest a simple combo theory: 5 coldest numbers from base window into next N games"

    def add_arguments(self, parser):
        parser.add_argument("--window", type=int, default=20)
        parser.add_argument("--step", type=int, default=10)
        parser.add_argument("--pick", type=int, default=5)
        parser.add_argument("--future", type=int, default=10)

    def handle(self, *args, **options):
        window_size = options["window"]
        step_size = options["step"]
        pick_count = options["pick"]
        future_size = options["future"]

        windows = (
            KinoWindowAnalysis.objects
            .filter(window_size=window_size, step_size=step_size)
            .prefetch_related("numbers")
            .order_by("start_time")
        )

        tested = 0
        skipped = 0

        total_hits_sum = 0
        unique_hits_sum = 0

        hit_distribution = Counter()

        best_result = None
        worst_result = None

        for window in windows:
            cold_numbers = sorted(
                list(window.numbers.all()),
                key=lambda item: (item.count, item.number)
            )[:pick_count]

            picked_numbers = [item.number for item in cold_numbers]

            future_draws = list(
                KinoDraw.objects
                .filter(draw_time__gt=window.end_time)
                .order_by("draw_time")[:future_size]
            )

            if len(future_draws) < future_size:
                skipped += 1
                continue

            future_counter = Counter()

            for draw in future_draws:
                future_counter.update(draw.numbers)

            hits = {
                number: future_counter.get(number, 0)
                for number in picked_numbers
            }

            total_hits = sum(hits.values())
            unique_hits = sum(1 for value in hits.values() if value > 0)

            tested += 1
            total_hits_sum += total_hits
            unique_hits_sum += unique_hits
            hit_distribution[unique_hits] += 1

            result = {
                "window_id": window.id,
                "start_draw": window.start_draw.draw_id,
                "end_draw": window.end_draw.draw_id,
                "picked_numbers": picked_numbers,
                "hits": hits,
                "total_hits": total_hits,
                "unique_hits": unique_hits,
            }

            if best_result is None or total_hits > best_result["total_hits"]:
                best_result = result

            if worst_result is None or total_hits < worst_result["total_hits"]:
                worst_result = result

        if tested == 0:
            self.stdout.write(self.style.WARNING("No completed tests found."))
            return

        avg_total_hits = total_hits_sum / tested
        avg_unique_hits = unique_hits_sum / tested

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Cold combo test finished"))
        self.stdout.write(f"Tested frames: {tested}")
        self.stdout.write(f"Skipped incomplete future frames: {skipped}")
        self.stdout.write(f"Average total hits in next {future_size}: {avg_total_hits:.2f}")
        self.stdout.write(f"Average unique picked numbers hit: {avg_unique_hits:.2f} / {pick_count}")

        self.stdout.write("")
        self.stdout.write("Unique-hit distribution:")
        for unique_hits, count in sorted(hit_distribution.items()):
            percentage = (count / tested) * 100
            self.stdout.write(
                f"{unique_hits}/{pick_count} numbers appeared: {count} times ({percentage:.2f}%)"
            )

        self.stdout.write("")
        self.stdout.write("Best result:")
        self.stdout.write(str(best_result))

        self.stdout.write("")
        self.stdout.write("Worst result:")
        self.stdout.write(str(worst_result))