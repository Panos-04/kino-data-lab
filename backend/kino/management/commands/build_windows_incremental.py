from collections import Counter

from django.core.management.base import BaseCommand
from django.db import transaction

from kino.models import KinoDraw, KinoWindowAnalysis, KinoWindowNumber


class Command(BaseCommand):
    help = "Incrementally build missing KINO window analysis rows"

    def add_arguments(self, parser):
        parser.add_argument("--window", type=int, default=20)
        parser.add_argument("--step", type=int, default=10)

    def handle(self, *args, **options):
        window_size = options["window"]
        step_size = options["step"]

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if len(draws) < window_size:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Need {window_size}, have {len(draws)}."
                )
            )
            return

        existing_starts = set(
            KinoWindowAnalysis.objects.filter(
                window_size=window_size,
                step_size=step_size,
            ).values_list("start_draw__draw_id", flat=True)
        )

        created_windows = 0
        updated_windows = 0

        self.stdout.write("")
        self.stdout.write(
            self.style.WARNING(
                f"Building missing windows: window={window_size}, step={step_size}"
            )
        )

        for start_index in range(0, len(draws) - window_size + 1, step_size):
            window_draws = draws[start_index:start_index + window_size]

            start_draw = window_draws[0]
            end_draw = window_draws[-1]

            if start_draw.draw_id in existing_starts:
                continue

            counter = Counter()

            for draw in window_draws:
                counter.update(draw.numbers)

            with transaction.atomic():
                analysis, created = KinoWindowAnalysis.objects.update_or_create(
                    window_size=window_size,
                    step_size=step_size,
                    start_draw=start_draw,
                    defaults={
                        "end_draw": end_draw,
                        "start_time": start_draw.draw_time,
                        "end_time": end_draw.draw_time,
                    },
                )

                KinoWindowNumber.objects.filter(analysis=analysis).delete()

                rows = []

                for number in range(1, 81):
                    count = counter.get(number, 0)
                    percentage = round((count / window_size) * 100, 2)

                    rows.append(
                        KinoWindowNumber(
                            analysis=analysis,
                            number=number,
                            count=count,
                            percentage=percentage,
                        )
                    )

                KinoWindowNumber.objects.bulk_create(rows, batch_size=1000)

            if created:
                created_windows += 1
            else:
                updated_windows += 1

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Incremental window build finished."))
        self.stdout.write(f"Created windows: {created_windows}")
        self.stdout.write(f"Updated windows: {updated_windows}")
        self.stdout.write(
            f"Total windows now: {KinoWindowAnalysis.objects.filter(window_size=window_size, step_size=step_size).count()}"
        )