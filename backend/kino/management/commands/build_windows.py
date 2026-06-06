from collections import Counter

from django.core.management.base import BaseCommand
from django.db import transaction

from kino.models import KinoDraw, KinoWindowAnalysis, KinoWindowNumber


class Command(BaseCommand):
    help = "Build and save KINO sliding-window heatmap analysis"

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
            "--rebuild",
            action="store_true",
            help="Delete existing windows with this window/step and rebuild"
        )

    def handle(self, *args, **options):
        window_size = options["window"]
        step_size = options["step"]
        rebuild = options["rebuild"]

        draws = list(KinoDraw.objects.order_by("draw_time"))

        if len(draws) < window_size:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Found {len(draws)}, need {window_size}."
                )
            )
            return

        if rebuild:
            deleted_count, _ = KinoWindowAnalysis.objects.filter(
                window_size=window_size,
                step_size=step_size,
            ).delete()

            self.stdout.write(
                self.style.WARNING(
                    f"Deleted {deleted_count} existing window rows."
                )
            )

        created_windows = 0
        updated_windows = 0

        for start_index in range(0, len(draws) - window_size + 1, step_size):
            end_index = start_index + window_size
            window_draws = draws[start_index:end_index]

            start_draw = window_draws[0]
            end_draw = window_draws[-1]

            counter = Counter()

            for draw in window_draws:
                counter.update(draw.numbers)

            with transaction.atomic():
                analysis, created = KinoWindowAnalysis.objects.update_or_create(
                    window_size=window_size,
                    step_size=step_size,
                    start_draw=start_draw,
                    end_draw=end_draw,
                    defaults={
                        "start_time": start_draw.draw_time,
                        "end_time": end_draw.draw_time,
                    }
                )

                KinoWindowNumber.objects.filter(analysis=analysis).delete()

                number_rows = []

                for number in range(1, 81):
                    count = counter.get(number, 0)
                    percentage = round((count / window_size) * 100, 2)

                    number_rows.append(
                        KinoWindowNumber(
                            analysis=analysis,
                            number=number,
                            count=count,
                            percentage=percentage,
                        )
                    )

                KinoWindowNumber.objects.bulk_create(number_rows)

            if created:
                created_windows += 1
            else:
                updated_windows += 1

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Finished building windows: window={window_size}, step={step_size}"
            )
        )
        self.stdout.write(f"Created windows: {created_windows}")
        self.stdout.write(f"Updated windows: {updated_windows}")