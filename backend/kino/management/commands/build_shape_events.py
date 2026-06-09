from django.core.management.base import BaseCommand

from kino.models import KinoDraw, KinoShapeEvent
from kino.services.shape_detector import detect_all_shapes, detect_shape


class Command(BaseCommand):
    help = "Build and save detected KINO shape events"

    def add_arguments(self, parser):
        parser.add_argument(
            "--shape",
            type=str,
            default="all",
            choices=[
                "all",
                "cross",
                "box_2x2",
                "l_shape",
                "vertical_4",
                "horizontal_4",
                "diagonal_down_4",
                "diagonal_up_4",
            ],
        )

        parser.add_argument(
            "--min-hits",
            type=int,
            default=None,
            help="Minimum hits required. If omitted, detector defaults are used."
        )

        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Delete existing shape events before rebuilding"
        )

    def handle(self, *args, **options):
        shape = options["shape"]
        min_hits = options["min_hits"]
        rebuild = options["rebuild"]

        if rebuild:
            if shape == "all":
                deleted_count, _ = KinoShapeEvent.objects.all().delete()
            else:
                deleted_count, _ = KinoShapeEvent.objects.filter(shape=shape).delete()

            self.stdout.write(
                self.style.WARNING(f"Deleted {deleted_count} existing shape events.")
            )

        draws = list(KinoDraw.objects.order_by("draw_time"))

        created_count = 0
        updated_count = 0

        for draw in draws:
            if shape == "all":
                events = detect_all_shapes(draw.numbers)
            else:
                events = detect_shape(
                    draw_numbers=draw.numbers,
                    shape_name=shape,
                    min_hits=min_hits,
                )

            for event in events:
                _, created = KinoShapeEvent.objects.update_or_create(
                    draw=draw,
                    shape=event["shape"],
                    center_number=event["center_number"],
                    hit_count=event["hit_count"],
                    defaults={
                        "center_row": event["center_row"],
                        "center_col": event["center_col"],
                        "shape_numbers": event["shape_numbers"],
                        "hit_numbers": event["hit_numbers"],
                        "shape_size": event["shape_size"],
                    }
                )

                if created:
                    created_count += 1
                else:
                    updated_count += 1

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Shape events build finished"))
        self.stdout.write(f"Created: {created_count}")
        self.stdout.write(f"Updated: {updated_count}")
        self.stdout.write(f"Total stored: {KinoShapeEvent.objects.count()}")