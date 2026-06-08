from collections import Counter, defaultdict

from django.core.management.base import BaseCommand

from kino.models import KinoDraw
from kino.services.shape_detector import detect_shape, detect_all_shapes


class Command(BaseCommand):
    help = "Detect shape patterns in KINO draws"

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
            help="Minimum hits required inside shape. Defaults depend on shape."
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="How many examples to print."
        )

    def handle(self, *args, **options):
        shape = options["shape"]
        min_hits = options["min_hits"]
        limit = options["limit"]

        draws = list(KinoDraw.objects.order_by("draw_time"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        detected_events = []
        shape_counter = Counter()
        center_counter = Counter()

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
                event["draw_id"] = draw.draw_id
                event["draw_time"] = draw.draw_time

                detected_events.append(event)
                shape_counter[event["shape"]] += 1
                center_counter[(event["shape"], event["center_number"])] += 1

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Shape pattern test finished"))
        self.stdout.write(f"Total draws tested: {len(draws)}")
        self.stdout.write(f"Total shape events: {len(detected_events)}")

        self.stdout.write("")
        self.stdout.write("Shape counts:")
        for shape_name, count in shape_counter.most_common():
            self.stdout.write(
                f"{shape_name}: {count} events "
                f"({(count / len(draws)) * 100:.3f}% per draw-event)"
            )

        self.stdout.write("")
        self.stdout.write("Most common shape centers:")
        for (shape_name, center_number), count in center_counter.most_common(20):
            self.stdout.write(
                f"{shape_name} centered at {center_number}: {count} times"
            )

        self.stdout.write("")
        self.stdout.write("Example events:")
        for event in detected_events[:limit]:
            self.stdout.write(str(event))