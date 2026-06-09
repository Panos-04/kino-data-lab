from collections import Counter, defaultdict

from django.core.management.base import BaseCommand

from kino.models import KinoDraw
from kino.services.shape_detector import detect_shape


class Command(BaseCommand):
    help = "Track movement vectors between repeated KINO shape patterns"

    def add_arguments(self, parser):
        parser.add_argument(
            "--shape",
            type=str,
            default="cross",
            choices=[
                "cross",
                "box_2x2",
                "l_shape",
                "vertical_4",
                "horizontal_4",
                "diagonal_down_4",
                "diagonal_up_4",
            ],
            help="Shape type to track",
        )

        parser.add_argument(
            "--min-hits",
            type=int,
            default=4,
            help="Minimum hits required inside shape",
        )

        parser.add_argument(
            "--future",
            type=int,
            default=10,
            help="How many future draws to search for the next same-shape event",
        )

        parser.add_argument(
            "--mode",
            type=str,
            default="one-to-one",
            choices=["all", "nearest", "best-overlap", "one-to-one"],
            help=(
                "Movement matching mode: "
                "all = connect to all future shape events; "
                "nearest = connect to all events in nearest future draw; "
                "best-overlap = connect to future events with best hit overlap; "
                "one-to-one = connect to one best future event"
            ),
        )

        parser.add_argument(
            "--exclude-same-center",
            action="store_true",
            help="Exclude movements where delta row and delta column are both 0",
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="How many example movements to print",
        )

    def handle(self, *args, **options):
        shape = options["shape"]
        min_hits = options["min_hits"]
        future_window = options["future"]
        mode = options["mode"]
        exclude_same_center = options["exclude_same_center"]
        limit = options["limit"]

        draws = list(KinoDraw.objects.order_by("draw_time"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        events_by_draw_index = defaultdict(list)
        all_events = []

        for draw_index, draw in enumerate(draws):
            events = detect_shape(
                draw_numbers=draw.numbers,
                shape_name=shape,
                min_hits=min_hits,
            )

            for event in events:
                event_data = {
                    **event,
                    "draw_index": draw_index,
                    "draw_id": draw.draw_id,
                    "draw_time": draw.draw_time,
                }

                events_by_draw_index[draw_index].append(event_data)
                all_events.append(event_data)

        movement_counter = Counter()
        gap_counter = Counter()
        center_counter = Counter()
        examples = []

        def overlap_score(first_event, second_event):
            first_hits = set(first_event["hit_numbers"])
            second_hits = set(second_event["hit_numbers"])
            return len(first_hits.intersection(second_hits))

        def distance_score(first_event, second_event):
            return (
                abs(second_event["center_row"] - first_event["center_row"])
                + abs(second_event["center_col"] - first_event["center_col"])
            )

        def get_future_events(current_event):
            current_index = current_event["draw_index"]
            future_events = []

            start_index = current_index + 1
            end_index = min(current_index + future_window + 1, len(draws))

            for future_index in range(start_index, end_index):
                future_events.extend(events_by_draw_index.get(future_index, []))

            return future_events

        def select_future_events(current_event, future_events):
            if not future_events:
                return []

            current_index = current_event["draw_index"]

            if mode == "all":
                return future_events

            if mode == "nearest":
                sorted_by_gap = sorted(
                    future_events,
                    key=lambda event: event["draw_index"] - current_index,
                )

                nearest_gap = sorted_by_gap[0]["draw_index"] - current_index

                return [
                    event
                    for event in sorted_by_gap
                    if event["draw_index"] - current_index == nearest_gap
                ]

            if mode == "best-overlap":
                best_score = max(
                    overlap_score(current_event, future_event)
                    for future_event in future_events
                )

                return [
                    future_event
                    for future_event in future_events
                    if overlap_score(current_event, future_event) == best_score
                ]

            if mode == "one-to-one":
                best_event = sorted(
                    future_events,
                    key=lambda future_event: (
                        future_event["draw_index"] - current_index,
                        -overlap_score(current_event, future_event),
                        distance_score(current_event, future_event),
                        future_event["center_number"],
                    ),
                )[0]

                return [best_event]

            return []

        for event in all_events:
            current_row = event["center_row"]
            current_col = event["center_col"]

            future_events = get_future_events(event)
            selected_future_events = select_future_events(event, future_events)

            for future_event in selected_future_events:
                delta_row = future_event["center_row"] - current_row
                delta_col = future_event["center_col"] - current_col
                gap = future_event["draw_index"] - event["draw_index"]

                if exclude_same_center and delta_row == 0 and delta_col == 0:
                    continue

                movement_key = (delta_row, delta_col)

                movement_counter[movement_key] += 1
                gap_counter[gap] += 1
                center_counter[
                    (
                        event["center_number"],
                        future_event["center_number"],
                    )
                ] += 1

                if len(examples) < limit:
                    examples.append(
                        {
                            "from_draw_id": event["draw_id"],
                            "to_draw_id": future_event["draw_id"],
                            "gap": gap,
                            "from_center": event["center_number"],
                            "to_center": future_event["center_number"],
                            "delta_row": delta_row,
                            "delta_col": delta_col,
                            "overlap_score": overlap_score(event, future_event),
                            "distance_score": distance_score(event, future_event),
                            "from_hits": event["hit_numbers"],
                            "to_hits": future_event["hit_numbers"],
                        }
                    )

        total_links = sum(movement_counter.values())

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Shape movement test finished"))
        self.stdout.write(f"Shape: {shape}")
        self.stdout.write(f"Min hits: {min_hits}")
        self.stdout.write(f"Future window: {future_window} games")
        self.stdout.write(f"Mode: {mode}")
        self.stdout.write(f"Exclude same center: {exclude_same_center}")
        self.stdout.write(f"Total draws tested: {len(draws)}")
        self.stdout.write(f"Total shape events: {len(all_events)}")
        self.stdout.write(f"Total movement links found: {total_links}")

        if total_links == 0:
            self.stdout.write(self.style.WARNING("No movement links found."))
            return

        self.stdout.write("")
        self.stdout.write("Most common movement vectors:")
        for (delta_row, delta_col), count in movement_counter.most_common(20):
            percentage = (count / total_links) * 100

            self.stdout.write(
                f"Δrow {delta_row:+}, Δcol {delta_col:+}: "
                f"{count} times ({percentage:.3f}%)"
            )

        self.stdout.write("")
        self.stdout.write("Most common gaps:")
        for gap, count in gap_counter.most_common(20):
            percentage = (count / total_links) * 100

            self.stdout.write(
                f"{gap} games later: {count} times ({percentage:.3f}%)"
            )

        self.stdout.write("")
        self.stdout.write("Most common center-to-center moves:")
        for (from_center, to_center), count in center_counter.most_common(20):
            percentage = (count / total_links) * 100

            self.stdout.write(
                f"{from_center} → {to_center}: "
                f"{count} times ({percentage:.3f}%)"
            )

        self.stdout.write("")
        self.stdout.write("Example movements:")
        for example in examples:
            self.stdout.write(str(example))