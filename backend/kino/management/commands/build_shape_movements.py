from collections import Counter, defaultdict

from django.core.management.base import BaseCommand
from django.db import transaction

from kino.models import KinoShapeEvent, KinoShapeMovement


class Command(BaseCommand):
    help = "Build and save KINO shape movement vectors"

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
        )

        parser.add_argument(
            "--min-hits",
            type=int,
            default=4,
        )

        parser.add_argument(
            "--future",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--mode",
            type=str,
            default="one-to-one",
            choices=["all", "nearest", "best-overlap", "one-to-one"],
        )

        parser.add_argument(
            "--exclude-same-center",
            action="store_true",
        )

        parser.add_argument(
            "--rebuild",
            action="store_true",
        )

    def handle(self, *args, **options):
        shape = options["shape"]
        min_hits = options["min_hits"]
        future_window = options["future"]
        mode = options["mode"]
        exclude_same_center = options["exclude_same_center"]
        rebuild = options["rebuild"]

        if rebuild:
            deleted_count, _ = KinoShapeMovement.objects.filter(
                shape=shape,
                min_hits=min_hits,
                future_window=future_window,
                mode=mode,
            ).delete()

            self.stdout.write(
                self.style.WARNING(f"Deleted {deleted_count} existing movements.")
            )

        events = list(
            KinoShapeEvent.objects
            .select_related("draw")
            .filter(
                shape=shape,
                hit_count__gte=min_hits,
            )
            .order_by("draw__draw_time", "center_number")
        )

        if not events:
            self.stdout.write(self.style.WARNING("No shape events found."))
            return

        events_by_draw_id = defaultdict(list)
        draw_ids = []

        for event in events:
            draw_id = event.draw.draw_id
            events_by_draw_id[draw_id].append(event)

            if draw_id not in draw_ids:
                draw_ids.append(draw_id)

        draw_ids = sorted(draw_ids)
        draw_index_by_id = {
            draw_id: index
            for index, draw_id in enumerate(draw_ids)
        }

        movement_counter = Counter()
        created_count = 0
        skipped_count = 0

        def overlap_score(first_event, second_event):
            first_hits = set(first_event.hit_numbers)
            second_hits = set(second_event.hit_numbers)
            return len(first_hits.intersection(second_hits))

        def distance_score(first_event, second_event):
            return (
                abs(second_event.center_row - first_event.center_row)
                + abs(second_event.center_col - first_event.center_col)
            )

        def get_future_events(current_event):
            current_draw_id = current_event.draw.draw_id
            current_index = draw_index_by_id[current_draw_id]

            future_events = []

            for future_index in range(
                current_index + 1,
                min(current_index + future_window + 1, len(draw_ids))
            ):
                future_draw_id = draw_ids[future_index]
                future_events.extend(events_by_draw_id.get(future_draw_id, []))

            return future_events

        def select_future_events(current_event, future_events):
            if not future_events:
                return []

            current_draw_id = current_event.draw.draw_id
            current_index = draw_index_by_id[current_draw_id]

            if mode == "all":
                return future_events

            if mode == "nearest":
                sorted_by_gap = sorted(
                    future_events,
                    key=lambda event: (
                        draw_index_by_id[event.draw.draw_id] - current_index
                    ),
                )

                nearest_gap = (
                    draw_index_by_id[sorted_by_gap[0].draw.draw_id] - current_index
                )

                return [
                    event
                    for event in sorted_by_gap
                    if draw_index_by_id[event.draw.draw_id] - current_index == nearest_gap
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
                        draw_index_by_id[future_event.draw.draw_id] - current_index,
                        -overlap_score(current_event, future_event),
                        distance_score(current_event, future_event),
                        future_event.center_number,
                    ),
                )[0]

                return [best_event]

            return []

        movement_rows = []

        for event in events:
            current_draw_id = event.draw.draw_id
            current_index = draw_index_by_id[current_draw_id]

            future_events = get_future_events(event)
            selected_future_events = select_future_events(event, future_events)

            for future_event in selected_future_events:
                future_draw_id = future_event.draw.draw_id
                future_index = draw_index_by_id[future_draw_id]

                delta_row = future_event.center_row - event.center_row
                delta_col = future_event.center_col - event.center_col
                gap = future_index - current_index

                if exclude_same_center and delta_row == 0 and delta_col == 0:
                    skipped_count += 1
                    continue

                movement_counter[(delta_row, delta_col)] += 1

                movement_rows.append(
                    KinoShapeMovement(
                        from_event=event,
                        to_event=future_event,
                        shape=shape,
                        from_draw_id=current_draw_id,
                        to_draw_id=future_draw_id,
                        from_center=event.center_number,
                        to_center=future_event.center_number,
                        delta_row=delta_row,
                        delta_col=delta_col,
                        gap=gap,
                        overlap_score=overlap_score(event, future_event),
                        distance_score=distance_score(event, future_event),
                        mode=mode,
                        future_window=future_window,
                        min_hits=min_hits,
                    )
                )

        with transaction.atomic():
            KinoShapeMovement.objects.bulk_create(
                movement_rows,
                ignore_conflicts=True,
                batch_size=1000,
            )

        created_count = len(movement_rows)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Shape movements build finished"))
        self.stdout.write(f"Shape: {shape}")
        self.stdout.write(f"Min hits: {min_hits}")
        self.stdout.write(f"Future window: {future_window}")
        self.stdout.write(f"Mode: {mode}")
        self.stdout.write(f"Exclude same center: {exclude_same_center}")
        self.stdout.write(f"Movement rows attempted: {created_count}")
        self.stdout.write(f"Skipped same center: {skipped_count}")
        self.stdout.write(f"Total stored movements: {KinoShapeMovement.objects.count()}")

        self.stdout.write("")
        self.stdout.write("Most common vectors in this build:")
        total = sum(movement_counter.values()) or 1

        for (delta_row, delta_col), count in movement_counter.most_common(20):
            percentage = (count / total) * 100

            self.stdout.write(
                f"Δrow {delta_row:+}, Δcol {delta_col:+}: "
                f"{count} times ({percentage:.3f}%)"
            )