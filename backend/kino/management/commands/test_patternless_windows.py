from collections import Counter

from django.core.management.base import BaseCommand

from kino.models import KinoDraw
from kino.services.board_pattern_detector import detect_board_patterns
from kino.services.shape_detector import detect_all_shapes


class Command(BaseCommand):
    help = "Analyze how often future KINO windows are patternless / spread-heavy"

    def add_arguments(self, parser):
        parser.add_argument(
            "--horizon",
            type=int,
            default=10,
            help="How many future games to inspect",
        )

        parser.add_argument(
            "--decision-step",
            type=int,
            default=5,
            help="Move decision point every N games",
        )

        parser.add_argument(
            "--min-history",
            type=int,
            default=100,
            help="Skip early draws before this index",
        )

        parser.add_argument(
            "--row-threshold",
            type=int,
            default=6,
            help="Row pattern threshold",
        )

        parser.add_argument(
            "--column-threshold",
            type=int,
            default=5,
            help="Column pattern threshold",
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="How many example windows to print",
        )

    def handle(self, *args, **options):
        horizon = options["horizon"]
        decision_step = options["decision_step"]
        min_history = options["min_history"]
        row_threshold = options["row_threshold"]
        column_threshold = options["column_threshold"]
        limit = options["limit"]

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if len(draws) < min_history + horizon + 1:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Have {len(draws)}, need at least {min_history + horizon + 1}."
                )
            )
            return

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Testing patternless future windows..."))
        self.stdout.write(f"Total draws: {len(draws)}")
        self.stdout.write(f"Horizon: next {horizon} games")
        self.stdout.write(f"Decision step: every {decision_step} games")
        self.stdout.write(f"Row threshold: {row_threshold}+")
        self.stdout.write(f"Column threshold: {column_threshold}+")

        decision_indices = list(
            range(
                min_history,
                len(draws) - horizon,
                decision_step,
            )
        )

        bucket_counter = Counter()
        pattern_score_counter = Counter()
        row_event_counter = Counter()
        column_event_counter = Counter()
        shape_event_counter = Counter()
        examples = []

        total_windows = 0
        total_row_events = 0
        total_column_events = 0
        total_shape_events = 0
        total_pattern_score = 0

        def classify_window(pattern_score):
            if pattern_score == 0:
                return "zero_patterns"
            if pattern_score <= 2:
                return "spread_low"
            if pattern_score <= 8:
                return "light_pattern"
            if pattern_score <= 20:
                return "normal_pattern"
            return "heavy_pattern"

        for counter, current_index in enumerate(decision_indices, start=1):
            if counter % 500 == 0:
                self.stdout.write(
                    f"  tested {counter:,}/{len(decision_indices):,} future windows..."
                )

            future_draws = draws[current_index + 1: current_index + horizon + 1]

            row_events = 0
            column_events = 0
            shape_events = 0

            row_groups = Counter()
            column_groups = Counter()
            shape_groups = Counter()

            future_draw_ids = []

            for draw in future_draws:
                future_draw_ids.append(draw.draw_id)

                board_events = detect_board_patterns(
                    draw_numbers=draw.numbers,
                    row_threshold=row_threshold,
                    column_threshold=column_threshold,
                )

                for event in board_events:
                    if event["pattern_type"] == "row":
                        row_events += 1
                        row_groups[event["group_number"]] += 1

                    elif event["pattern_type"] == "column":
                        column_events += 1
                        column_groups[event["group_number"]] += 1

                detected_shapes = detect_all_shapes(draw.numbers)

                shape_events += len(detected_shapes)

                for shape_event in detected_shapes:
                    shape_groups[shape_event["shape"]] += 1

            pattern_score = row_events + column_events + shape_events
            bucket = classify_window(pattern_score)

            total_windows += 1
            total_row_events += row_events
            total_column_events += column_events
            total_shape_events += shape_events
            total_pattern_score += pattern_score

            bucket_counter[bucket] += 1
            pattern_score_counter[pattern_score] += 1
            row_event_counter[row_events] += 1
            column_event_counter[column_events] += 1
            shape_event_counter[shape_events] += 1

            if len(examples) < limit:
                examples.append({
                    "decision_draw_id": draws[current_index].draw_id,
                    "future_draw_ids": future_draw_ids,
                    "bucket": bucket,
                    "pattern_score": pattern_score,
                    "row_events": row_events,
                    "column_events": column_events,
                    "shape_events": shape_events,
                    "top_rows": row_groups.most_common(5),
                    "top_columns": column_groups.most_common(5),
                    "top_shapes": shape_groups.most_common(5),
                })

        if total_windows == 0:
            self.stdout.write(self.style.WARNING("No windows tested."))
            return

        avg_pattern_score = total_pattern_score / total_windows
        avg_row_events = total_row_events / total_windows
        avg_column_events = total_column_events / total_windows
        avg_shape_events = total_shape_events / total_windows

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Patternless window test finished"))
        self.stdout.write(f"Total tested windows: {total_windows}")
        self.stdout.write(f"Average pattern score: {avg_pattern_score:.3f}")
        self.stdout.write(f"Average row events/window: {avg_row_events:.3f}")
        self.stdout.write(f"Average column events/window: {avg_column_events:.3f}")
        self.stdout.write(f"Average shape events/window: {avg_shape_events:.3f}")

        self.stdout.write("")
        self.stdout.write("Window buckets:")
        for bucket, count in bucket_counter.most_common():
            percentage = (count / total_windows) * 100
            self.stdout.write(f"{bucket}: {count} windows ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Most common pattern scores:")
        for score, count in pattern_score_counter.most_common(30):
            percentage = (count / total_windows) * 100
            self.stdout.write(f"score {score}: {count} windows ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Row-event distribution:")
        for event_count, count in row_event_counter.most_common(20):
            percentage = (count / total_windows) * 100
            self.stdout.write(f"{event_count} row events: {count} windows ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Column-event distribution:")
        for event_count, count in column_event_counter.most_common(20):
            percentage = (count / total_windows) * 100
            self.stdout.write(f"{event_count} column events: {count} windows ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Shape-event distribution:")
        for event_count, count in shape_event_counter.most_common(20):
            percentage = (count / total_windows) * 100
            self.stdout.write(f"{event_count} shape events: {count} windows ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Example future windows:")
        for example in examples:
            self.stdout.write(str(example))