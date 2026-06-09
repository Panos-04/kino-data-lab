from collections import Counter, defaultdict

from django.core.management.base import BaseCommand

from kino.models import KinoDraw
from kino.services.board_pattern_detector import detect_board_patterns
from kino.services.shape_detector import detect_all_shapes


class Command(BaseCommand):
    help = "Analyze KINO operation/regime sequences draw-by-draw"

    def add_arguments(self, parser):
        parser.add_argument("--row-threshold", type=int, default=6)
        parser.add_argument("--column-threshold", type=int, default=5)
        parser.add_argument("--limit", type=int, default=30)

    def handle(self, *args, **options):
        row_threshold = options["row_threshold"]
        column_threshold = options["column_threshold"]
        limit = options["limit"]

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Analyzing KINO operation sequences..."))
        self.stdout.write(f"Draws: {len(draws)}")

        def number_position(number):
            row = (number - 1) // 10 + 1
            column = (number - 1) % 10 + 1
            return row, column

        def board_zone(avg_row, avg_col):
            if avg_row <= 3:
                vertical = "top"
            elif avg_row >= 6:
                vertical = "bottom"
            else:
                vertical = "center"

            if avg_col <= 3:
                horizontal = "left"
            elif avg_col >= 8:
                horizontal = "right"
            else:
                horizontal = "center"

            if vertical == "center" and horizontal == "center":
                return "center"

            return f"{vertical}_{horizontal}"

        def spread_score(draw_numbers):
            rows = []
            columns = []

            for number in draw_numbers:
                row, column = number_position(number)
                rows.append(row)
                columns.append(column)

            unique_rows = len(set(rows))
            unique_columns = len(set(columns))

            # Higher means more spread.
            return unique_rows + unique_columns

        def center_of_mass(draw_numbers):
            rows = []
            columns = []

            for number in draw_numbers:
                row, column = number_position(number)
                rows.append(row)
                columns.append(column)

            avg_row = sum(rows) / len(rows)
            avg_col = sum(columns) / len(columns)

            return avg_row, avg_col

        def classify_operation(pattern_score, spread, shape_count, row_count, column_count):
            # Pattern-heavy means visible board structure is forming.
            if pattern_score >= 4 or shape_count >= 3:
                return "heavy_pattern"

            if pattern_score >= 2:
                return "normal_pattern"

            # Spread-heavy means many rows/columns touched but no strong pattern.
            if pattern_score == 0 and spread >= 16:
                return "scatter_spread"

            if pattern_score == 0:
                return "quiet_random"

            return "light_pattern"

        operations = []

        previous_avg_row = None
        previous_avg_col = None

        for index, draw in enumerate(draws):
            board_events = detect_board_patterns(
                draw_numbers=draw.numbers,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
            )

            shape_events = detect_all_shapes(draw.numbers)

            row_count = sum(1 for event in board_events if event["pattern_type"] == "row")
            column_count = sum(1 for event in board_events if event["pattern_type"] == "column")
            shape_count = len(shape_events)

            pattern_score = row_count + column_count + shape_count
            spread = spread_score(draw.numbers)

            avg_row, avg_col = center_of_mass(draw.numbers)
            zone = board_zone(avg_row, avg_col)

            if previous_avg_row is None:
                delta_row = 0
                delta_col = 0
                movement_label = "start"
            else:
                delta_row = avg_row - previous_avg_row
                delta_col = avg_col - previous_avg_col

                if abs(delta_row) < 0.35 and abs(delta_col) < 0.35:
                    movement_label = "stable"
                else:
                    vertical = "down" if delta_row > 0 else "up"
                    horizontal = "right" if delta_col > 0 else "left"

                    if abs(delta_row) >= 0.35 and abs(delta_col) >= 0.35:
                        movement_label = f"{vertical}_{horizontal}"
                    elif abs(delta_row) >= 0.35:
                        movement_label = vertical
                    else:
                        movement_label = horizontal

            operation = classify_operation(
                pattern_score=pattern_score,
                spread=spread,
                shape_count=shape_count,
                row_count=row_count,
                column_count=column_count,
            )

            operations.append({
                "index": index,
                "draw_id": draw.draw_id,
                "operation": operation,
                "pattern_score": pattern_score,
                "spread_score": spread,
                "row_events": row_count,
                "column_events": column_count,
                "shape_events": shape_count,
                "avg_row": round(avg_row, 3),
                "avg_col": round(avg_col, 3),
                "zone": zone,
                "movement_label": movement_label,
                "delta_row": round(delta_row, 3),
                "delta_col": round(delta_col, 3),
            })

            previous_avg_row = avg_row
            previous_avg_col = avg_col

            if (index + 1) % 5000 == 0:
                self.stdout.write(f"  analyzed {index + 1:,} draws...")

        # ------------------------------------------------------------
        # Streaks
        # ------------------------------------------------------------

        streaks = []

        current_operation = operations[0]["operation"]
        current_start = 0
        current_items = [operations[0]]

        for item in operations[1:]:
            if item["operation"] == current_operation:
                current_items.append(item)
                continue

            streaks.append({
                "operation": current_operation,
                "length": len(current_items),
                "start_draw_id": current_items[0]["draw_id"],
                "end_draw_id": current_items[-1]["draw_id"],
                "start_index": current_start,
                "end_index": current_items[-1]["index"],
                "avg_pattern_score": round(
                    sum(x["pattern_score"] for x in current_items) / len(current_items),
                    3,
                ),
                "avg_spread_score": round(
                    sum(x["spread_score"] for x in current_items) / len(current_items),
                    3,
                ),
                "zones": Counter(x["zone"] for x in current_items).most_common(3),
                "movements": Counter(x["movement_label"] for x in current_items).most_common(3),
            })

            current_operation = item["operation"]
            current_start = item["index"]
            current_items = [item]

        streaks.append({
            "operation": current_operation,
            "length": len(current_items),
            "start_draw_id": current_items[0]["draw_id"],
            "end_draw_id": current_items[-1]["draw_id"],
            "start_index": current_start,
            "end_index": current_items[-1]["index"],
            "avg_pattern_score": round(
                sum(x["pattern_score"] for x in current_items) / len(current_items),
                3,
            ),
            "avg_spread_score": round(
                sum(x["spread_score"] for x in current_items) / len(current_items),
                3,
            ),
            "zones": Counter(x["zone"] for x in current_items).most_common(3),
            "movements": Counter(x["movement_label"] for x in current_items).most_common(3),
        })

        # ------------------------------------------------------------
        # Transitions
        # ------------------------------------------------------------

        transitions = Counter()

        for previous, current in zip(operations, operations[1:]):
            transitions[(previous["operation"], current["operation"])] += 1

        operation_counts = Counter(item["operation"] for item in operations)
        zone_counts = Counter(item["zone"] for item in operations)
        movement_counts = Counter(item["movement_label"] for item in operations)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Operation sequence analysis finished"))

        self.stdout.write("")
        self.stdout.write("Operation counts:")
        for operation, count in operation_counts.most_common():
            percentage = (count / len(operations)) * 100
            self.stdout.write(f"{operation}: {count:,} draws ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Zone counts:")
        for zone, count in zone_counts.most_common():
            percentage = (count / len(operations)) * 100
            self.stdout.write(f"{zone}: {count:,} draws ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Movement labels:")
        for movement, count in movement_counts.most_common(20):
            percentage = (count / len(operations)) * 100
            self.stdout.write(f"{movement}: {count:,} draws ({percentage:.3f}%)")

        self.stdout.write("")
        self.stdout.write("Most common operation transitions:")
        for (from_operation, to_operation), count in transitions.most_common(20):
            self.stdout.write(f"{from_operation} → {to_operation}: {count:,}")

        self.stdout.write("")
        self.stdout.write("Longest operation streaks:")
        for streak in sorted(streaks, key=lambda item: item["length"], reverse=True)[:limit]:
            self.stdout.write(str(streak))

        self.stdout.write("")
        self.stdout.write("Latest 20 operations:")
        for item in operations[-20:]:
            self.stdout.write(str(item))