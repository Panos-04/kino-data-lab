from collections import Counter, defaultdict

from django.core.management.base import BaseCommand

from kino.models import KinoDraw


ROWS = {
    1: list(range(1, 11)),
    2: list(range(11, 21)),
    3: list(range(21, 31)),
    4: list(range(31, 41)),
    5: list(range(41, 51)),
    6: list(range(51, 61)),
    7: list(range(61, 71)),
    8: list(range(71, 81)),
}

COLUMNS = {
    col: [col + (row * 10) for row in range(0, 8)]
    for col in range(1, 11)
}


def count_hits(draw_numbers, group_numbers):
    draw_set = set(draw_numbers)
    hit_numbers = sorted(draw_set.intersection(group_numbers))

    return {
        "count": len(hit_numbers),
        "numbers": hit_numbers,
    }


class Command(BaseCommand):
    help = "Test KINO board row/column patterns"

    def add_arguments(self, parser):
        parser.add_argument(
            "--row-threshold",
            type=int,
            default=6,
            help="Minimum hits in a row to count as pattern"
        )

        parser.add_argument(
            "--column-threshold",
            type=int,
            default=5,
            help="Minimum hits in a column to count as pattern"
        )

        parser.add_argument(
            "--limit-results",
            type=int,
            default=20,
            help="How many example results to print"
        )

    def handle(self, *args, **options):
        row_threshold = options["row_threshold"]
        column_threshold = options["column_threshold"]
        limit_results = options["limit_results"]

        draws = list(KinoDraw.objects.order_by("draw_time"))

        if not draws:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        row_patterns = []
        column_patterns = []

        row_counter = Counter()
        column_counter = Counter()

        # For streaks: key like ("row", 3) or ("column", 7)
        current_streaks = defaultdict(int)
        best_streaks = defaultdict(int)

        for draw in draws:
            active_keys_this_draw = set()

            for row_id, row_numbers in ROWS.items():
                result = count_hits(draw.numbers, row_numbers)

                if result["count"] >= row_threshold:
                    key = ("row", row_id)
                    active_keys_this_draw.add(key)
                    row_counter[row_id] += 1

                    row_patterns.append({
                        "draw_id": draw.draw_id,
                        "draw_time": draw.draw_time,
                        "type": "row",
                        "group": row_id,
                        "hit_count": result["count"],
                        "hit_numbers": result["numbers"],
                    })

            for column_id, column_numbers in COLUMNS.items():
                result = count_hits(draw.numbers, column_numbers)

                if result["count"] >= column_threshold:
                    key = ("column", column_id)
                    active_keys_this_draw.add(key)
                    column_counter[column_id] += 1

                    column_patterns.append({
                        "draw_id": draw.draw_id,
                        "draw_time": draw.draw_time,
                        "type": "column",
                        "group": column_id,
                        "hit_count": result["count"],
                        "hit_numbers": result["numbers"],
                    })

            # Update streaks
            all_possible_keys = [
                ("row", row_id) for row_id in ROWS.keys()
            ] + [
                ("column", column_id) for column_id in COLUMNS.keys()
            ]

            for key in all_possible_keys:
                if key in active_keys_this_draw:
                    current_streaks[key] += 1
                    best_streaks[key] = max(best_streaks[key], current_streaks[key])
                else:
                    current_streaks[key] = 0

        total_draws = len(draws)

        row_pattern_count = len(row_patterns)
        column_pattern_count = len(column_patterns)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Board pattern test finished"))
        self.stdout.write(f"Total draws tested: {total_draws}")
        self.stdout.write("")

        self.stdout.write(
            f"Row patterns: {row_pattern_count} total "
            f"({(row_pattern_count / total_draws) * 100:.3f}% per draw-event)"
        )
        self.stdout.write(
            f"Column patterns: {column_pattern_count} total "
            f"({(column_pattern_count / total_draws) * 100:.3f}% per draw-event)"
        )

        self.stdout.write("")
        self.stdout.write("Rows with most 6+ hits:")
        for row_id, count in row_counter.most_common():
            self.stdout.write(
                f"Row {row_id}: {count} times "
                f"({(count / total_draws) * 100:.3f}% of draws)"
            )

        self.stdout.write("")
        self.stdout.write("Columns with most 5+ hits:")
        for column_id, count in column_counter.most_common():
            self.stdout.write(
                f"Column {column_id}: {count} times "
                f"({(count / total_draws) * 100:.3f}% of draws)"
            )

        self.stdout.write("")
        self.stdout.write("Best continuation streaks:")
        streak_rows = []

        for key, streak in best_streaks.items():
            if streak <= 1:
                continue

            pattern_type, group_id = key

            streak_rows.append({
                "type": pattern_type,
                "group": group_id,
                "streak": streak,
            })

        streak_rows = sorted(
            streak_rows,
            key=lambda item: item["streak"],
            reverse=True
        )

        for item in streak_rows[:20]:
            self.stdout.write(
                f"{item['type'].title()} {item['group']}: "
                f"{item['streak']} consecutive draws"
            )

        self.stdout.write("")
        self.stdout.write("Example row patterns:")
        for item in row_patterns[:limit_results]:
            self.stdout.write(str(item))

        self.stdout.write("")
        self.stdout.write("Example column patterns:")
        for item in column_patterns[:limit_results]:
            self.stdout.write(str(item))