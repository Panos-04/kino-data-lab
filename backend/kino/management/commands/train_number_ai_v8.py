
from collections import Counter, defaultdict
from math import comb, log2, sqrt

import numpy as np

from django.core.management.base import BaseCommand
from django.utils import timezone

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kino.models import (
    KinoDraw,
    KinoAIResult,
    KinoBoardPatternEvent,
    KinoShapeEvent,
    KinoShapeMovement,
)


class Command(BaseCommand):
    """
    Experimental V8 KINO AI command.

    Keeps your working train_number_ai_10game.py safe.

    Adds:
    1. Calibration analysis
    2. Walk-forward stability slices
    3. ROI-first regime mode selection
    4. Combo correlation / basket quality
    5. Entropy / dispersion features
    6. Operation transition summaries
    7. Confidence gate + win/loss audit examples
    8. Swap-model rescue: trains a second model to choose the best 1-number swap
    """

    help = "Train experimental KINO AI V8 with swap-model rescue"

    def add_arguments(self, parser):
        parser.add_argument("--horizon", type=int, default=10)
        parser.add_argument("--decision-step", type=int, default=5)
        parser.add_argument("--min-history", type=int, default=100)
        parser.add_argument("--test-ratio", type=float, default=0.2)
        parser.add_argument("--target-hits", type=int, default=3)
        parser.add_argument("--pick", type=int, default=12)

        parser.add_argument("--stake", type=float, default=1.0)
        parser.add_argument("--bonus-fee", type=float, default=1.0)
        parser.add_argument(
            "--payout-table",
            type=str,
            default="kino",
            choices=["kino", "bonus"],
        )

        parser.add_argument("--confidence-play-threshold", type=float, default=80.0)
        parser.add_argument("--confidence-watch-threshold", type=float, default=65.0)
        parser.add_argument(
            "--swap-threshold",
            type=float,
            default=0.55,
            help="Minimum predicted probability for the V8 swap model to apply one swap.",
        )
        parser.add_argument(
            "--swap-max-train-decisions",
            type=int,
            default=0,
            help="Optional cap for swap-model training decisions. 0 means use all training decisions.",
        )

    def handle(self, *args, **options):
        horizon = options["horizon"]
        decision_step = options["decision_step"]
        min_history = options["min_history"]
        test_ratio = options["test_ratio"]
        target_hits = options["target_hits"]
        pick = options["pick"]
        stake = options["stake"]
        bonus_fee = options["bonus_fee"]
        payout_table_name = options["payout_table"]
        confidence_play_threshold = options["confidence_play_threshold"]
        confidence_watch_threshold = options["confidence_watch_threshold"]
        swap_threshold = options["swap_threshold"]
        swap_max_train_decisions = options["swap_max_train_decisions"]

        if pick != 12:
            raise ValueError(
                "V8 currently supports verified 12-number payout tables only. "
                "Run with --pick 12."
            )

        normal_payout_table = {
            12: 1_000_000,
            11: 25_000,
            10: 2_500,
            9: 1_000,
            8: 150,
            7: 25,
            6: 5,
            5: 0,
            4: 0,
            3: 0,
            2: 0,
            1: 0,
            0: 4,
        }

        bonus_payout_table = {
            12: 2_000_000,
            11: 75_000,
            10: 5_500,
            9: 2_200,
            8: 350,
            7: 50,
            6: 10,
            5: 4,
            4: 3.5,
            3: 3,
            2: 2.5,
            1: 2,
            0: 0,
        }

        def log_step(message):
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(f"▶ {message}"))

        def log_done(message):
            self.stdout.write(self.style.SUCCESS(f"✓ {message}"))

        def number_row(number):
            return (number - 1) // 10 + 1

        def number_column(number):
            return (number - 1) % 10 + 1

        def row_numbers(row):
            return list(range((row - 1) * 10 + 1, row * 10 + 1))

        def column_numbers(column):
            return [column + row_index * 10 for row_index in range(0, 8)]

        def zone_for_number(number):
            row = number_row(number)
            column = number_column(number)

            if row <= 3:
                vertical = "top"
            elif row >= 6:
                vertical = "bottom"
            else:
                vertical = "center"

            if column <= 3:
                horizontal = "left"
            elif column >= 8:
                horizontal = "right"
            else:
                horizontal = "center"

            if vertical == "center" and horizontal == "center":
                return "center"

            return f"{vertical}_{horizontal}"

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

        def entropy_from_counts(counts):
            total = sum(counts)

            if total <= 0:
                return 0.0

            entropy = 0.0

            for count in counts:
                if count <= 0:
                    continue

                probability = count / total
                entropy -= probability * log2(probability)

            return float(entropy)

        def operation_from_scores(pattern_score, spread_score, shape_count, row_entropy, column_entropy):
            if pattern_score >= 4 or shape_count >= 3:
                return "heavy_pattern"

            if pattern_score >= 2:
                return "normal_pattern"

            if (
                pattern_score == 0
                and spread_score >= 16
                and row_entropy >= 2.5
                and column_entropy >= 3.0
            ):
                return "scatter_spread"

            if pattern_score == 0:
                return "quiet_random"

            return "light_pattern"

        # ------------------------------------------------------------
        # Load draws
        # ------------------------------------------------------------

        log_step("Loading draws...")

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if len(draws) < min_history + horizon + 100:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Have {len(draws)}, need at least "
                    f"{min_history + horizon + 100}."
                )
            )
            return

        draw_sets = [set(draw.numbers) for draw in draws]
        draw_ids = [draw.draw_id for draw in draws]
        draw_index_by_id = {draw.draw_id: index for index, draw in enumerate(draws)}

        log_done(f"Loaded {len(draws):,} draws")

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Training KINO AI V8..."))
        self.stdout.write(f"Draws: {len(draws):,}")
        self.stdout.write(f"Horizon: {horizon}")
        self.stdout.write(f"Decision step: {decision_step}")
        self.stdout.write(f"Pick: {pick}")
        self.stdout.write(f"Target hits per number: {target_hits}+ in next {horizon}")
        self.stdout.write(f"Payout table: {payout_table_name}")
        self.stdout.write(f"Stake: €{stake:.2f}")

        round_cost = stake + bonus_fee if payout_table_name == "bonus" else stake
        self.stdout.write(f"Cost per round: €{round_cost:.2f}")
        self.stdout.write(f"Cost per combo decision: €{round_cost * horizon:.2f}")

        # ------------------------------------------------------------
        # Number prefix counts
        # ------------------------------------------------------------

        log_step("Building number prefix counts...")

        prefix_counts = np.zeros((len(draws) + 1, 81), dtype=np.int32)

        for index, numbers in enumerate(draw_sets):
            prefix_counts[index + 1] = prefix_counts[index]

            for number in numbers:
                prefix_counts[index + 1][number] += 1

            if (index + 1) % 5000 == 0:
                self.stdout.write(f"  prefix counts built for {index + 1:,} draws...")

        log_done("Number prefix counts ready")

        def make_prefix(matrix):
            return np.vstack(
                [
                    np.zeros((1, matrix.shape[1]), dtype=np.int32),
                    np.cumsum(matrix, axis=0),
                ]
            )

        def make_1d_prefix(vector):
            return np.concatenate(
                [
                    np.zeros(1, dtype=np.float32),
                    np.cumsum(vector).astype(np.float32),
                ]
            )

        def recent_from_prefix(prefix, current_index, column_index, window_size):
            start_index = max(0, current_index - window_size + 1)
            end_index = current_index + 1

            return int(prefix[end_index][column_index] - prefix[start_index][column_index])

        def recent_from_1d_prefix(prefix, current_index, window_size):
            start_index = max(0, current_index - window_size + 1)
            end_index = current_index + 1

            return float(prefix[end_index] - prefix[start_index])

        # ------------------------------------------------------------
        # Stored analysis caches
        # ------------------------------------------------------------

        log_step("Loading board pattern events...")

        board_row_events = np.zeros((len(draws), 9), dtype=np.int32)
        board_column_events = np.zeros((len(draws), 11), dtype=np.int32)
        board_number_hits = np.zeros((len(draws), 81), dtype=np.int32)

        board_event_count = 0

        for event in (
            KinoBoardPatternEvent.objects
            .select_related("draw")
            .iterator(chunk_size=2000)
        ):
            board_event_count += 1

            if board_event_count % 10000 == 0:
                self.stdout.write(f"  board events loaded: {board_event_count:,}")

            draw_index = draw_index_by_id.get(event.draw.draw_id)

            if draw_index is None:
                continue

            if event.pattern_type == "row":
                board_row_events[draw_index][event.group_number] += 1

            elif event.pattern_type == "column":
                board_column_events[draw_index][event.group_number] += 1

            for number in event.hit_numbers:
                board_number_hits[draw_index][number] += 1

        log_done(f"Board pattern events loaded: {board_event_count:,}")

        log_step("Loading shape events...")

        shape_number_hits = np.zeros((len(draws), 81), dtype=np.int32)
        shape_number_area = np.zeros((len(draws), 81), dtype=np.int32)
        shape_centers = np.zeros((len(draws), 81), dtype=np.int32)

        shape_event_count = 0

        for event in (
            KinoShapeEvent.objects
            .select_related("draw")
            .iterator(chunk_size=2000)
        ):
            shape_event_count += 1

            if shape_event_count % 10000 == 0:
                self.stdout.write(f"  shape events loaded: {shape_event_count:,}")

            draw_index = draw_index_by_id.get(event.draw.draw_id)

            if draw_index is None:
                continue

            shape_centers[draw_index][event.center_number] += 1

            for number in event.hit_numbers:
                shape_number_hits[draw_index][number] += 1

            for number in event.shape_numbers:
                shape_number_area[draw_index][number] += 1

        log_done(f"Shape events loaded: {shape_event_count:,}")

        log_step("Loading shape movements...")

        movement_target_centers = np.zeros((len(draws), 81), dtype=np.int32)
        movement_source_centers_completed = np.zeros((len(draws), 81), dtype=np.int32)

        movement_count = 0

        for movement in KinoShapeMovement.objects.iterator(chunk_size=2000):
            movement_count += 1

            if movement_count % 10000 == 0:
                self.stdout.write(f"  movements loaded: {movement_count:,}")

            to_draw_index = draw_index_by_id.get(movement.to_draw_id)

            if to_draw_index is None:
                continue

            movement_target_centers[to_draw_index][movement.to_center] += 1
            movement_source_centers_completed[to_draw_index][movement.from_center] += 1

        log_done(f"Shape movements loaded: {movement_count:,}")

        # ------------------------------------------------------------
        # Prefix matrices
        # ------------------------------------------------------------

        log_step("Building prefix matrices...")

        board_row_events_prefix = make_prefix(board_row_events)
        board_column_events_prefix = make_prefix(board_column_events)
        board_number_hits_prefix = make_prefix(board_number_hits)

        shape_number_hits_prefix = make_prefix(shape_number_hits)
        shape_number_area_prefix = make_prefix(shape_number_area)
        shape_centers_prefix = make_prefix(shape_centers)

        movement_target_centers_prefix = make_prefix(movement_target_centers)
        movement_source_centers_completed_prefix = make_prefix(movement_source_centers_completed)

        board_total_events = (
            board_row_events.sum(axis=1)
            + board_column_events.sum(axis=1)
        ).astype(np.float32)

        shape_total_events = shape_centers.sum(axis=1).astype(np.float32)
        movement_total_events = movement_target_centers.sum(axis=1).astype(np.float32)

        board_total_prefix = make_1d_prefix(board_total_events)
        shape_total_prefix = make_1d_prefix(shape_total_events)
        movement_total_prefix = make_1d_prefix(movement_total_events)

        log_done("Prefix matrices ready")

        # ------------------------------------------------------------
        # Operation / entropy / transition vectors
        # ------------------------------------------------------------

        log_step("Building operation / entropy / transition vectors...")

        operation_names = [
            "heavy_pattern",
            "normal_pattern",
            "light_pattern",
            "scatter_spread",
            "quiet_random",
        ]

        zone_names = [
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center",
            "center_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ]

        draw_pattern_score = np.zeros(len(draws), dtype=np.float32)
        draw_spread_score = np.zeros(len(draws), dtype=np.float32)
        draw_avg_row = np.zeros(len(draws), dtype=np.float32)
        draw_avg_col = np.zeros(len(draws), dtype=np.float32)
        draw_delta_row = np.zeros(len(draws), dtype=np.float32)
        draw_delta_col = np.zeros(len(draws), dtype=np.float32)
        draw_abs_movement = np.zeros(len(draws), dtype=np.float32)

        draw_row_entropy = np.zeros(len(draws), dtype=np.float32)
        draw_column_entropy = np.zeros(len(draws), dtype=np.float32)
        draw_zone_entropy = np.zeros(len(draws), dtype=np.float32)
        draw_neighbor_density = np.zeros(len(draws), dtype=np.float32)
        draw_avg_pair_distance = np.zeros(len(draws), dtype=np.float32)

        operation_labels = []
        zone_labels = []

        operation_one_hot = np.zeros((len(draws), len(operation_names)), dtype=np.int32)
        zone_one_hot = np.zeros((len(draws), len(zone_names)), dtype=np.int32)

        operation_streak_length = np.zeros(len(draws), dtype=np.float32)
        heavy_streak_length = np.zeros(len(draws), dtype=np.float32)
        normal_streak_length = np.zeros(len(draws), dtype=np.float32)
        light_streak_length = np.zeros(len(draws), dtype=np.float32)
        scatter_streak_length = np.zeros(len(draws), dtype=np.float32)
        quiet_streak_length = np.zeros(len(draws), dtype=np.float32)

        previous_avg_row = None
        previous_avg_col = None
        previous_operation = None
        current_streak = 0

        for index, numbers in enumerate(draw_sets):
            rows = [number_row(number) for number in numbers]
            columns = [number_column(number) for number in numbers]
            zones = [zone_for_number(number) for number in numbers]

            avg_row = sum(rows) / len(rows)
            avg_col = sum(columns) / len(columns)

            row_counts = [rows.count(row) for row in range(1, 9)]
            column_counts = [columns.count(column) for column in range(1, 11)]
            zone_counts = [zones.count(zone) for zone in zone_names]

            row_entropy = entropy_from_counts(row_counts)
            column_entropy = entropy_from_counts(column_counts)
            zone_entropy = entropy_from_counts(zone_counts)

            unique_rows = len(set(rows))
            unique_columns = len(set(columns))
            spread_score = unique_rows + unique_columns

            positions = [(number_row(number), number_column(number)) for number in numbers]
            distances = []
            neighbor_pairs = 0

            for pos_i in range(len(positions)):
                for pos_j in range(pos_i + 1, len(positions)):
                    row_a, col_a = positions[pos_i]
                    row_b, col_b = positions[pos_j]
                    distance = sqrt((row_a - row_b) ** 2 + (col_a - col_b) ** 2)
                    distances.append(distance)

                    if abs(row_a - row_b) <= 1 and abs(col_a - col_b) <= 1:
                        neighbor_pairs += 1

            avg_pair_distance = float(np.mean(distances)) if distances else 0.0
            max_pairs = (len(positions) * (len(positions) - 1)) / 2
            neighbor_density = (neighbor_pairs / max_pairs) if max_pairs else 0.0

            pattern_score = float(board_total_events[index] + shape_total_events[index])
            shape_count = float(shape_total_events[index])

            operation = operation_from_scores(
                pattern_score=pattern_score,
                spread_score=spread_score,
                shape_count=shape_count,
                row_entropy=row_entropy,
                column_entropy=column_entropy,
            )

            zone = board_zone(avg_row, avg_col)

            if previous_avg_row is None:
                delta_row = 0.0
                delta_col = 0.0
            else:
                delta_row = avg_row - previous_avg_row
                delta_col = avg_col - previous_avg_col

            abs_movement = abs(delta_row) + abs(delta_col)

            if operation == previous_operation:
                current_streak += 1
            else:
                current_streak = 1

            operation_labels.append(operation)
            zone_labels.append(zone)

            draw_pattern_score[index] = pattern_score
            draw_spread_score[index] = spread_score
            draw_avg_row[index] = avg_row
            draw_avg_col[index] = avg_col
            draw_delta_row[index] = delta_row
            draw_delta_col[index] = delta_col
            draw_abs_movement[index] = abs_movement

            draw_row_entropy[index] = row_entropy
            draw_column_entropy[index] = column_entropy
            draw_zone_entropy[index] = zone_entropy
            draw_neighbor_density[index] = neighbor_density
            draw_avg_pair_distance[index] = avg_pair_distance

            operation_streak_length[index] = current_streak

            if operation in operation_names:
                operation_one_hot[index][operation_names.index(operation)] = 1

            if zone in zone_names:
                zone_one_hot[index][zone_names.index(zone)] = 1

            heavy_streak_length[index] = current_streak if operation == "heavy_pattern" else 0
            normal_streak_length[index] = current_streak if operation == "normal_pattern" else 0
            light_streak_length[index] = current_streak if operation == "light_pattern" else 0
            scatter_streak_length[index] = current_streak if operation == "scatter_spread" else 0
            quiet_streak_length[index] = current_streak if operation == "quiet_random" else 0

            previous_avg_row = avg_row
            previous_avg_col = avg_col
            previous_operation = operation

        pattern_score_prefix = make_1d_prefix(draw_pattern_score)
        spread_score_prefix = make_1d_prefix(draw_spread_score)
        abs_movement_prefix = make_1d_prefix(draw_abs_movement)

        row_entropy_prefix = make_1d_prefix(draw_row_entropy)
        column_entropy_prefix = make_1d_prefix(draw_column_entropy)
        zone_entropy_prefix = make_1d_prefix(draw_zone_entropy)
        neighbor_density_prefix = make_1d_prefix(draw_neighbor_density)
        avg_pair_distance_prefix = make_1d_prefix(draw_avg_pair_distance)

        operation_one_hot_prefix = make_prefix(operation_one_hot)
        zone_one_hot_prefix = make_prefix(zone_one_hot)

        transition_counter = Counter()
        zone_transition_counter = Counter()

        for previous_index in range(0, len(draws) - 1):
            transition_counter[
                (operation_labels[previous_index], operation_labels[previous_index + 1])
            ] += 1

            zone_transition_counter[
                (zone_labels[previous_index], zone_labels[previous_index + 1])
            ] += 1

        operation_transition_summary = [
            {"from": key[0], "to": key[1], "count": count}
            for key, count in transition_counter.most_common(30)
        ]

        zone_transition_summary = [
            {"from": key[0], "to": key[1], "count": count}
            for key, count in zone_transition_counter.most_common(30)
        ]

        log_done("Operation / entropy / transition vectors ready")

        # ------------------------------------------------------------
        # Feature helpers
        # ------------------------------------------------------------

        def count_in_window(current_index, number, window_size):
            start_index = max(0, current_index - window_size + 1)
            end_index = current_index + 1

            return prefix_counts[end_index][number] - prefix_counts[start_index][number]

        def future_count(current_index, number):
            start_index = current_index + 1
            end_index = min(current_index + horizon + 1, len(draws))

            return prefix_counts[end_index][number] - prefix_counts[start_index][number]

        def gap_since_seen(current_index, number):
            for index in range(current_index, max(-1, current_index - 500), -1):
                if number in draw_sets[index]:
                    return current_index - index

            return 999

        def extra_number_features(current_index, number):
            row = number_row(number)
            column = number_column(number)

            return [
                recent_from_prefix(board_row_events_prefix, current_index, row, 10),
                recent_from_prefix(board_row_events_prefix, current_index, row, 50),
                recent_from_prefix(board_column_events_prefix, current_index, column, 10),
                recent_from_prefix(board_column_events_prefix, current_index, column, 50),

                recent_from_prefix(board_number_hits_prefix, current_index, number, 10),
                recent_from_prefix(board_number_hits_prefix, current_index, number, 50),

                recent_from_prefix(shape_number_hits_prefix, current_index, number, 10),
                recent_from_prefix(shape_number_hits_prefix, current_index, number, 50),
                recent_from_prefix(shape_number_area_prefix, current_index, number, 10),
                recent_from_prefix(shape_number_area_prefix, current_index, number, 50),

                recent_from_prefix(shape_centers_prefix, current_index, number, 20),
                recent_from_prefix(shape_centers_prefix, current_index, number, 100),

                recent_from_prefix(movement_target_centers_prefix, current_index, number, 20),
                recent_from_prefix(movement_target_centers_prefix, current_index, number, 100),
                recent_from_prefix(movement_source_centers_completed_prefix, current_index, number, 20),
                recent_from_prefix(movement_source_centers_completed_prefix, current_index, number, 100),
            ]

        def operation_features(current_index):
            op_counts_last_10 = [
                recent_from_prefix(operation_one_hot_prefix, current_index, op_index, 10)
                for op_index in range(len(operation_names))
            ]

            zone_counts_last_10 = [
                recent_from_prefix(zone_one_hot_prefix, current_index, zone_index, 10)
                for zone_index in range(len(zone_names))
            ]

            return [
                draw_pattern_score[current_index],
                draw_spread_score[current_index],
                draw_avg_row[current_index],
                draw_avg_col[current_index],
                draw_delta_row[current_index],
                draw_delta_col[current_index],
                draw_abs_movement[current_index],

                draw_row_entropy[current_index],
                draw_column_entropy[current_index],
                draw_zone_entropy[current_index],
                draw_neighbor_density[current_index],
                draw_avg_pair_distance[current_index],

                recent_from_1d_prefix(pattern_score_prefix, current_index, 3),
                recent_from_1d_prefix(pattern_score_prefix, current_index, 5),
                recent_from_1d_prefix(pattern_score_prefix, current_index, 10),

                recent_from_1d_prefix(spread_score_prefix, current_index, 3),
                recent_from_1d_prefix(spread_score_prefix, current_index, 5),
                recent_from_1d_prefix(spread_score_prefix, current_index, 10),

                recent_from_1d_prefix(abs_movement_prefix, current_index, 3),
                recent_from_1d_prefix(abs_movement_prefix, current_index, 5),
                recent_from_1d_prefix(abs_movement_prefix, current_index, 10),

                recent_from_1d_prefix(row_entropy_prefix, current_index, 5),
                recent_from_1d_prefix(column_entropy_prefix, current_index, 5),
                recent_from_1d_prefix(zone_entropy_prefix, current_index, 5),
                recent_from_1d_prefix(neighbor_density_prefix, current_index, 5),
                recent_from_1d_prefix(avg_pair_distance_prefix, current_index, 5),

                operation_streak_length[current_index],
                heavy_streak_length[current_index],
                normal_streak_length[current_index],
                light_streak_length[current_index],
                scatter_streak_length[current_index],
                quiet_streak_length[current_index],

                *op_counts_last_10,
                *zone_counts_last_10,
            ]

        def current_pattern_pressure_score(current_index):
            board_10 = recent_from_1d_prefix(board_total_prefix, current_index, 10)
            board_50 = recent_from_1d_prefix(board_total_prefix, current_index, 50)
            shape_10 = recent_from_1d_prefix(shape_total_prefix, current_index, 10)
            shape_50 = recent_from_1d_prefix(shape_total_prefix, current_index, 50)
            movement_20 = recent_from_1d_prefix(movement_total_prefix, current_index, 20)
            movement_100 = recent_from_1d_prefix(movement_total_prefix, current_index, 100)
            operation_pressure = recent_from_1d_prefix(pattern_score_prefix, current_index, 5)
            entropy_pressure = recent_from_1d_prefix(zone_entropy_prefix, current_index, 5)

            return (
                board_10 * 2.0
                + board_50 * 0.35
                + shape_10 * 1.0
                + shape_50 * 0.15
                + movement_20 * 1.5
                + movement_100 * 0.20
                + operation_pressure * 0.75
                - scatter_streak_length[current_index] * 2.0
                + entropy_pressure * 0.25
            )

        feature_names = [
            "number", "row", "column",
            "count_last_5", "count_last_10", "count_last_20", "count_last_50", "count_last_100",
            "ratio_last_10", "ratio_last_20", "ratio_last_50",
            "gap_since_seen", "appeared_current_draw", "appeared_previous_draw",
            "row_hits_current_draw", "column_hits_current_draw",

            "row_pattern_last_10", "row_pattern_last_50",
            "column_pattern_last_10", "column_pattern_last_50",
            "number_board_pattern_hit_last_10", "number_board_pattern_hit_last_50",
            "number_shape_hit_last_10", "number_shape_hit_last_50",
            "number_shape_area_last_10", "number_shape_area_last_50",
            "number_shape_center_last_20", "number_shape_center_last_100",
            "movement_target_center_last_20", "movement_target_center_last_100",
            "movement_source_center_completed_last_20", "movement_source_center_completed_last_100",

            "draw_pattern_score", "draw_spread_score", "draw_avg_row", "draw_avg_col",
            "draw_delta_row", "draw_delta_col", "draw_abs_movement",
            "draw_row_entropy", "draw_column_entropy", "draw_zone_entropy",
            "draw_neighbor_density", "draw_avg_pair_distance",

            "pattern_score_last_3", "pattern_score_last_5", "pattern_score_last_10",
            "spread_score_last_3", "spread_score_last_5", "spread_score_last_10",
            "abs_movement_last_3", "abs_movement_last_5", "abs_movement_last_10",
            "row_entropy_last_5", "column_entropy_last_5", "zone_entropy_last_5",
            "neighbor_density_last_5", "avg_pair_distance_last_5",

            "operation_streak_length",
            "heavy_streak_length", "normal_streak_length", "light_streak_length",
            "scatter_streak_length", "quiet_streak_length",

            *[f"operation_last_10_{name}" for name in operation_names],
            *[f"zone_last_10_{name}" for name in zone_names],
        ]

        def feature_group_for_name(name):
            if name in ["number", "row", "column"]:
                return "identity"

            if "count_last" in name or "ratio_last" in name or "gap" in name or "appeared" in name:
                return "hot_cold"

            if "row_pattern" in name or "column_pattern" in name or "board_pattern" in name:
                return "board_pattern"

            if "shape" in name:
                return "shape"

            if "movement" in name:
                return "movement"

            if (
                "entropy" in name
                or "spread" in name
                or "density" in name
                or "distance" in name
            ):
                return "entropy_dispersion"

            if (
                "operation" in name
                or "streak" in name
                or "zone" in name
                or "pattern_score" in name
                or "avg_" in name
                or "delta" in name
            ):
                return "operation_transition"

            return "other"

        # ------------------------------------------------------------
        # Baseline target probability
        # ------------------------------------------------------------

        baseline_target_probability = 0.0

        for hits in range(target_hits, horizon + 1):
            baseline_target_probability += (
                comb(horizon, hits)
                * (0.25 ** hits)
                * (0.75 ** (horizon - hits))
            )

        # ------------------------------------------------------------
        # Decision points and regimes
        # ------------------------------------------------------------

        decision_indices = list(range(min_history, len(draws) - horizon, decision_step))
        split_draw_index = int(len(draws) * (1 - test_ratio))

        train_decision_indices = [
            index for index in decision_indices
            if index < split_draw_index
        ]

        train_regime_scores = np.array(
            [current_pattern_pressure_score(index) for index in train_decision_indices],
            dtype=np.float32,
        )

        if len(train_regime_scores) == 0:
            self.stdout.write(self.style.WARNING("No train decision regime scores found."))
            return

        regime_thresholds = {
            "q25": float(np.quantile(train_regime_scores, 0.25)),
            "q50": float(np.quantile(train_regime_scores, 0.50)),
            "q75": float(np.quantile(train_regime_scores, 0.75)),
        }

        def classify_regime(score):
            if score <= regime_thresholds["q25"]:
                return "spread_low"

            if score <= regime_thresholds["q50"]:
                return "light_pattern"

            if score <= regime_thresholds["q75"]:
                return "normal_pattern"

            return "heavy_pattern"

        decision_regime_by_index = {
            index: classify_regime(current_pattern_pressure_score(index))
            for index in decision_indices
        }

        # ------------------------------------------------------------
        # Build ML rows
        # ------------------------------------------------------------

        log_step(f"Building ML rows for {len(decision_indices):,} decision points...")

        X_rows = []
        y_rows = []
        future_count_rows = []
        row_draw_indices = []

        for decision_counter, current_index in enumerate(decision_indices, start=1):
            if decision_counter % 250 == 0:
                self.stdout.write(
                    f"  built {decision_counter:,}/{len(decision_indices):,} decision points "
                    f"({decision_counter * 80:,} rows)"
                )

            current_numbers = draw_sets[current_index]
            previous_numbers = draw_sets[current_index - 1]
            current_operation_features = operation_features(current_index)

            for number in range(1, 81):
                row = number_row(number)
                column = number_column(number)

                count_last_5 = count_in_window(current_index, number, 5)
                count_last_10 = count_in_window(current_index, number, 10)
                count_last_20 = count_in_window(current_index, number, 20)
                count_last_50 = count_in_window(current_index, number, 50)
                count_last_100 = count_in_window(current_index, number, 100)

                row_hits_current = len(current_numbers.intersection(row_numbers(row)))
                column_hits_current = len(current_numbers.intersection(column_numbers(column)))
                next_horizon_count = future_count(current_index, number)

                features = [
                    number,
                    row,
                    column,
                    count_last_5,
                    count_last_10,
                    count_last_20,
                    count_last_50,
                    count_last_100,
                    count_last_10 / 10,
                    count_last_20 / 20,
                    count_last_50 / 50,
                    gap_since_seen(current_index, number),
                    1 if number in current_numbers else 0,
                    1 if number in previous_numbers else 0,
                    row_hits_current,
                    column_hits_current,
                ]

                features.extend(extra_number_features(current_index, number))
                features.extend(current_operation_features)

                target = 1 if next_horizon_count >= target_hits else 0

                X_rows.append(features)
                y_rows.append(target)
                future_count_rows.append(next_horizon_count)
                row_draw_indices.append(current_index)

        log_done(f"Built ML dataset: {len(X_rows):,} rows")

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int8)
        future_counts = np.array(future_count_rows, dtype=np.int16)
        row_draw_indices = np.array(row_draw_indices, dtype=np.int32)

        train_mask = row_draw_indices < split_draw_index
        test_mask = row_draw_indices >= split_draw_index

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        future_counts_train = future_counts[train_mask]
        future_counts_test = future_counts[test_mask]

        train_draw_indices = row_draw_indices[train_mask]
        test_draw_indices = row_draw_indices[test_mask]

        self.stdout.write(f"Training rows: {len(X_train):,}")
        self.stdout.write(f"Testing rows: {len(X_test):,}")

        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            self.stdout.write(
                self.style.ERROR(
                    f"Training target has only one class: {unique_classes.tolist()}"
                )
            )
            self.stdout.write("Try horizon 1 -> target 1, horizon 5 -> target 2, horizon 10 -> target 3")
            return

        # ------------------------------------------------------------
        # Train model
        # ------------------------------------------------------------

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=200,
                        solver="saga",
                        n_jobs=-1,
                        verbose=1,
                    ),
                ),
            ]
        )

        log_step("Fitting model...")
        model.fit(X_train, y_train)
        log_done("Model fitting complete")

        # ------------------------------------------------------------
        # Predictions / calibration
        # ------------------------------------------------------------

        log_step("Predicting probabilities and calculating calibration...")

        train_probabilities = model.predict_proba(X_train)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, zero_division=0)
        recall = recall_score(y_test, test_predictions, zero_division=0)

        def calibration_table(probabilities, targets, buckets=10):
            if len(probabilities) == 0:
                return []

            quantiles = np.quantile(probabilities, np.linspace(0, 1, buckets + 1))
            rows = []

            for bucket_index in range(buckets):
                low = quantiles[bucket_index]
                high = quantiles[bucket_index + 1]

                if bucket_index == buckets - 1:
                    mask = (probabilities >= low) & (probabilities <= high)
                else:
                    mask = (probabilities >= low) & (probabilities < high)

                count = int(mask.sum())

                if count == 0:
                    continue

                avg_probability = float(np.mean(probabilities[mask]))
                actual_rate = float(np.mean(targets[mask]))

                rows.append(
                    {
                        "bucket": bucket_index + 1,
                        "low_probability": round(float(low), 6),
                        "high_probability": round(float(high), 6),
                        "count": count,
                        "avg_probability": round(avg_probability, 6),
                        "actual_rate": round(actual_rate, 6),
                        "lift_vs_baseline": round(actual_rate - baseline_target_probability, 6),
                    }
                )

            return rows

        train_calibration = calibration_table(train_probabilities, y_train)
        test_calibration = calibration_table(test_probabilities, y_test)

        def empirical_probability_lift(probability):
            if not train_calibration:
                return 0.0

            best_row = train_calibration[0]

            for row in train_calibration:
                if row["low_probability"] <= probability <= row["high_probability"]:
                    best_row = row
                    break

                if probability >= row["high_probability"]:
                    best_row = row

            return float(best_row.get("lift_vs_baseline", 0.0))

        log_done("Calibration ready")

        # ------------------------------------------------------------
        # Pick / profit / quality helpers
        # ------------------------------------------------------------

        def are_neighbors(first_number, second_number):
            row_a = number_row(first_number)
            col_a = number_column(first_number)
            row_b = number_row(second_number)
            col_b = number_column(second_number)

            return abs(row_a - row_b) <= 1 and abs(col_a - col_b) <= 1

        def select_raw_pick(draw_probs, numbers, pick_size):
            return np.argsort(draw_probs)[::-1][:pick_size]

        def select_low_pick(draw_probs, numbers, pick_size):
            return np.argsort(draw_probs)[:pick_size]

        def select_spread_pick(draw_probs, numbers, pick_size, max_per_row=2, max_per_column=2, max_neighbors=2):
            ranked_indices = np.argsort(draw_probs)[::-1]
            selected_indices = []
            row_counts = {}
            column_counts = {}

            for index in ranked_indices:
                number = int(numbers[index])
                row = number_row(number)
                column = number_column(number)

                if row_counts.get(row, 0) >= max_per_row:
                    continue

                if column_counts.get(column, 0) >= max_per_column:
                    continue

                neighbor_count = sum(
                    1 for selected_index in selected_indices
                    if are_neighbors(number, int(numbers[selected_index]))
                )

                if neighbor_count >= max_neighbors:
                    continue

                selected_indices.append(index)
                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

                if len(selected_indices) >= pick_size:
                    break

            if len(selected_indices) < pick_size:
                selected_set = set(selected_indices)

                for index in ranked_indices:
                    if index in selected_set:
                        continue

                    selected_indices.append(index)
                    selected_set.add(index)

                    if len(selected_indices) >= pick_size:
                        break

            return np.array(selected_indices[:pick_size])

        def select_hybrid_pick(draw_probs, numbers, pick_size):
            ranked_indices = np.argsort(draw_probs)[::-1]
            locked_count = min(4, pick_size)
            selected_indices = list(ranked_indices[:locked_count])

            row_counts = {}
            column_counts = {}

            for index in selected_indices:
                number = int(numbers[index])
                row = number_row(number)
                column = number_column(number)
                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

            for index in ranked_indices:
                if index in selected_indices:
                    continue

                number = int(numbers[index])
                row = number_row(number)
                column = number_column(number)

                if row_counts.get(row, 0) >= 2:
                    continue

                if column_counts.get(column, 0) >= 2:
                    continue

                selected_indices.append(index)
                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

                if len(selected_indices) >= pick_size:
                    break

            if len(selected_indices) < pick_size:
                selected_set = set(selected_indices)

                for index in ranked_indices:
                    if index in selected_set:
                        continue

                    selected_indices.append(index)
                    selected_set.add(index)

                    if len(selected_indices) >= pick_size:
                        break

            return np.array(selected_indices[:pick_size])

        def select_relaxed_hybrid_pick(draw_probs, numbers, pick_size):
            """
            A less restrictive hybrid selector.

            Why this exists:
            near-miss analysis showed many missing winners were already ranked
            close to the selected combo. The normal hybrid selector may be
            over-filtering strong ranked numbers because of row/column limits.
            """
            ranked_indices = np.argsort(draw_probs)[::-1]
            locked_count = min(6, pick_size)
            selected_indices = list(ranked_indices[:locked_count])

            row_counts = {}
            column_counts = {}

            for index in selected_indices:
                number = int(numbers[index])
                row = number_row(number)
                column = number_column(number)
                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

            for index in ranked_indices:
                if index in selected_indices:
                    continue

                number = int(numbers[index])
                row = number_row(number)
                column = number_column(number)

                if row_counts.get(row, 0) >= 3:
                    continue

                if column_counts.get(column, 0) >= 3:
                    continue

                selected_indices.append(index)
                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

                if len(selected_indices) >= pick_size:
                    break

            if len(selected_indices) < pick_size:
                selected_set = set(selected_indices)

                for index in ranked_indices:
                    if index in selected_set:
                        continue

                    selected_indices.append(index)
                    selected_set.add(index)

                    if len(selected_indices) >= pick_size:
                        break

            return np.array(selected_indices[:pick_size])

        def select_rescue_pick(base_indices, draw_probs, numbers, pick_size, rescue_count=1, reserve_limit=20):
            """
            Rescue selector.

            Starts from the regime-aware combo, then replaces the weakest 1/2/3
            selected numbers with the best reserve candidates from the model's
            top-ranked numbers.

            This is a pre-draw rule. It does NOT look at future winning numbers.
            It only uses the same probability ranking available before the draw.
            """
            ranked_indices = list(np.argsort(draw_probs)[::-1])
            selected_indices = [int(index) for index in list(base_indices)]
            selected_set = set(selected_indices)

            reserve_indices = [
                int(index)
                for index in ranked_indices[:reserve_limit]
                if int(index) not in selected_set
            ]

            if not reserve_indices:
                return np.array(selected_indices[:pick_size])

            # Weakest selected = lowest model probability among selected numbers.
            weak_selected_indices = sorted(
                selected_indices,
                key=lambda index: float(draw_probs[index]),
            )

            swaps_done = 0

            for weak_index, reserve_index in zip(weak_selected_indices, reserve_indices):
                if swaps_done >= rescue_count:
                    break

                # Only rescue when the reserve is actually ranked/probability better.
                if float(draw_probs[reserve_index]) <= float(draw_probs[weak_index]):
                    continue

                replace_position = selected_indices.index(weak_index)
                selected_indices[replace_position] = reserve_index
                selected_set.remove(weak_index)
                selected_set.add(reserve_index)
                swaps_done += 1

            return np.array(selected_indices[:pick_size])

        def select_mode_indices(mode, draw_probs, numbers, pick_size):
            if mode == "raw":
                return select_raw_pick(draw_probs, numbers, pick_size)

            if mode == "spread":
                return select_spread_pick(draw_probs, numbers, pick_size)

            if mode == "hybrid":
                return select_hybrid_pick(draw_probs, numbers, pick_size)

            if mode == "relaxed_hybrid":
                return select_relaxed_hybrid_pick(draw_probs, numbers, pick_size)

            if mode == "miss":
                return select_low_pick(draw_probs, numbers, pick_size)

            return select_hybrid_pick(draw_probs, numbers, pick_size)

        def build_groups(draw_indices):
            groups = defaultdict(list)

            for position, draw_index in enumerate(draw_indices.tolist()):
                groups[int(draw_index)].append(position)

            return {
                draw_index: np.array(positions, dtype=np.int32)
                for draw_index, positions in groups.items()
            }

        def calculate_normal_payout(hit_count):
            return float(normal_payout_table.get(hit_count, 0)) * stake

        def calculate_bonus_payout(hit_count):
            return float(bonus_payout_table.get(hit_count, 0)) * stake

        def calculate_pick_profit(selected_numbers, current_index):
            selected_set = set(selected_numbers)
            total_cost = 0.0
            total_return = 0.0
            hit_distribution = defaultdict(int)
            bonus_hit_distribution = defaultdict(int)
            payout_source_distribution = defaultdict(int)
            round_details = []

            start_index = current_index + 1
            end_index = min(current_index + horizon + 1, len(draws))
            this_round_cost = stake + bonus_fee if payout_table_name == "bonus" else stake

            for future_index in range(start_index, end_index):
                future_numbers = list(draws[future_index].numbers)
                future_set = set(future_numbers)
                bonus_number = future_numbers[-1] if future_numbers else None

                hit_numbers = sorted(selected_set.intersection(future_set))
                hit_count = len(hit_numbers)
                bonus_hit = bonus_number in selected_set

                if payout_table_name == "kino":
                    payout = calculate_normal_payout(hit_count)
                    payout_source = "kino"

                elif payout_table_name == "bonus":
                    if bonus_hit:
                        payout = calculate_bonus_payout(hit_count)
                        payout_source = "bonus"
                    else:
                        payout = calculate_normal_payout(hit_count)
                        payout_source = "kino_fallback"

                else:
                    raise ValueError(f"Unknown payout table: {payout_table_name}")

                total_cost += this_round_cost
                total_return += payout
                hit_distribution[hit_count] += 1
                bonus_hit_distribution[str(bonus_hit)] += 1
                payout_source_distribution[payout_source] += 1

                round_details.append(
                    {
                        "future_draw_id": draws[future_index].draw_id,
                        "hit_count": hit_count,
                        "hit_numbers": hit_numbers,
                        "bonus_number": bonus_number,
                        "bonus_hit": bonus_hit,
                        "payout": payout,
                        "payout_source": payout_source,
                    }
                )

            profit = total_return - total_cost
            roi = (profit / total_cost) * 100 if total_cost > 0 else 0.0

            return {
                "rounds_played": end_index - start_index,
                "cost": total_cost,
                "return": total_return,
                "profit": profit,
                "roi": roi,
                "hit_distribution": dict(hit_distribution),
                "bonus_hit_distribution": dict(bonus_hit_distribution),
                "payout_source_distribution": dict(payout_source_distribution),
                "round_details": round_details,
            }

        def summarize_profit(results):
            total_rounds = sum(item["rounds_played"] for item in results)
            total_cost = sum(item["cost"] for item in results)
            total_return = sum(item["return"] for item in results)
            total_profit = total_return - total_cost
            roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0.0

            hit_distribution = defaultdict(int)
            bonus_hit_distribution = defaultdict(int)
            payout_source_distribution = defaultdict(int)

            for item in results:
                for hits, count in item["hit_distribution"].items():
                    hit_distribution[int(hits)] += count

                for value, count in item.get("bonus_hit_distribution", {}).items():
                    bonus_hit_distribution[value] += count

                for source, count in item.get("payout_source_distribution", {}).items():
                    payout_source_distribution[source] += count

            profitable_decisions = sum(1 for item in results if item["profit"] > 0)
            losing_decisions = sum(1 for item in results if item["profit"] < 0)
            break_even_decisions = sum(1 for item in results if item["profit"] == 0)

            zero_hit_rounds = hit_distribution.get(0, 0)
            paying_high_rounds = sum(
                count for hits, count in hit_distribution.items()
                if int(hits) >= 6
            )
            dead_zone_rounds = sum(
                count for hits, count in hit_distribution.items()
                if 1 <= int(hits) <= 5
            )
            paying_rounds = zero_hit_rounds + paying_high_rounds
            paying_round_rate = (paying_rounds / total_rounds) * 100 if total_rounds else 0.0
            dead_zone_rate = (dead_zone_rounds / total_rounds) * 100 if total_rounds else 0.0

            return {
                "stake_per_round": round(stake, 2),
                "bonus_fee_per_round": round(bonus_fee, 2),
                "cost_per_round": round(round_cost, 2),
                "rounds_per_combo": horizon,
                "cost_per_combo_decision": round(round_cost * horizon, 2),
                "total_combo_decisions": len(results),
                "total_rounds_played": total_rounds,
                "total_cost": round(total_cost, 2),
                "total_return": round(total_return, 2),
                "total_profit": round(total_profit, 2),
                "roi": round(roi, 4),
                "profitable_decisions": profitable_decisions,
                "losing_decisions": losing_decisions,
                "break_even_decisions": break_even_decisions,
                "hit_distribution": dict(sorted(hit_distribution.items())),
                "bonus_hit_distribution": dict(bonus_hit_distribution),
                "payout_source_distribution": dict(payout_source_distribution),
                "zero_hit_rounds": zero_hit_rounds,
                "paying_high_rounds": paying_high_rounds,
                "dead_zone_rounds": dead_zone_rounds,
                "paying_rounds": paying_rounds,
                "paying_round_rate": round(paying_round_rate, 4),
                "dead_zone_rate": round(dead_zone_rate, 4),
            }

        def combo_quality(selected_numbers):
            rows = [number_row(number) for number in selected_numbers]
            columns = [number_column(number) for number in selected_numbers]
            zones = [zone_for_number(number) for number in selected_numbers]
            positions = [(number_row(number), number_column(number)) for number in selected_numbers]

            distances = []
            neighbor_pairs = 0

            for pos_i in range(len(positions)):
                for pos_j in range(pos_i + 1, len(positions)):
                    row_a, col_a = positions[pos_i]
                    row_b, col_b = positions[pos_j]
                    distance = sqrt((row_a - row_b) ** 2 + (col_a - col_b) ** 2)
                    distances.append(distance)

                    if abs(row_a - row_b) <= 1 and abs(col_a - col_b) <= 1:
                        neighbor_pairs += 1

            row_counts = Counter(rows)
            column_counts = Counter(columns)
            zone_counts = Counter(zones)
            max_pairs = (len(positions) * (len(positions) - 1)) / 2

            return {
                "unique_rows": len(set(rows)),
                "unique_columns": len(set(columns)),
                "row_entropy": round(entropy_from_counts(list(row_counts.values())), 4),
                "column_entropy": round(entropy_from_counts(list(column_counts.values())), 4),
                "zone_entropy": round(entropy_from_counts(list(zone_counts.values())), 4),
                "neighbor_pairs": int(neighbor_pairs),
                "neighbor_density": round(neighbor_pairs / max_pairs if max_pairs else 0.0, 4),
                "avg_pair_distance": round(float(np.mean(distances)) if distances else 0.0, 4),
                "row_counts": dict(row_counts),
                "column_counts": dict(column_counts),
                "zone_counts": dict(zone_counts),
            }

        def component_features_for_number(current_index, number):
            row = number_row(number)
            column = number_column(number)

            return {
                "hot_last_10": int(count_in_window(current_index, number, 10)),
                "hot_last_20": int(count_in_window(current_index, number, 20)),
                "gap": int(gap_since_seen(current_index, number)),
                "row_pattern_last_10": recent_from_prefix(board_row_events_prefix, current_index, row, 10),
                "column_pattern_last_10": recent_from_prefix(board_column_events_prefix, current_index, column, 10),
                "board_hit_last_10": recent_from_prefix(board_number_hits_prefix, current_index, number, 10),
                "shape_hit_last_10": recent_from_prefix(shape_number_hits_prefix, current_index, number, 10),
                "shape_area_last_10": recent_from_prefix(shape_number_area_prefix, current_index, number, 10),
                "shape_center_last_20": recent_from_prefix(shape_centers_prefix, current_index, number, 20),
                "movement_target_last_20": recent_from_prefix(movement_target_centers_prefix, current_index, number, 20),
                "movement_source_last_20": recent_from_prefix(movement_source_centers_completed_prefix, current_index, number, 20),
            }

        def explain_number(current_index, number, probability):
            features = component_features_for_number(current_index, number)
            components = []
            reasons = []

            if features["shape_area_last_10"] > 0 or features["shape_hit_last_10"] > 0:
                components.append("shape")
                reasons.append(f"shape area/hit {features['shape_area_last_10']}/{features['shape_hit_last_10']}")

            if features["shape_center_last_20"] > 0:
                components.append("shape_center")
                reasons.append(f"shape center x{features['shape_center_last_20']}")

            if features["movement_target_last_20"] > 0:
                components.append("movement_target")
                reasons.append(f"movement target x{features['movement_target_last_20']}")

            if features["movement_source_last_20"] > 0:
                components.append("movement_source")
                reasons.append(f"movement source x{features['movement_source_last_20']}")

            if (
                features["row_pattern_last_10"] > 0
                or features["column_pattern_last_10"] > 0
                or features["board_hit_last_10"] > 0
            ):
                components.append("board_pattern")
                reasons.append(
                    f"board pressure {features['row_pattern_last_10']}/"
                    f"{features['column_pattern_last_10']}/"
                    f"{features['board_hit_last_10']}"
                )

            if probability > baseline_target_probability * 1.2:
                components.append("high_probability")
                reasons.append("above calibrated baseline")

            if features["hot_last_10"] >= 4:
                components.append("hot")
                reasons.append(f"hot last 10: {features['hot_last_10']}")

            if features["gap"] >= 15:
                components.append("cold_gap")
                reasons.append(f"gap {features['gap']}")

            if not components:
                components.append("filler")
                reasons.append("no strong component")

            return {
                "number": int(number),
                "components": components,
                "reasons": reasons,
                "features": features,
                "probability": round(float(probability), 6),
                "probability_percent": round(float(probability) * 100, 4),
            }

        def confidence_score_for_combo(
            current_index,
            selected_numbers,
            selected_probabilities,
            selected_mode,
            regime,
            explanations,
            regime_train_summary,
        ):
            quality = combo_quality(selected_numbers)
            component_counter = Counter()

            for explanation in explanations:
                for component in explanation["components"]:
                    component_counter[component] += 1

            avg_probability = float(np.mean(selected_probabilities))
            max_probability = float(np.max(selected_probabilities))
            empirical_lift = empirical_probability_lift(avg_probability)

            score = 50.0
            reasons = []

            if empirical_lift > 0:
                score += min(15.0, empirical_lift * 200)
                reasons.append(f"positive train calibration lift {empirical_lift:.4f}")
            else:
                score += max(-15.0, empirical_lift * 200)
                reasons.append(f"weak/negative calibration lift {empirical_lift:.4f}")

            if avg_probability > baseline_target_probability:
                score += 7
                reasons.append("average probability above target baseline")

            if max_probability > baseline_target_probability * 1.35:
                score += 5
                reasons.append("at least one very strong number")

            strong_components = (
                component_counter["shape"]
                + component_counter["shape_center"]
                + component_counter["movement_target"]
                + component_counter["board_pattern"]
            )

            if strong_components >= 6:
                score += 10
                reasons.append("many strong component numbers")

            if component_counter["filler"] >= 5:
                score -= 15
                reasons.append("too many filler numbers")

            if quality["unique_rows"] >= 6 and quality["unique_columns"] >= 8:
                score += 5
                reasons.append("healthy board coverage")

            if quality["neighbor_density"] > 0.12 and selected_mode in ["spread", "miss"]:
                score -= 8
                reasons.append("too clustered for spread/miss mode")

            if quality["neighbor_density"] > 0.12 and selected_mode in ["raw", "hybrid"]:
                score += 4
                reasons.append("clustered combo fits hit-seeking mode")

            operation = operation_labels[current_index]
            streak = operation_streak_length[current_index]

            if operation == "heavy_pattern" and selected_mode in ["raw", "hybrid"]:
                score += 6
                reasons.append("heavy pattern operation supports hit-seeking")

            if operation == "scatter_spread" and selected_mode in ["spread", "miss"]:
                score += 6
                reasons.append("scatter operation supports spread/miss")

            if regime_train_summary.get(regime, {}).get("best_mode") == selected_mode:
                score += 8
                reasons.append("mode matches ROI winner for regime")

            score = max(0.0, min(100.0, score))

            if score >= confidence_play_threshold:
                decision = "PLAY"
            elif score >= confidence_watch_threshold:
                decision = "WATCH"
            else:
                decision = "SKIP"

            return {
                "confidence_score": round(score, 4),
                "decision": decision,
                "reasons": reasons,
                "avg_probability": round(avg_probability, 6),
                "max_probability": round(max_probability, 6),
                "empirical_lift": round(empirical_lift, 6),
                "component_counts": dict(component_counter),
                "combo_quality": quality,
                "operation": operation,
                "operation_streak": int(streak),
                "regime": regime,
                "selected_mode": selected_mode,
            }

        def rescue_bucket(value, thresholds, labels):
            for threshold, label in zip(thresholds, labels):
                if value <= threshold:
                    return label

            return labels[-1]

        def build_rescue_context_key(current_index, selected_mode, selected_numbers, explanations=None):
            """
            Context used to learn whether rescue_1 helps or hurts.

            This uses only information available before the future rounds:
            operation, regime, zone, basket quality, and component mix.
            """
            quality = combo_quality(selected_numbers)
            component_counter = Counter()

            for explanation in explanations or []:
                for component in explanation.get("components", []):
                    component_counter[component] += 1

            filler_count = component_counter.get("filler", 0)
            strong_count = (
                component_counter.get("shape", 0)
                + component_counter.get("shape_center", 0)
                + component_counter.get("movement_target", 0)
                + component_counter.get("board_pattern", 0)
            )

            neighbor_bucket = rescue_bucket(
                quality.get("neighbor_density", 0.0),
                thresholds=[0.04, 0.08, 0.12],
                labels=["neighbor_low", "neighbor_medium", "neighbor_high", "neighbor_very_high"],
            )

            entropy_bucket = rescue_bucket(
                quality.get("zone_entropy", 0.0),
                thresholds=[1.5, 2.0, 2.5],
                labels=["entropy_low", "entropy_medium", "entropy_high", "entropy_very_high"],
            )

            filler_bucket = rescue_bucket(
                filler_count,
                thresholds=[0, 2, 4],
                labels=["filler_0", "filler_1_2", "filler_3_4", "filler_5_plus"],
            )

            strong_bucket = rescue_bucket(
                strong_count,
                thresholds=[3, 6, 9],
                labels=["strong_low", "strong_medium", "strong_high", "strong_very_high"],
            )

            regime = decision_regime_by_index.get(current_index, "normal_pattern")

            return (
                regime,
                operation_labels[current_index],
                zone_labels[current_index],
                selected_mode,
                neighbor_bucket,
                entropy_bucket,
                filler_bucket,
                strong_bucket,
            )

        def build_rescue_coarse_key(current_index, selected_mode):
            regime = decision_regime_by_index.get(current_index, "normal_pattern")
            return (
                regime,
                operation_labels[current_index],
                zone_labels[current_index],
                selected_mode,
            )

        def update_rescue_stats(stats, key, profit_delta, hit_delta):
            item = stats[key]
            item["count"] += 1
            item["profit_delta"] += float(profit_delta)
            item["hit_delta"] += int(hit_delta)

            if profit_delta > 0:
                item["improved"] += 1
            elif profit_delta < 0:
                item["worsened"] += 1
            else:
                item["same"] += 1

        def summarize_rescue_stats(stats, min_count=5, limit=40):
            rows = []

            for key, item in stats.items():
                count = item.get("count", 0)

                if count < min_count:
                    continue

                avg_profit_delta = item.get("profit_delta", 0.0) / count
                avg_hit_delta = item.get("hit_delta", 0) / count
                improved_rate = (item.get("improved", 0) / count) * 100
                worsened_rate = (item.get("worsened", 0) / count) * 100

                rows.append(
                    {
                        "key": " | ".join(str(part) for part in key),
                        "count": count,
                        "avg_profit_delta": round(float(avg_profit_delta), 4),
                        "total_profit_delta": round(float(item.get("profit_delta", 0.0)), 2),
                        "avg_hit_delta": round(float(avg_hit_delta), 4),
                        "improved": item.get("improved", 0),
                        "worsened": item.get("worsened", 0),
                        "same": item.get("same", 0),
                        "improved_rate": round(float(improved_rate), 4),
                        "worsened_rate": round(float(worsened_rate), 4),
                    }
                )

            return sorted(
                rows,
                key=lambda row: (row["avg_profit_delta"], row["count"]),
                reverse=True,
            )[:limit]

        def should_apply_smart_rescue(current_index, selected_mode, selected_numbers, explanations):
            """
            Conditional rescue gate.

            Exact context has priority. If exact context has too little training data,
            we fall back to a coarse regime/operation/zone/mode context.
            """
            exact_key = build_rescue_context_key(
                current_index=current_index,
                selected_mode=selected_mode,
                selected_numbers=selected_numbers,
                explanations=explanations,
            )
            coarse_key = build_rescue_coarse_key(current_index, selected_mode)

            candidates = [
                ("exact", train_rescue_context_stats.get(exact_key), 10, exact_key),
                ("coarse", train_rescue_coarse_stats.get(coarse_key), 25, coarse_key),
            ]

            for source, item, min_count, key in candidates:
                if not item:
                    continue

                count = item.get("count", 0)

                if count < min_count:
                    continue

                avg_profit_delta = item.get("profit_delta", 0.0) / count
                improved_rate = item.get("improved", 0) / count
                worsened_rate = item.get("worsened", 0) / count

                apply_rescue = avg_profit_delta > 0 and improved_rate >= worsened_rate

                return apply_rescue, {
                    "source": source,
                    "key": " | ".join(str(part) for part in key),
                    "count": count,
                    "avg_profit_delta": round(float(avg_profit_delta), 4),
                    "improved_rate": round(float(improved_rate * 100), 4),
                    "worsened_rate": round(float(worsened_rate * 100), 4),
                    "decision": "apply_rescue_1" if apply_rescue else "keep_base",
                }

            return False, {
                "source": "none",
                "key": "no_reliable_context",
                "count": 0,
                "avg_profit_delta": 0.0,
                "improved_rate": 0.0,
                "worsened_rate": 0.0,
                "decision": "keep_base",
            }

        def should_apply_safe_smart_rescue(current_index, selected_mode, selected_numbers, explanations):
            """
            Stricter conditional rescue gate.

            This is deliberately less aggressive than smart_rescue_1.
            It only applies rescue_1 when historical context says rescue usually helps,
            not merely when average profit is slightly positive.
            """
            exact_key = build_rescue_context_key(
                current_index=current_index,
                selected_mode=selected_mode,
                selected_numbers=selected_numbers,
                explanations=explanations,
            )
            coarse_key = build_rescue_coarse_key(current_index, selected_mode)

            candidates = [
                ("exact", train_rescue_context_stats.get(exact_key), 30, exact_key),
                ("coarse", train_rescue_coarse_stats.get(coarse_key), 80, coarse_key),
            ]

            for source, item, min_count, key in candidates:
                if not item:
                    continue

                count = item.get("count", 0)

                if count < min_count:
                    continue

                avg_profit_delta = item.get("profit_delta", 0.0) / count
                improved_rate = item.get("improved", 0) / count
                worsened_rate = item.get("worsened", 0) / count
                improvement_edge = improved_rate - worsened_rate

                # Strict rule:
                # - enough samples
                # - positive average profit delta
                # - improves at least as often as it worsens
                # - small edge buffer so we do not trigger on almost equal contexts
                apply_rescue = (
                    avg_profit_delta > 0
                    and improved_rate >= worsened_rate
                    and improvement_edge >= 0.005
                )

                return apply_rescue, {
                    "source": source,
                    "key": " | ".join(str(part) for part in key),
                    "count": count,
                    "avg_profit_delta": round(float(avg_profit_delta), 4),
                    "improved_rate": round(float(improved_rate * 100), 4),
                    "worsened_rate": round(float(worsened_rate * 100), 4),
                    "improvement_edge": round(float(improvement_edge * 100), 4),
                    "decision": "apply_rescue_1" if apply_rescue else "keep_base",
                }

            return False, {
                "source": "none",
                "key": "no_safe_context",
                "count": 0,
                "avg_profit_delta": 0.0,
                "improved_rate": 0.0,
                "worsened_rate": 0.0,
                "improvement_edge": 0.0,
                "decision": "keep_base",
            }

        train_groups = build_groups(train_draw_indices)
        test_groups = build_groups(test_draw_indices)

        # ------------------------------------------------------------
        # Learn best mode per regime by TRAIN ROI
        # ------------------------------------------------------------

        log_step("Learning best pick mode per regime by ROI...")

        train_mode_hits_by_regime = defaultdict(lambda: defaultdict(list))
        train_mode_roi_by_regime = defaultdict(lambda: defaultdict(list))

        for counter, (draw_index, positions) in enumerate(train_groups.items(), start=1):
            if counter % 500 == 0:
                self.stdout.write(f"  analyzed train ROI {counter:,}/{len(train_groups):,}")

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")
            draw_probs = train_probabilities[positions]
            draw_future_counts = future_counts_train[positions]
            draw_features = X_train[positions]
            numbers = draw_features[:, 0].astype(int)

            for mode in ["raw", "spread", "hybrid", "relaxed_hybrid", "miss"]:
                selected_indices = select_mode_indices(mode, draw_probs, numbers, pick)
                selected_numbers = [int(numbers[index]) for index in selected_indices]

                hits = int(draw_future_counts[selected_indices].sum())
                profit_result = calculate_pick_profit(selected_numbers, draw_index)

                train_mode_hits_by_regime[regime][mode].append(hits)
                train_mode_roi_by_regime[regime][mode].append(profit_result["roi"])

        regime_mode_map = {}
        regime_train_summary = {}

        for regime in ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]:
            mode_hit_averages = {}
            mode_roi_averages = {}

            for mode in ["raw", "spread", "hybrid", "relaxed_hybrid", "miss"]:
                hit_values = train_mode_hits_by_regime[regime][mode]
                roi_values = train_mode_roi_by_regime[regime][mode]
                mode_hit_averages[mode] = float(np.mean(hit_values)) if hit_values else 0.0
                mode_roi_averages[mode] = float(np.mean(roi_values)) if roi_values else -999999.0

            best_mode = max(mode_roi_averages, key=mode_roi_averages.get)
            regime_mode_map[regime] = best_mode

            regime_train_summary[regime] = {
                "best_mode": best_mode,
                "selection_metric": "roi",
                "hit_averages": {
                    mode: round(value, 4)
                    for mode, value in mode_hit_averages.items()
                },
                "roi_averages": {
                    mode: round(value, 4)
                    for mode, value in mode_roi_averages.items()
                },
            }

        log_done("Regime ROI map ready")

        # ------------------------------------------------------------
        # Learn when rescue_1 helps from TRAIN data
        # ------------------------------------------------------------

        log_step("Learning conditional smart_rescue_1 contexts from training data...")

        train_rescue_context_stats = defaultdict(
            lambda: {
                "count": 0,
                "profit_delta": 0.0,
                "hit_delta": 0,
                "improved": 0,
                "worsened": 0,
                "same": 0,
            }
        )
        train_rescue_coarse_stats = defaultdict(
            lambda: {
                "count": 0,
                "profit_delta": 0.0,
                "hit_delta": 0,
                "improved": 0,
                "worsened": 0,
                "same": 0,
            }
        )

        for counter, (draw_index, positions) in enumerate(train_groups.items(), start=1):
            if counter % 500 == 0:
                self.stdout.write(
                    f"  learned rescue contexts for {counter:,}/{len(train_groups):,} train decisions..."
                )

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")
            selected_mode_for_regime = regime_mode_map.get(regime, "hybrid")

            draw_probs = train_probabilities[positions]
            draw_future_counts = future_counts_train[positions]
            draw_features = X_train[positions]
            numbers = draw_features[:, 0].astype(int)

            base_indices = select_mode_indices(selected_mode_for_regime, draw_probs, numbers, pick)
            rescue_indices = select_rescue_pick(base_indices, draw_probs, numbers, pick, rescue_count=1)

            base_numbers = [int(numbers[index]) for index in base_indices]
            rescue_numbers = [int(numbers[index]) for index in rescue_indices]

            base_profit = calculate_pick_profit(base_numbers, draw_index)
            rescue_profit = calculate_pick_profit(rescue_numbers, draw_index)

            base_hits = int(draw_future_counts[base_indices].sum())
            rescue_hits = int(draw_future_counts[rescue_indices].sum())

            profit_delta = rescue_profit.get("profit", 0.0) - base_profit.get("profit", 0.0)
            hit_delta = rescue_hits - base_hits

            base_explanations = [
                explain_number(draw_index, int(numbers[index]), float(draw_probs[index]))
                for index in base_indices
            ]

            exact_key = build_rescue_context_key(
                current_index=draw_index,
                selected_mode=selected_mode_for_regime,
                selected_numbers=base_numbers,
                explanations=base_explanations,
            )
            coarse_key = build_rescue_coarse_key(draw_index, selected_mode_for_regime)

            update_rescue_stats(train_rescue_context_stats, exact_key, profit_delta, hit_delta)
            update_rescue_stats(train_rescue_coarse_stats, coarse_key, profit_delta, hit_delta)

        smart_rescue_context_summary = summarize_rescue_stats(train_rescue_context_stats, min_count=5, limit=40)
        smart_rescue_coarse_summary = summarize_rescue_stats(train_rescue_coarse_stats, min_count=10, limit=40)

        log_done("Conditional rescue contexts learned")

        # ------------------------------------------------------------
        # V8: train swap-model rescue
        # ------------------------------------------------------------

        log_step("Training V8 swap model from all 1-swap candidates...")

        swap_feature_names = [
            "drop_probability",
            "add_probability",
            "probability_delta",
            "drop_rank",
            "add_rank",
            "rank_delta",
            "drop_row",
            "drop_column",
            "add_row",
            "add_column",
            "same_row",
            "same_column",
            "row_distance",
            "column_distance",
            "drop_hot_last_10",
            "add_hot_last_10",
            "hot_delta_last_10",
            "drop_hot_last_20",
            "add_hot_last_20",
            "hot_delta_last_20",
            "drop_gap",
            "add_gap",
            "gap_delta",
            "drop_board_signal",
            "add_board_signal",
            "board_signal_delta",
            "drop_shape_signal",
            "add_shape_signal",
            "shape_signal_delta",
            "drop_movement_signal",
            "add_movement_signal",
            "movement_signal_delta",
            "base_unique_rows",
            "base_unique_columns",
            "base_neighbor_density",
            "base_zone_entropy",
            "new_unique_rows",
            "new_unique_columns",
            "new_neighbor_density",
            "new_zone_entropy",
            "unique_rows_delta",
            "unique_columns_delta",
            "neighbor_density_delta",
            "zone_entropy_delta",
            "draw_pattern_score",
            "draw_spread_score",
            "draw_row_entropy",
            "draw_column_entropy",
            "draw_zone_entropy",
            "draw_neighbor_density",
            "operation_index",
            "zone_index",
            "selected_mode_index",
            "regime_index",
        ]

        swap_mode_names = ["raw", "spread", "hybrid", "relaxed_hybrid", "miss"]
        swap_regime_names = ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]

        def get_rank_map(draw_probs):
            ranked_indices = list(np.argsort(draw_probs)[::-1])
            return {int(local_index): rank for rank, local_index in enumerate(ranked_indices, start=1)}

        def reserve_pool_indices(base_indices, draw_probs, reserve_limit=20):
            ranked_indices = list(np.argsort(draw_probs)[::-1])
            selected_set = set(int(index) for index in base_indices)
            return [
                int(index)
                for index in ranked_indices[:reserve_limit]
                if int(index) not in selected_set
            ]

        def one_swap_indices(base_indices, drop_index, add_index):
            selected = [int(index) for index in base_indices]
            replace_position = selected.index(int(drop_index))
            selected[replace_position] = int(add_index)
            return np.array(selected, dtype=np.int32)

        def number_signal_features(current_index, number):
            features = component_features_for_number(current_index, number)
            board_signal = (
                features.get("row_pattern_last_10", 0)
                + features.get("column_pattern_last_10", 0)
                + features.get("board_hit_last_10", 0)
            )
            shape_signal = (
                features.get("shape_hit_last_10", 0)
                + features.get("shape_area_last_10", 0)
                + features.get("shape_center_last_20", 0)
            )
            movement_signal = (
                features.get("movement_target_last_20", 0)
                + features.get("movement_source_last_20", 0)
            )
            return features, board_signal, shape_signal, movement_signal

        def build_swap_features(
            current_index,
            base_indices,
            drop_index,
            add_index,
            draw_probs,
            numbers,
            selected_mode,
            regime,
            rank_map,
        ):
            drop_number = int(numbers[int(drop_index)])
            add_number = int(numbers[int(add_index)])

            base_numbers = [int(numbers[int(index)]) for index in base_indices]
            new_indices = one_swap_indices(base_indices, drop_index, add_index)
            new_numbers = [int(numbers[int(index)]) for index in new_indices]

            base_quality = combo_quality(base_numbers)
            new_quality = combo_quality(new_numbers)

            drop_features, drop_board, drop_shape, drop_movement = number_signal_features(current_index, drop_number)
            add_features, add_board, add_shape, add_movement = number_signal_features(current_index, add_number)

            drop_probability = float(draw_probs[int(drop_index)])
            add_probability = float(draw_probs[int(add_index)])
            drop_rank = int(rank_map.get(int(drop_index), 80))
            add_rank = int(rank_map.get(int(add_index), 80))

            operation = operation_labels[current_index]
            zone = zone_labels[current_index]

            return [
                drop_probability,
                add_probability,
                add_probability - drop_probability,
                drop_rank,
                add_rank,
                drop_rank - add_rank,
                number_row(drop_number),
                number_column(drop_number),
                number_row(add_number),
                number_column(add_number),
                1 if number_row(drop_number) == number_row(add_number) else 0,
                1 if number_column(drop_number) == number_column(add_number) else 0,
                abs(number_row(drop_number) - number_row(add_number)),
                abs(number_column(drop_number) - number_column(add_number)),
                drop_features.get("hot_last_10", 0),
                add_features.get("hot_last_10", 0),
                add_features.get("hot_last_10", 0) - drop_features.get("hot_last_10", 0),
                drop_features.get("hot_last_20", 0),
                add_features.get("hot_last_20", 0),
                add_features.get("hot_last_20", 0) - drop_features.get("hot_last_20", 0),
                drop_features.get("gap", 0),
                add_features.get("gap", 0),
                add_features.get("gap", 0) - drop_features.get("gap", 0),
                drop_board,
                add_board,
                add_board - drop_board,
                drop_shape,
                add_shape,
                add_shape - drop_shape,
                drop_movement,
                add_movement,
                add_movement - drop_movement,
                base_quality.get("unique_rows", 0),
                base_quality.get("unique_columns", 0),
                base_quality.get("neighbor_density", 0.0),
                base_quality.get("zone_entropy", 0.0),
                new_quality.get("unique_rows", 0),
                new_quality.get("unique_columns", 0),
                new_quality.get("neighbor_density", 0.0),
                new_quality.get("zone_entropy", 0.0),
                new_quality.get("unique_rows", 0) - base_quality.get("unique_rows", 0),
                new_quality.get("unique_columns", 0) - base_quality.get("unique_columns", 0),
                new_quality.get("neighbor_density", 0.0) - base_quality.get("neighbor_density", 0.0),
                new_quality.get("zone_entropy", 0.0) - base_quality.get("zone_entropy", 0.0),
                draw_pattern_score[current_index],
                draw_spread_score[current_index],
                draw_row_entropy[current_index],
                draw_column_entropy[current_index],
                draw_zone_entropy[current_index],
                draw_neighbor_density[current_index],
                operation_names.index(operation) if operation in operation_names else -1,
                zone_names.index(zone) if zone in zone_names else -1,
                swap_mode_names.index(selected_mode) if selected_mode in swap_mode_names else -1,
                swap_regime_names.index(regime) if regime in swap_regime_names else -1,
            ]

        swap_X_rows = []
        swap_y_rows = []
        swap_profit_delta_rows = []
        swap_train_examples = []

        train_group_items = list(train_groups.items())
        if swap_max_train_decisions and swap_max_train_decisions > 0:
            train_group_items = train_group_items[-swap_max_train_decisions:]

        for counter, (draw_index, positions) in enumerate(train_group_items, start=1):
            if counter % 250 == 0:
                self.stdout.write(
                    f"  built swap candidates for {counter:,}/{len(train_group_items):,} train decisions..."
                )

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")
            selected_mode_for_regime = regime_mode_map.get(regime, "hybrid")

            draw_probs = train_probabilities[positions]
            draw_features = X_train[positions]
            numbers = draw_features[:, 0].astype(int)
            rank_map = get_rank_map(draw_probs)

            base_indices = select_mode_indices(selected_mode_for_regime, draw_probs, numbers, pick)
            base_numbers = [int(numbers[index]) for index in base_indices]
            base_profit = calculate_pick_profit(base_numbers, draw_index)

            reserve_indices = reserve_pool_indices(base_indices, draw_probs, reserve_limit=20)

            if not reserve_indices:
                continue

            for drop_index in base_indices:
                for add_index in reserve_indices:
                    swap_indices = one_swap_indices(base_indices, drop_index, add_index)
                    swap_numbers = [int(numbers[index]) for index in swap_indices]
                    swap_profit = calculate_pick_profit(swap_numbers, draw_index)
                    profit_delta = swap_profit.get("profit", 0.0) - base_profit.get("profit", 0.0)

                    features = build_swap_features(
                        current_index=draw_index,
                        base_indices=base_indices,
                        drop_index=drop_index,
                        add_index=add_index,
                        draw_probs=draw_probs,
                        numbers=numbers,
                        selected_mode=selected_mode_for_regime,
                        regime=regime,
                        rank_map=rank_map,
                    )

                    swap_X_rows.append(features)
                    swap_y_rows.append(1 if profit_delta > 0 else 0)
                    swap_profit_delta_rows.append(profit_delta)

                    if profit_delta > 0 and len(swap_train_examples) < 20:
                        swap_train_examples.append(
                            {
                                "draw_id": draw_ids[draw_index],
                                "drop_number": int(numbers[int(drop_index)]),
                                "add_number": int(numbers[int(add_index)]),
                                "drop_rank": rank_map.get(int(drop_index)),
                                "add_rank": rank_map.get(int(add_index)),
                                "profit_delta": round(float(profit_delta), 2),
                                "base_profit": round(float(base_profit.get("profit", 0.0)), 2),
                                "swap_profit": round(float(swap_profit.get("profit", 0.0)), 2),
                            }
                        )

        swap_model_enabled = False
        swap_model = None
        swap_training_summary = {
            "enabled": False,
            "candidate_rows": len(swap_X_rows),
            "positive_rate": 0.0,
            "avg_profit_delta": 0.0,
            "reason": "not_enough_classes",
        }

        if swap_X_rows:
            swap_X = np.array(swap_X_rows, dtype=np.float32)
            swap_y = np.array(swap_y_rows, dtype=np.int8)
            swap_profit_deltas = np.array(swap_profit_delta_rows, dtype=np.float32)
            unique_swap_classes = np.unique(swap_y)

            swap_training_summary = {
                "enabled": False,
                "candidate_rows": int(len(swap_X_rows)),
                "positive_rows": int(swap_y.sum()),
                "negative_rows": int(len(swap_y) - swap_y.sum()),
                "positive_rate": round(float(np.mean(swap_y) * 100), 4),
                "avg_profit_delta": round(float(np.mean(swap_profit_deltas)), 4),
                "median_profit_delta": round(float(np.median(swap_profit_deltas)), 4),
                "best_profit_delta": round(float(np.max(swap_profit_deltas)), 4),
                "worst_profit_delta": round(float(np.min(swap_profit_deltas)), 4),
                "swap_threshold": swap_threshold,
                "train_examples": swap_train_examples,
            }

            if len(unique_swap_classes) >= 2:
                swap_model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=200,
                                solver="saga",
                                n_jobs=-1,
                                verbose=0,
                            ),
                        ),
                    ]
                )
                swap_model.fit(swap_X, swap_y)
                swap_model_enabled = True
                swap_training_summary["enabled"] = True
                swap_training_summary["reason"] = "trained"

        def select_swap_model_pick(base_indices, draw_probs, numbers, pick_size, current_index, selected_mode, regime):
            if not swap_model_enabled or swap_model is None:
                return np.array(base_indices, dtype=np.int32), {
                    "enabled": False,
                    "decision": "keep_base",
                    "reason": "swap_model_not_trained",
                }

            rank_map = get_rank_map(draw_probs)
            reserve_indices = reserve_pool_indices(base_indices, draw_probs, reserve_limit=20)

            if not reserve_indices:
                return np.array(base_indices, dtype=np.int32), {
                    "enabled": True,
                    "decision": "keep_base",
                    "reason": "no_reserve_candidates",
                }

            candidate_features = []
            candidate_meta = []

            for drop_index in base_indices:
                for add_index in reserve_indices:
                    features = build_swap_features(
                        current_index=current_index,
                        base_indices=base_indices,
                        drop_index=drop_index,
                        add_index=add_index,
                        draw_probs=draw_probs,
                        numbers=numbers,
                        selected_mode=selected_mode,
                        regime=regime,
                        rank_map=rank_map,
                    )
                    candidate_features.append(features)
                    candidate_meta.append((int(drop_index), int(add_index)))

            if not candidate_features:
                return np.array(base_indices, dtype=np.int32), {
                    "enabled": True,
                    "decision": "keep_base",
                    "reason": "no_swap_candidates",
                }

            candidate_X = np.array(candidate_features, dtype=np.float32)
            candidate_scores = swap_model.predict_proba(candidate_X)[:, 1]
            best_position = int(np.argmax(candidate_scores))
            best_score = float(candidate_scores[best_position])
            best_drop_index, best_add_index = candidate_meta[best_position]

            top_candidates = []
            for rank_position in np.argsort(candidate_scores)[::-1][:10]:
                drop_index, add_index = candidate_meta[int(rank_position)]
                top_candidates.append(
                    {
                        "drop_number": int(numbers[drop_index]),
                        "add_number": int(numbers[add_index]),
                        "drop_rank": int(rank_map.get(drop_index, 80)),
                        "add_rank": int(rank_map.get(add_index, 80)),
                        "drop_probability": round(float(draw_probs[drop_index]), 6),
                        "add_probability": round(float(draw_probs[add_index]), 6),
                        "predicted_positive_probability": round(float(candidate_scores[int(rank_position)]), 6),
                    }
                )

            if best_score >= swap_threshold:
                swapped_indices = one_swap_indices(base_indices, best_drop_index, best_add_index)
                return swapped_indices, {
                    "enabled": True,
                    "decision": "apply_swap",
                    "predicted_positive_probability": round(best_score, 6),
                    "threshold": swap_threshold,
                    "drop_number": int(numbers[best_drop_index]),
                    "add_number": int(numbers[best_add_index]),
                    "drop_rank": int(rank_map.get(best_drop_index, 80)),
                    "add_rank": int(rank_map.get(best_add_index, 80)),
                    "top_candidates": top_candidates,
                }

            return np.array(base_indices, dtype=np.int32), {
                "enabled": True,
                "decision": "keep_base",
                "predicted_positive_probability": round(best_score, 6),
                "threshold": swap_threshold,
                "drop_number": int(numbers[best_drop_index]),
                "add_number": int(numbers[best_add_index]),
                "drop_rank": int(rank_map.get(best_drop_index, 80)),
                "add_rank": int(rank_map.get(best_add_index, 80)),
                "top_candidates": top_candidates,
            }

        log_done(
            f"V8 swap model ready | enabled={swap_model_enabled} | "
            f"candidates={swap_training_summary.get('candidate_rows', 0):,} | "
            f"positive_rate={swap_training_summary.get('positive_rate', 0)}%"
        )

        # ------------------------------------------------------------
        # Test modes, confidence, audits
        # ------------------------------------------------------------

        log_step("Testing modes with ROI / confidence / audit logic...")

        mode_names = [
            "raw",
            "spread",
            "hybrid",
            "relaxed_hybrid",
            "miss",
            "regime_aware",
            "rescue_1",
            "smart_rescue_1",
            "safe_smart_rescue_1",
            "swap_model_1",
            "rescue_2",
            "rescue_3",
            "random",
        ]
        mode_hit_lists = {mode: [] for mode in mode_names}
        mode_profit_results = {mode: [] for mode in mode_names}

        regime_test_summary = defaultdict(lambda: defaultdict(list))
        confidence_audits = []

        rng = np.random.default_rng(seed=42)
        unique_test_draw_indices = sorted(test_groups.keys())

        for counter, draw_index in enumerate(unique_test_draw_indices, start=1):
            if counter % 250 == 0:
                self.stdout.write(f"  tested {counter:,}/{len(unique_test_draw_indices):,}")

            positions = test_groups[draw_index]
            regime = decision_regime_by_index.get(draw_index, "normal_pattern")
            selected_mode_for_regime = regime_mode_map.get(regime, "hybrid")

            draw_probs = test_probabilities[positions]
            draw_future_counts = future_counts_test[positions]
            draw_features = X_test[positions]
            numbers = draw_features[:, 0].astype(int)

            base_regime_indices = select_mode_indices(
                selected_mode_for_regime,
                draw_probs,
                numbers,
                pick,
            )

            base_regime_numbers = [int(numbers[index]) for index in base_regime_indices]
            base_regime_explanations = [
                explain_number(draw_index, int(numbers[index]), float(draw_probs[index]))
                for index in base_regime_indices
            ]

            should_rescue, smart_rescue_info = should_apply_smart_rescue(
                current_index=draw_index,
                selected_mode=selected_mode_for_regime,
                selected_numbers=base_regime_numbers,
                explanations=base_regime_explanations,
            )
            safe_should_rescue, safe_smart_rescue_info = should_apply_safe_smart_rescue(
                current_index=draw_index,
                selected_mode=selected_mode_for_regime,
                selected_numbers=base_regime_numbers,
                explanations=base_regime_explanations,
            )

            rescue_1_indices = select_rescue_pick(base_regime_indices, draw_probs, numbers, pick, rescue_count=1)
            smart_rescue_1_indices = rescue_1_indices if should_rescue else base_regime_indices
            safe_smart_rescue_1_indices = rescue_1_indices if safe_should_rescue else base_regime_indices
            swap_model_1_indices, swap_model_1_info = select_swap_model_pick(
                base_indices=base_regime_indices,
                draw_probs=draw_probs,
                numbers=numbers,
                pick_size=pick,
                current_index=draw_index,
                selected_mode=selected_mode_for_regime,
                regime=regime,
            )

            mode_indices = {
                "raw": select_raw_pick(draw_probs, numbers, pick),
                "spread": select_spread_pick(draw_probs, numbers, pick),
                "hybrid": select_hybrid_pick(draw_probs, numbers, pick),
                "relaxed_hybrid": select_relaxed_hybrid_pick(draw_probs, numbers, pick),
                "miss": select_low_pick(draw_probs, numbers, pick),
                "regime_aware": base_regime_indices,
                "rescue_1": rescue_1_indices,
                "smart_rescue_1": smart_rescue_1_indices,
                "safe_smart_rescue_1": safe_smart_rescue_1_indices,
                "swap_model_1": swap_model_1_indices,
                "rescue_2": select_rescue_pick(base_regime_indices, draw_probs, numbers, pick, rescue_count=2),
                "rescue_3": select_rescue_pick(base_regime_indices, draw_probs, numbers, pick, rescue_count=3),
                "random": rng.choice(len(draw_features), size=pick, replace=False),
            }

            for mode, indices in mode_indices.items():
                selected_numbers = [int(numbers[index]) for index in indices]
                hits = int(draw_future_counts[indices].sum())
                profit_result = calculate_pick_profit(selected_numbers, draw_index)

                mode_hit_lists[mode].append(hits)
                mode_profit_results[mode].append(profit_result)

            regime_indices = mode_indices["regime_aware"]
            regime_selected_numbers = [int(numbers[index]) for index in regime_indices]

            explanations = [
                explain_number(draw_index, int(numbers[index]), float(draw_probs[index]))
                for index in regime_indices
            ]

            confidence = confidence_score_for_combo(
                current_index=draw_index,
                selected_numbers=regime_selected_numbers,
                selected_probabilities=[float(draw_probs[index]) for index in regime_indices],
                selected_mode=selected_mode_for_regime,
                regime=regime,
                explanations=explanations,
                regime_train_summary=regime_train_summary,
            )

            # Used only internally for the near-miss analyzer. We do not expose
            # the full 80-number maps directly in the frontend, because that
            # would make the saved JSON too noisy.
            ranked_local_indices = np.argsort(draw_probs)[::-1]
            number_rank_map = {
                int(numbers[local_index]): int(rank)
                for rank, local_index in enumerate(ranked_local_indices, start=1)
            }
            number_probability_map = {
                int(numbers[local_index]): round(float(draw_probs[local_index]), 6)
                for local_index in range(len(numbers))
            }

            confidence_audits.append(
                {
                    "draw_id": draw_ids[draw_index],
                    "draw_index": int(draw_index),
                    "selected_numbers": regime_selected_numbers,
                    "selected_mode": selected_mode_for_regime,
                    "regime": regime,
                    "operation": operation_labels[draw_index],
                    "zone": zone_labels[draw_index],
                    "confidence_score": confidence["confidence_score"],
                    "confidence_decision": confidence["decision"],
                    "confidence_reasons": confidence["reasons"],
                    "component_counts": confidence["component_counts"],
                    "combo_quality": confidence["combo_quality"],
                    "number_explanations": explanations,
                    "number_rank_map": number_rank_map,
                    "number_probability_map": number_probability_map,
                    "profit_result": mode_profit_results["regime_aware"][-1],
                    "swap_model_1_info": swap_model_1_info,
                }
            )

            for mode in ["raw", "spread", "hybrid", "relaxed_hybrid", "miss", "regime_aware", "rescue_1", "smart_rescue_1", "safe_smart_rescue_1", "swap_model_1", "rescue_2", "rescue_3"]:
                regime_test_summary[regime][mode].append(mode_hit_lists[mode][-1])

        log_done("Testing complete")

        # ------------------------------------------------------------
        # Confidence summaries
        # ------------------------------------------------------------

        def summarize_confidence(results):
            thresholds = [50, 55, 60, 65, 70, 75, 80, 85]
            output = {}

            for threshold in thresholds:
                filtered = [
                    item["profit_result"]
                    for item in results
                    if item["confidence_score"] >= threshold
                ]

                if filtered:
                    summary = summarize_profit(filtered)
                    output[str(threshold)] = {
                        "played_decisions": len(filtered),
                        "skipped_decisions": len(results) - len(filtered),
                        "roi": summary["roi"],
                        "profit": summary["total_profit"],
                        "cost": summary["total_cost"],
                        "return": summary["total_return"],
                        "paying_round_rate": summary["paying_round_rate"],
                        "dead_zone_rate": summary["dead_zone_rate"],
                        "hit_distribution": summary["hit_distribution"],
                    }
                else:
                    output[str(threshold)] = {
                        "played_decisions": 0,
                        "skipped_decisions": len(results),
                        "roi": None,
                        "profit": None,
                    }

            buckets = {
                "0_49": [],
                "50_59": [],
                "60_69": [],
                "70_79": [],
                "80_100": [],
            }

            for item in results:
                score = item["confidence_score"]

                if score < 50:
                    buckets["0_49"].append(item["profit_result"])
                elif score < 60:
                    buckets["50_59"].append(item["profit_result"])
                elif score < 70:
                    buckets["60_69"].append(item["profit_result"])
                elif score < 80:
                    buckets["70_79"].append(item["profit_result"])
                else:
                    buckets["80_100"].append(item["profit_result"])

            bucket_output = {}

            for bucket_name, bucket_results in buckets.items():
                if bucket_results:
                    summary = summarize_profit(bucket_results)
                    bucket_output[bucket_name] = {
                        "decisions": len(bucket_results),
                        "roi": summary["roi"],
                        "profit": summary["total_profit"],
                        "cost": summary["total_cost"],
                        "return": summary["total_return"],
                        "paying_round_rate": summary["paying_round_rate"],
                        "dead_zone_rate": summary["dead_zone_rate"],
                    }
                else:
                    bucket_output[bucket_name] = {
                        "decisions": 0,
                        "roi": None,
                        "profit": None,
                    }

            return {
                "thresholds": output,
                "buckets": bucket_output,
            }

        def compact_audit_item(item):
            result = item["profit_result"]

            return {
                "draw_id": item["draw_id"],
                "draw_index": item["draw_index"],
                "selected_numbers": item["selected_numbers"],
                "selected_mode": item["selected_mode"],
                "regime": item["regime"],
                "operation": item["operation"],
                "zone": item["zone"],
                "confidence_score": item["confidence_score"],
                "confidence_decision": item["confidence_decision"],
                "confidence_reasons": item["confidence_reasons"],
                "component_counts": item["component_counts"],
                "combo_quality": item.get("combo_quality", {}),
                "cost": round(float(result["cost"]), 2),
                "return": round(float(result["return"]), 2),
                "profit": round(float(result["profit"]), 2),
                "roi": round(float(result["roi"]), 4),
                "hit_distribution": result["hit_distribution"],
                "bonus_hit_distribution": result.get("bonus_hit_distribution", {}),
                "payout_source_distribution": result.get("payout_source_distribution", {}),
                "round_details": result.get("round_details", []),
                "number_explanations": item.get("number_explanations", []),
            }

        confidence_summary = summarize_confidence(confidence_audits)

        high_confidence_wins = [
            item for item in confidence_audits
            if item["confidence_score"] >= confidence_play_threshold
            and item["profit_result"]["profit"] > 0
        ]

        high_confidence_losses = [
            item for item in confidence_audits
            if item["confidence_score"] >= confidence_play_threshold
            and item["profit_result"]["profit"] < 0
        ]

        low_confidence_wins = [
            item for item in confidence_audits
            if item["confidence_score"] < confidence_watch_threshold
            and item["profit_result"]["profit"] > 0
        ]

        low_confidence_losses = [
            item for item in confidence_audits
            if item["confidence_score"] < confidence_watch_threshold
            and item["profit_result"]["profit"] < 0
        ]

        audit_examples_by_quality = {
            "high_confidence_wins": [
                compact_audit_item(item)
                for item in sorted(
                    high_confidence_wins,
                    key=lambda x: (x["confidence_score"], x["profit_result"]["profit"]),
                    reverse=True,
                )[:6]
            ],
            "high_confidence_losses": [
                compact_audit_item(item)
                for item in sorted(
                    high_confidence_losses,
                    key=lambda x: (-x["confidence_score"], x["profit_result"]["profit"]),
                )[:6]
            ],
            "low_confidence_wins": [
                compact_audit_item(item)
                for item in sorted(
                    low_confidence_wins,
                    key=lambda x: (x["confidence_score"], -x["profit_result"]["profit"]),
                )[:6]
            ],
            "low_confidence_losses": [
                compact_audit_item(item)
                for item in sorted(
                    low_confidence_losses,
                    key=lambda x: (x["confidence_score"], x["profit_result"]["profit"]),
                )[:6]
            ],
        }


        # ------------------------------------------------------------
        # Near-miss report: why 7/8/9-hit rounds almost worked
        # ------------------------------------------------------------

        def rank_bucket(rank):
            if rank is None:
                return "unknown"
            if rank <= 12:
                return "rank_1_12"
            if rank <= 20:
                return "rank_13_20"
            if rank <= 30:
                return "rank_21_30"
            if rank <= 40:
                return "rank_31_40"
            return "rank_41_80"

        def compact_number_info(number, number_rank_map, number_probability_map):
            return {
                "number": int(number),
                "rank": number_rank_map.get(int(number)),
                "probability": number_probability_map.get(int(number)),
                "probability_percent": round(number_probability_map.get(int(number), 0.0) * 100, 4),
                "row": number_row(int(number)),
                "column": number_column(int(number)),
                "zone": zone_for_number(int(number)),
            }

        def build_near_miss_report(audits, target_hit_counts=(7, 8, 9), max_examples_per_hit_count=8):
            """
            Studies individual future rounds where the selected 12-number combo
            hit 7, 8, or 9 numbers.

            Main question:
            If we got 8/12, were the missing winning numbers ranked close
            to our selected combo, for example ranks 13-20? If yes, selector
            logic can be improved. If no, the model simply did not see them.
            """

            rounds = []
            aggregate = {
                "target_hit_counts": list(target_hit_counts),
                "total_near_miss_rounds": 0,
                "by_hit_count": defaultdict(int),
                "best_missing_rank_buckets": defaultdict(int),
                "best_missing_rank_values": [],
                "recoverable_top_12": 0,
                "recoverable_top_20": 0,
                "recoverable_top_30": 0,
                "recoverable_top_40": 0,
                "selected_miss_rank_values": [],
                "replacement_gain_if_best_1": defaultdict(int),
                "replacement_gain_if_best_2": defaultdict(int),
                "replacement_gain_if_best_3": defaultdict(int),
                "operation_counts": defaultdict(int),
                "regime_counts": defaultdict(int),
                "mode_counts": defaultdict(int),
                "zone_counts": defaultdict(int),
                "combo_quality_buckets": defaultdict(int),
            }

            for audit in audits:
                selected_numbers = set(audit.get("selected_numbers", []))
                number_rank_map = audit.get("number_rank_map", {})
                number_probability_map = audit.get("number_probability_map", {})
                profit_result = audit.get("profit_result", {})
                round_details = profit_result.get("round_details", [])

                for round_detail in round_details:
                    hit_count = int(round_detail.get("hit_count", 0))

                    if hit_count not in target_hit_counts:
                        continue

                    hit_numbers = set(round_detail.get("hit_numbers", []))

                    # We only have hit_numbers saved, not the full 20-number future draw.
                    # Rebuild the future draw from the draw_id.
                    future_draw_id = round_detail.get("future_draw_id")
                    future_draw_index = draw_index_by_id.get(future_draw_id)

                    if future_draw_index is None:
                        continue

                    future_numbers = set(draws[future_draw_index].numbers)
                    selected_missed_numbers = sorted(selected_numbers - hit_numbers)
                    unselected_winning_numbers = sorted(future_numbers - selected_numbers)

                    unselected_winning_infos = [
                        compact_number_info(number, number_rank_map, number_probability_map)
                        for number in unselected_winning_numbers
                    ]
                    unselected_winning_infos = sorted(
                        unselected_winning_infos,
                        key=lambda item: item["rank"] if item["rank"] is not None else 999,
                    )

                    selected_miss_infos = [
                        compact_number_info(number, number_rank_map, number_probability_map)
                        for number in selected_missed_numbers
                    ]
                    selected_miss_infos = sorted(
                        selected_miss_infos,
                        key=lambda item: item["rank"] if item["rank"] is not None else 999,
                        reverse=True,
                    )

                    best_missing = unselected_winning_infos[0] if unselected_winning_infos else None
                    best_missing_rank = best_missing.get("rank") if best_missing else None
                    best_missing_bucket = rank_bucket(best_missing_rank)

                    gain_if_replace_1 = min(hit_count + 1, 12) if len(unselected_winning_infos) >= 1 and len(selected_miss_infos) >= 1 else hit_count
                    gain_if_replace_2 = min(hit_count + 2, 12) if len(unselected_winning_infos) >= 2 and len(selected_miss_infos) >= 2 else gain_if_replace_1
                    gain_if_replace_3 = min(hit_count + 3, 12) if len(unselected_winning_infos) >= 3 and len(selected_miss_infos) >= 3 else gain_if_replace_2

                    quality = audit.get("combo_quality", {})
                    quality_bucket = classify_combo_quality_bucket(quality) if "classify_combo_quality_bucket" in locals() else "unknown"

                    aggregate["total_near_miss_rounds"] += 1
                    aggregate["by_hit_count"][hit_count] += 1
                    aggregate["best_missing_rank_buckets"][best_missing_bucket] += 1
                    aggregate["operation_counts"][audit.get("operation", "unknown")] += 1
                    aggregate["regime_counts"][audit.get("regime", "unknown")] += 1
                    aggregate["mode_counts"][audit.get("selected_mode", "unknown")] += 1
                    aggregate["zone_counts"][audit.get("zone", "unknown")] += 1
                    aggregate["combo_quality_buckets"][quality_bucket] += 1
                    aggregate["replacement_gain_if_best_1"][gain_if_replace_1] += 1
                    aggregate["replacement_gain_if_best_2"][gain_if_replace_2] += 1
                    aggregate["replacement_gain_if_best_3"][gain_if_replace_3] += 1

                    if best_missing_rank is not None:
                        aggregate["best_missing_rank_values"].append(best_missing_rank)

                        if best_missing_rank <= 12:
                            aggregate["recoverable_top_12"] += 1
                        if best_missing_rank <= 20:
                            aggregate["recoverable_top_20"] += 1
                        if best_missing_rank <= 30:
                            aggregate["recoverable_top_30"] += 1
                        if best_missing_rank <= 40:
                            aggregate["recoverable_top_40"] += 1

                    for item in selected_miss_infos:
                        if item.get("rank") is not None:
                            aggregate["selected_miss_rank_values"].append(item["rank"])

                    rounds.append(
                        {
                            "base_draw_id": audit.get("draw_id"),
                            "future_draw_id": future_draw_id,
                            "hit_count": hit_count,
                            "selected_mode": audit.get("selected_mode"),
                            "regime": audit.get("regime"),
                            "operation": audit.get("operation"),
                            "zone": audit.get("zone"),
                            "confidence_score": audit.get("confidence_score"),
                            "combo_quality_bucket": quality_bucket,
                            "selected_numbers": sorted(selected_numbers),
                            "hit_numbers": sorted(hit_numbers),
                            "selected_missed_numbers": selected_miss_infos,
                            "best_unselected_winning_numbers": unselected_winning_infos[:8],
                            "best_missing_rank": best_missing_rank,
                            "best_missing_rank_bucket": best_missing_bucket,
                            "could_reach_9_by_one_swap": hit_count == 8 and len(unselected_winning_infos) >= 1 and len(selected_miss_infos) >= 1,
                            "could_reach_10_by_two_swaps": hit_count == 8 and len(unselected_winning_infos) >= 2 and len(selected_miss_infos) >= 2,
                            "gain_if_replace_best_1": gain_if_replace_1,
                            "gain_if_replace_best_2": gain_if_replace_2,
                            "gain_if_replace_best_3": gain_if_replace_3,
                            "payout": round_detail.get("payout"),
                            "bonus_number": round_detail.get("bonus_number"),
                            "bonus_hit": round_detail.get("bonus_hit"),
                            "payout_source": round_detail.get("payout_source"),
                        }
                    )

            total = aggregate["total_near_miss_rounds"]
            best_missing_ranks = aggregate["best_missing_rank_values"]
            selected_miss_ranks = aggregate["selected_miss_rank_values"]

            examples_by_hit_count = {}
            for hit_count in target_hit_counts:
                hit_rounds = [item for item in rounds if item["hit_count"] == hit_count]
                hit_rounds = sorted(
                    hit_rounds,
                    key=lambda item: (
                        item["best_missing_rank"] if item["best_missing_rank"] is not None else 999,
                        -item["confidence_score"],
                    ),
                )
                examples_by_hit_count[str(hit_count)] = hit_rounds[:max_examples_per_hit_count]

            def rate(value):
                return round((value / total) * 100, 4) if total else 0.0

            return {
                "description": "Near-miss report for 7/8/9-hit rounds. Shows whether missing winning numbers were close in model rank.",
                "total_near_miss_rounds": total,
                "by_hit_count": dict(sorted(aggregate["by_hit_count"].items())),
                "best_missing_rank_buckets": dict(aggregate["best_missing_rank_buckets"]),
                "average_best_missing_rank": round(float(np.mean(best_missing_ranks)), 4) if best_missing_ranks else None,
                "median_best_missing_rank": round(float(np.median(best_missing_ranks)), 4) if best_missing_ranks else None,
                "average_selected_miss_rank": round(float(np.mean(selected_miss_ranks)), 4) if selected_miss_ranks else None,
                "median_selected_miss_rank": round(float(np.median(selected_miss_ranks)), 4) if selected_miss_ranks else None,
                "recoverable_top_12_rate": rate(aggregate["recoverable_top_12"]),
                "recoverable_top_20_rate": rate(aggregate["recoverable_top_20"]),
                "recoverable_top_30_rate": rate(aggregate["recoverable_top_30"]),
                "recoverable_top_40_rate": rate(aggregate["recoverable_top_40"]),
                "replacement_gain_if_best_1": dict(sorted(aggregate["replacement_gain_if_best_1"].items())),
                "replacement_gain_if_best_2": dict(sorted(aggregate["replacement_gain_if_best_2"].items())),
                "replacement_gain_if_best_3": dict(sorted(aggregate["replacement_gain_if_best_3"].items())),
                "operation_counts": dict(aggregate["operation_counts"]),
                "regime_counts": dict(aggregate["regime_counts"]),
                "mode_counts": dict(aggregate["mode_counts"]),
                "zone_counts": dict(aggregate["zone_counts"]),
                "combo_quality_buckets": dict(aggregate["combo_quality_buckets"]),
                "examples_by_hit_count": examples_by_hit_count,
            }

        near_miss_report = build_near_miss_report(confidence_audits)

        def total_hits_from_profit_result(result):
            return sum(
                int(round_detail.get("hit_count", 0))
                for round_detail in result.get("round_details", [])
            )

        def build_rescue_comparison_summary():
            """
            Compares rescue modes against the original regime-aware combo.

            Main question:
            Did rescue_1 / rescue_2 / rescue_3 convert near misses like 8/12
            into 9/12+ without destroying too many other rounds?
            """
            base_results = mode_profit_results.get("regime_aware", [])
            output = {}

            for rescue_mode in ["rescue_1", "smart_rescue_1", "safe_smart_rescue_1", "swap_model_1", "rescue_2", "rescue_3", "relaxed_hybrid"]:
                rescue_results = mode_profit_results.get(rescue_mode, [])

                improved_decisions = 0
                worsened_decisions = 0
                same_decisions = 0
                total_profit_delta = 0.0
                total_hit_delta = 0

                converted_7_to_8_plus = 0
                converted_8_to_9_plus = 0
                converted_8_to_10_plus = 0
                destroyed_6_plus_to_dead_zone = 0
                rescued_dead_zone_to_6_plus = 0

                examples = []

                for decision_index, (base_result, rescue_result) in enumerate(zip(base_results, rescue_results)):
                    base_total_hits = total_hits_from_profit_result(base_result)
                    rescue_total_hits = total_hits_from_profit_result(rescue_result)
                    hit_delta = rescue_total_hits - base_total_hits
                    profit_delta = rescue_result.get("profit", 0.0) - base_result.get("profit", 0.0)

                    total_hit_delta += hit_delta
                    total_profit_delta += profit_delta

                    if profit_delta > 0:
                        improved_decisions += 1
                    elif profit_delta < 0:
                        worsened_decisions += 1
                    else:
                        same_decisions += 1

                    base_rounds = base_result.get("round_details", [])
                    rescue_rounds = rescue_result.get("round_details", [])

                    for base_round, rescue_round in zip(base_rounds, rescue_rounds):
                        base_hits = int(base_round.get("hit_count", 0))
                        rescue_hits = int(rescue_round.get("hit_count", 0))

                        if base_hits == 7 and rescue_hits >= 8:
                            converted_7_to_8_plus += 1

                        if base_hits == 8 and rescue_hits >= 9:
                            converted_8_to_9_plus += 1

                        if base_hits == 8 and rescue_hits >= 10:
                            converted_8_to_10_plus += 1

                        if base_hits >= 6 and 1 <= rescue_hits <= 5:
                            destroyed_6_plus_to_dead_zone += 1

                        if 1 <= base_hits <= 5 and rescue_hits >= 6:
                            rescued_dead_zone_to_6_plus += 1

                    if profit_delta > 0 and len(examples) < 6:
                        examples.append(
                            {
                                "decision_number": decision_index + 1,
                                "profit_delta": round(float(profit_delta), 2),
                                "hit_delta": int(hit_delta),
                                "base_profit": round(float(base_result.get("profit", 0.0)), 2),
                                "rescue_profit": round(float(rescue_result.get("profit", 0.0)), 2),
                                "base_hit_distribution": base_result.get("hit_distribution", {}),
                                "rescue_hit_distribution": rescue_result.get("hit_distribution", {}),
                            }
                        )

                total_decisions = len(base_results)
                output[rescue_mode] = {
                    "total_decisions": total_decisions,
                    "improved_decisions": improved_decisions,
                    "worsened_decisions": worsened_decisions,
                    "same_decisions": same_decisions,
                    "improved_rate": round((improved_decisions / total_decisions) * 100, 4) if total_decisions else 0.0,
                    "worsened_rate": round((worsened_decisions / total_decisions) * 100, 4) if total_decisions else 0.0,
                    "total_profit_delta": round(float(total_profit_delta), 2),
                    "total_hit_delta": int(total_hit_delta),
                    "converted_7_to_8_plus": converted_7_to_8_plus,
                    "converted_8_to_9_plus": converted_8_to_9_plus,
                    "converted_8_to_10_plus": converted_8_to_10_plus,
                    "destroyed_6_plus_to_dead_zone": destroyed_6_plus_to_dead_zone,
                    "rescued_dead_zone_to_6_plus": rescued_dead_zone_to_6_plus,
                    "examples": examples,
                }

            return output

        rescue_comparison_summary = build_rescue_comparison_summary()

        # ------------------------------------------------------------
        # Main summaries
        # ------------------------------------------------------------

        mode_average_hits = {
            mode: float(np.mean(values)) if values else 0.0
            for mode, values in mode_hit_lists.items()
        }

        theoretical_baseline = pick * horizon * 0.25

        mode_lifts = {
            mode: mode_average_hits[mode] - theoretical_baseline
            for mode in mode_average_hits
        }

        mode_profit_summaries = {
            mode: summarize_profit(results)
            for mode, results in mode_profit_results.items()
        }

        best_hit_mode = max(mode_average_hits, key=mode_average_hits.get)
        best_roi_mode = max(
            mode_profit_summaries,
            key=lambda mode: mode_profit_summaries[mode]["roi"],
        )

        regime_test_output = {}

        for regime in ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]:
            regime_test_output[regime] = {}

            for mode in ["raw", "spread", "hybrid", "relaxed_hybrid", "miss", "regime_aware", "rescue_1", "smart_rescue_1", "safe_smart_rescue_1", "swap_model_1", "rescue_2", "rescue_3"]:
                values = regime_test_summary[regime][mode]
                regime_test_output[regime][mode] = round(float(np.mean(values)), 4) if values else None

            regime_test_output[regime]["selected_mode"] = regime_mode_map.get(regime)

        selected_history = mode_hit_lists[best_hit_mode][-200:]

        walk_forward_slices = []
        test_draws_sorted = sorted(test_groups.keys())

        if test_draws_sorted:
            chunks = np.array_split(np.array(test_draws_sorted), 5)
            profit_by_draw = {
                draw_index: mode_profit_results["regime_aware"][position]
                for position, draw_index in enumerate(test_draws_sorted)
            }

            for slice_index, chunk in enumerate(chunks, start=1):
                chunk_list = [int(x) for x in chunk.tolist()]
                chunk_results = [
                    profit_by_draw[index]
                    for index in chunk_list
                    if index in profit_by_draw
                ]

                if not chunk_results:
                    continue

                summary = summarize_profit(chunk_results)

                walk_forward_slices.append(
                    {
                        "slice": slice_index,
                        "start_draw_id": draw_ids[chunk_list[0]],
                        "end_draw_id": draw_ids[chunk_list[-1]],
                        "decisions": len(chunk_results),
                        "roi": summary["roi"],
                        "profit": summary["total_profit"],
                        "cost": summary["total_cost"],
                        "return": summary["total_return"],
                        "hit_distribution": summary["hit_distribution"],
                    }
                )

        # ------------------------------------------------------------
        # V7 expanded analysis summaries
        # ------------------------------------------------------------

        analysis_checklist = {
            "calibration_analysis": True,
            "walk_forward_stability_slices": True,
            "roi_first_mode_selection": True,
            "combo_correlation_basket_quality": True,
            "entropy_dispersion_features": True,
            "operation_transition_analysis": True,
            "loss_win_audit_examples_confidence_gate": True,
        }

        def summarize_calibration_rows(rows):
            if not rows:
                return {
                    "available": False,
                    "message": "No calibration rows available.",
                }

            top_bucket = rows[-1]
            bottom_bucket = rows[0]
            best_bucket = max(rows, key=lambda item: item["actual_rate"])
            worst_bucket = min(rows, key=lambda item: item["actual_rate"])

            monotonic_steps = 0
            checked_steps = 0

            for previous, current in zip(rows, rows[1:]):
                checked_steps += 1
                if current["actual_rate"] >= previous["actual_rate"]:
                    monotonic_steps += 1

            monotonic_score = (
                monotonic_steps / checked_steps
                if checked_steps > 0
                else 0.0
            )

            return {
                "available": True,
                "bucket_count": len(rows),
                "top_bucket": top_bucket,
                "bottom_bucket": bottom_bucket,
                "best_bucket": best_bucket,
                "worst_bucket": worst_bucket,
                "top_vs_bottom_actual_rate_lift": round(
                    top_bucket["actual_rate"] - bottom_bucket["actual_rate"],
                    6,
                ),
                "top_vs_baseline_lift": round(
                    top_bucket["actual_rate"] - baseline_target_probability,
                    6,
                ),
                "monotonic_score": round(monotonic_score, 4),
                "interpretation": (
                    "good" if monotonic_score >= 0.70 and top_bucket["actual_rate"] > bottom_bucket["actual_rate"]
                    else "weak_or_noisy"
                ),
            }

        calibration_analysis_summary = {
            "target_meaning": f"Number appears at least {target_hits} times in next {horizon} draws.",
            "baseline_target_probability": round(baseline_target_probability, 6),
            "train": summarize_calibration_rows(train_calibration),
            "test": summarize_calibration_rows(test_calibration),
        }

        def build_walk_forward_mode_slices():
            output = {}
            stability = {}
            sorted_test_draws = sorted(test_groups.keys())

            if not sorted_test_draws:
                return output, stability

            chunks = np.array_split(np.array(sorted_test_draws), 5)

            for mode in mode_names:
                mode_rows = []
                profit_by_draw_for_mode = {
                    draw_index: mode_profit_results[mode][position]
                    for position, draw_index in enumerate(sorted_test_draws)
                }

                for slice_index, chunk in enumerate(chunks, start=1):
                    chunk_list = [int(value) for value in chunk.tolist()]
                    chunk_results = [
                        profit_by_draw_for_mode[index]
                        for index in chunk_list
                        if index in profit_by_draw_for_mode
                    ]

                    if not chunk_results:
                        continue

                    summary = summarize_profit(chunk_results)
                    mode_rows.append(
                        {
                            "slice": slice_index,
                            "start_draw_id": draw_ids[chunk_list[0]],
                            "end_draw_id": draw_ids[chunk_list[-1]],
                            "decisions": len(chunk_results),
                            "roi": summary["roi"],
                            "profit": summary["total_profit"],
                            "cost": summary["total_cost"],
                            "return": summary["total_return"],
                            "paying_round_rate": summary["paying_round_rate"],
                            "dead_zone_rate": summary["dead_zone_rate"],
                            "hit_distribution": summary["hit_distribution"],
                        }
                    )

                output[mode] = mode_rows

                roi_values = [row["roi"] for row in mode_rows]
                stability[mode] = {
                    "slice_count": len(mode_rows),
                    "positive_slices": sum(1 for value in roi_values if value > 0),
                    "negative_slices": sum(1 for value in roi_values if value < 0),
                    "average_roi": round(float(np.mean(roi_values)), 4) if roi_values else None,
                    "min_roi": round(float(np.min(roi_values)), 4) if roi_values else None,
                    "max_roi": round(float(np.max(roi_values)), 4) if roi_values else None,
                    "roi_std": round(float(np.std(roi_values)), 4) if roi_values else None,
                    "stable_positive": bool(roi_values and sum(1 for value in roi_values if value > 0) >= 4),
                }

            return output, stability

        walk_forward_mode_slices, walk_forward_stability_summary = build_walk_forward_mode_slices()

        roi_first_mode_selection_summary = {
            "selection_metric": "train_roi",
            "global_best_roi_mode": best_roi_mode,
            "global_best_hit_mode": best_hit_mode,
            "global_mode_roi": {
                mode: mode_profit_summaries[mode]["roi"]
                for mode in mode_names
            },
            "global_mode_profit": {
                mode: mode_profit_summaries[mode]["total_profit"]
                for mode in mode_names
            },
            "global_mode_average_hits": mode_average_hits,
            "regime_mode_map": regime_mode_map,
            "regime_train_summary": regime_train_summary,
            "note": "Regime-aware mode is selected by ROI on the train period, not by average hits.",
        }

        def classify_combo_quality_bucket(quality):
            neighbor_density = quality.get("neighbor_density", 0.0)
            unique_rows = quality.get("unique_rows", 0)
            unique_columns = quality.get("unique_columns", 0)
            zone_entropy = quality.get("zone_entropy", 0.0)

            if neighbor_density >= 0.14:
                return "clustered"

            if unique_rows >= 6 and unique_columns >= 8 and neighbor_density <= 0.08:
                return "wide_spread"

            if zone_entropy >= 2.0:
                return "multi_zone"

            return "balanced"

        def summarize_audits_by_key(items, key_func):
            grouped = defaultdict(list)

            for item in items:
                grouped[key_func(item)].append(item["profit_result"])

            output = {}

            for key, values in grouped.items():
                summary = summarize_profit(values)
                output[str(key)] = {
                    "decisions": len(values),
                    "roi": summary["roi"],
                    "profit": summary["total_profit"],
                    "cost": summary["total_cost"],
                    "return": summary["total_return"],
                    "paying_round_rate": summary["paying_round_rate"],
                    "dead_zone_rate": summary["dead_zone_rate"],
                    "hit_distribution": summary["hit_distribution"],
                }

            return dict(sorted(output.items()))

        combo_quality_bucket_summary = summarize_audits_by_key(
            confidence_audits,
            lambda item: classify_combo_quality_bucket(item.get("combo_quality", {})),
        )

        entropy_dispersion_summary = {
            "by_combo_quality_bucket": combo_quality_bucket_summary,
            "by_operation": summarize_audits_by_key(
                confidence_audits,
                lambda item: item.get("operation", "unknown"),
            ),
            "by_zone": summarize_audits_by_key(
                confidence_audits,
                lambda item: item.get("zone", "unknown"),
            ),
            "latest_entropy_state": {
                "row_entropy": round(float(draw_row_entropy[len(draws) - 1]), 4),
                "column_entropy": round(float(draw_column_entropy[len(draws) - 1]), 4),
                "zone_entropy": round(float(draw_zone_entropy[len(draws) - 1]), 4),
                "neighbor_density": round(float(draw_neighbor_density[len(draws) - 1]), 4),
                "avg_pair_distance": round(float(draw_avg_pair_distance[len(draws) - 1]), 4),
            },
        }

        operation_context_summary = summarize_audits_by_key(
            confidence_audits,
            lambda item: f"{item.get('operation', 'unknown')}|{item.get('regime', 'unknown')}|{item.get('selected_mode', 'unknown')}",
        )

        def transition_probability_summary(counter, top_n=40):
            totals_by_from = Counter()

            for transition_key, count in counter.items():
                from_state, _to_state = transition_key
                totals_by_from[from_state] += count

            rows = []

            for (from_state, to_state), count in counter.items():
                total = totals_by_from[from_state]
                probability = (count / total) if total else 0.0
                rows.append(
                    {
                        "from": from_state,
                        "to": to_state,
                        "count": count,
                        "from_total": total,
                        "probability": round(probability, 6),
                        "probability_percent": round(probability * 100, 4),
                    }
                )

            rows = sorted(
                rows,
                key=lambda item: (item["from"], -item["probability"], -item["count"]),
            )

            return rows[:top_n]

        operation_transition_probability_summary = transition_probability_summary(transition_counter)
        zone_transition_probability_summary = transition_probability_summary(zone_transition_counter)

        loss_win_audit_summary = {
            "confidence_play_threshold": confidence_play_threshold,
            "confidence_watch_threshold": confidence_watch_threshold,
            "high_confidence_win_count": len(high_confidence_wins),
            "high_confidence_loss_count": len(high_confidence_losses),
            "low_confidence_win_count": len(low_confidence_wins),
            "low_confidence_loss_count": len(low_confidence_losses),
            "audit_examples_saved_per_group": 6,
            "high_confidence_win_rate": round(
                len(high_confidence_wins) / (len(high_confidence_wins) + len(high_confidence_losses)) * 100,
                4,
            ) if (len(high_confidence_wins) + len(high_confidence_losses)) else None,
            "low_confidence_win_rate": round(
                len(low_confidence_wins) / (len(low_confidence_wins) + len(low_confidence_losses)) * 100,
                4,
            ) if (len(low_confidence_wins) + len(low_confidence_losses)) else None,
        }

        coefficients = model.named_steps["model"].coef_[0]

        feature_importance = sorted(
            [
                {
                    "feature": feature_names[index],
                    "group": feature_group_for_name(feature_names[index]),
                    "coefficient": round(float(coef), 6),
                    "absolute_strength": round(abs(float(coef)), 6),
                }
                for index, coef in enumerate(coefficients)
            ],
            key=lambda item: item["absolute_strength"],
            reverse=True,
        )

        feature_group_strength = defaultdict(float)

        for item in feature_importance:
            feature_group_strength[item["group"]] += item["absolute_strength"]

        feature_group_strength = {
            group: round(value, 6)
            for group, value in sorted(
                feature_group_strength.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        # ------------------------------------------------------------
        # Score latest draw
        # ------------------------------------------------------------

        log_step("Scoring latest draw...")

        latest_index = len(draws) - 1
        latest_numbers = draw_sets[latest_index]
        previous_numbers = draw_sets[latest_index - 1]
        latest_operation_features = operation_features(latest_index)

        latest_features = []

        for number in range(1, 81):
            row = number_row(number)
            column = number_column(number)

            count_last_5 = count_in_window(latest_index, number, 5)
            count_last_10 = count_in_window(latest_index, number, 10)
            count_last_20 = count_in_window(latest_index, number, 20)
            count_last_50 = count_in_window(latest_index, number, 50)
            count_last_100 = count_in_window(latest_index, number, 100)

            row_hits_current = len(latest_numbers.intersection(row_numbers(row)))
            column_hits_current = len(latest_numbers.intersection(column_numbers(column)))

            row_features = [
                number,
                row,
                column,
                count_last_5,
                count_last_10,
                count_last_20,
                count_last_50,
                count_last_100,
                count_last_10 / 10,
                count_last_20 / 20,
                count_last_50 / 50,
                gap_since_seen(latest_index, number),
                1 if number in latest_numbers else 0,
                1 if number in previous_numbers else 0,
                row_hits_current,
                column_hits_current,
            ]

            row_features.extend(extra_number_features(latest_index, number))
            row_features.extend(latest_operation_features)

            latest_features.append(row_features)

        latest_X = np.array(latest_features, dtype=np.float32)
        latest_probabilities = model.predict_proba(latest_X)[:, 1]
        latest_numbers_array = np.array(range(1, 81), dtype=np.int32)

        latest_regime_score = current_pattern_pressure_score(latest_index)
        latest_regime = classify_regime(latest_regime_score)
        latest_selected_mode = regime_mode_map.get(latest_regime, "hybrid")

        latest_base_regime_indices = select_mode_indices(
            latest_selected_mode,
            latest_probabilities,
            latest_numbers_array,
            pick,
        )

        latest_base_regime_numbers = [
            int(latest_numbers_array[index])
            for index in latest_base_regime_indices
        ]
        latest_base_regime_explanations_for_rescue = [
            explain_number(latest_index, int(latest_numbers_array[index]), float(latest_probabilities[index]))
            for index in latest_base_regime_indices
        ]
        latest_should_rescue, latest_smart_rescue_info = should_apply_smart_rescue(
            current_index=latest_index,
            selected_mode=latest_selected_mode,
            selected_numbers=latest_base_regime_numbers,
            explanations=latest_base_regime_explanations_for_rescue,
        )
        latest_safe_should_rescue, latest_safe_smart_rescue_info = should_apply_safe_smart_rescue(
            current_index=latest_index,
            selected_mode=latest_selected_mode,
            selected_numbers=latest_base_regime_numbers,
            explanations=latest_base_regime_explanations_for_rescue,
        )
        latest_rescue_1_indices = select_rescue_pick(
            latest_base_regime_indices,
            latest_probabilities,
            latest_numbers_array,
            pick,
            rescue_count=1,
        )
        latest_smart_rescue_1_indices = latest_rescue_1_indices if latest_should_rescue else latest_base_regime_indices
        latest_safe_smart_rescue_1_indices = latest_rescue_1_indices if latest_safe_should_rescue else latest_base_regime_indices
        latest_swap_model_1_indices, latest_swap_model_1_info = select_swap_model_pick(
            base_indices=latest_base_regime_indices,
            draw_probs=latest_probabilities,
            numbers=latest_numbers_array,
            pick_size=pick,
            current_index=latest_index,
            selected_mode=latest_selected_mode,
            regime=latest_regime,
        )

        latest_indices = {
            "raw": select_raw_pick(latest_probabilities, latest_numbers_array, pick),
            "spread": select_spread_pick(latest_probabilities, latest_numbers_array, pick),
            "hybrid": select_hybrid_pick(latest_probabilities, latest_numbers_array, pick),
            "relaxed_hybrid": select_relaxed_hybrid_pick(latest_probabilities, latest_numbers_array, pick),
            "miss": select_low_pick(latest_probabilities, latest_numbers_array, pick),
            "regime_aware": latest_base_regime_indices,
            "rescue_1": latest_rescue_1_indices,
            "smart_rescue_1": latest_smart_rescue_1_indices,
            "safe_smart_rescue_1": latest_safe_smart_rescue_1_indices,
            "swap_model_1": latest_swap_model_1_indices,
            "rescue_2": select_rescue_pick(latest_base_regime_indices, latest_probabilities, latest_numbers_array, pick, rescue_count=2),
            "rescue_3": select_rescue_pick(latest_base_regime_indices, latest_probabilities, latest_numbers_array, pick, rescue_count=3),
        }

        latest_scores = []

        for index, probability in enumerate(latest_probabilities):
            number = index + 1

            latest_scores.append(
                {
                    "number": number,
                    "row": number_row(number),
                    "column": number_column(number),
                    "probability": round(float(probability), 6),
                    "probability_percent": round(float(probability) * 100, 4),
                    "above_baseline": round(
                        (float(probability) - baseline_target_probability) * 100,
                        4,
                    ),
                    "empirical_lift": round(empirical_probability_lift(float(probability)), 6),
                    "count_last_10": int(count_in_window(latest_index, number, 10)),
                    "count_last_20": int(count_in_window(latest_index, number, 20)),
                    "count_last_50": int(count_in_window(latest_index, number, 50)),
                }
            )

        latest_scores = sorted(
            latest_scores,
            key=lambda item: item["probability"],
            reverse=True,
        )

        for rank, item in enumerate(latest_scores, start=1):
            item["rank"] = rank

        score_by_number = {
            item["number"]: item
            for item in latest_scores
        }

        def selected_scores(mode):
            selected_numbers = [
                int(latest_numbers_array[index])
                for index in latest_indices[mode]
            ]

            return [
                score_by_number[number]
                for number in selected_numbers
            ]

        latest_regime_numbers = [
            int(latest_numbers_array[index])
            for index in latest_indices["regime_aware"]
        ]

        latest_explanations = [
            explain_number(
                latest_index,
                int(latest_numbers_array[index]),
                float(latest_probabilities[index]),
            )
            for index in latest_indices["regime_aware"]
        ]

        latest_confidence = confidence_score_for_combo(
            current_index=latest_index,
            selected_numbers=latest_regime_numbers,
            selected_probabilities=[
                float(latest_probabilities[index])
                for index in latest_indices["regime_aware"]
            ],
            selected_mode=latest_selected_mode,
            regime=latest_regime,
            explanations=latest_explanations,
            regime_train_summary=regime_train_summary,
        )

        log_done(
            f"Latest draw scored | operation={operation_labels[latest_index]} "
            f"| regime={latest_regime} | mode={latest_selected_mode} "
            f"| confidence={latest_confidence['confidence_score']}"
        )

        # ------------------------------------------------------------
        # Save result
        # ------------------------------------------------------------

        log_step("Saving AI result...")

        result = KinoAIResult.objects.create(
            model_name="number_ai_v8_swap_model",
            train_draws=len(set(train_draw_indices.tolist())),
            test_draws=len(unique_test_draw_indices),
            baseline_top20_hits=theoretical_baseline,
            model_top20_hits=mode_average_hits[best_hit_mode],
            lift=mode_lifts[best_hit_mode],
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            data={
                "pick": pick,
                "mode": "v8_swap_model_rescue",
                "feature_version": "v8_swap_model_rescue",
                "horizon": horizon,
                "decision_step": decision_step,
                "target_hits": target_hits,
                "baseline_target_probability": round(baseline_target_probability, 6),

                "stake": stake,
                "bonus_fee": bonus_fee,
                "payout_table": payout_table_name,
                "cost_per_round": round(round_cost, 2),
                "cost_per_combo_decision": round(round_cost * horizon, 2),

                "latest_draw_id": draw_ids[-1],
                "split_draw_id": draw_ids[split_draw_index],
                "training_rows": int(len(X_train)),
                "testing_rows": int(len(X_test)),

                "best_hit_mode": best_hit_mode,
                "best_roi_mode": best_roi_mode,
                "best_mode": best_hit_mode,

                "raw_pick_average_hits": mode_average_hits["raw"],
                "spread_pick_average_hits": mode_average_hits["spread"],
                "hybrid_pick_average_hits": mode_average_hits["hybrid"],
                "relaxed_hybrid_pick_average_hits": mode_average_hits["relaxed_hybrid"],
                "miss_pick_average_hits": mode_average_hits["miss"],
                "regime_pick_average_hits": mode_average_hits["regime_aware"],
                "rescue_1_pick_average_hits": mode_average_hits["rescue_1"],
                "smart_rescue_1_pick_average_hits": mode_average_hits["smart_rescue_1"],
                "safe_smart_rescue_1_pick_average_hits": mode_average_hits["safe_smart_rescue_1"],
                "rescue_2_pick_average_hits": mode_average_hits["rescue_2"],
                "rescue_3_pick_average_hits": mode_average_hits["rescue_3"],
                "random_pick_average_hits": mode_average_hits["random"],

                "raw_lift": mode_lifts["raw"],
                "spread_lift": mode_lifts["spread"],
                "hybrid_lift": mode_lifts["hybrid"],
                "relaxed_hybrid_lift": mode_lifts["relaxed_hybrid"],
                "miss_lift": mode_lifts["miss"],
                "regime_lift": mode_lifts["regime_aware"],
                "rescue_1_lift": mode_lifts["rescue_1"],
                "smart_rescue_1_lift": mode_lifts["smart_rescue_1"],
                "safe_smart_rescue_1_lift": mode_lifts["safe_smart_rescue_1"],
                "rescue_2_lift": mode_lifts["rescue_2"],
                "rescue_3_lift": mode_lifts["rescue_3"],
                "random_lift": mode_lifts["random"],

                "raw_profit_summary": mode_profit_summaries["raw"],
                "spread_profit_summary": mode_profit_summaries["spread"],
                "hybrid_profit_summary": mode_profit_summaries["hybrid"],
                "relaxed_hybrid_profit_summary": mode_profit_summaries["relaxed_hybrid"],
                "miss_profit_summary": mode_profit_summaries["miss"],
                "regime_profit_summary": mode_profit_summaries["regime_aware"],
                "rescue_1_profit_summary": mode_profit_summaries["rescue_1"],
                "smart_rescue_1_profit_summary": mode_profit_summaries["smart_rescue_1"],
                "safe_smart_rescue_1_profit_summary": mode_profit_summaries["safe_smart_rescue_1"],
                "swap_model_1_profit_summary": mode_profit_summaries["swap_model_1"],
                "rescue_2_profit_summary": mode_profit_summaries["rescue_2"],
                "rescue_3_profit_summary": mode_profit_summaries["rescue_3"],
                "random_profit_summary": mode_profit_summaries["random"],
                "mode_profit_summaries": mode_profit_summaries,
                "mode_average_hits": mode_average_hits,
                "mode_lifts": mode_lifts,

                "raw_pick_hits_by_test_decision": mode_hit_lists["raw"][-200:],
                "spread_pick_hits_by_test_decision": mode_hit_lists["spread"][-200:],
                "hybrid_pick_hits_by_test_decision": mode_hit_lists["hybrid"][-200:],
                "relaxed_hybrid_pick_hits_by_test_decision": mode_hit_lists["relaxed_hybrid"][-200:],
                "miss_pick_hits_by_test_decision": mode_hit_lists["miss"][-200:],
                "regime_pick_hits_by_test_decision": mode_hit_lists["regime_aware"][-200:],
                "rescue_1_pick_hits_by_test_decision": mode_hit_lists["rescue_1"][-200:],
                "smart_rescue_1_pick_hits_by_test_decision": mode_hit_lists["smart_rescue_1"][-200:],
                "safe_smart_rescue_1_pick_hits_by_test_decision": mode_hit_lists["safe_smart_rescue_1"][-200:],
                "swap_model_1_pick_hits_by_test_decision": mode_hit_lists["swap_model_1"][-200:],
                "rescue_2_pick_hits_by_test_decision": mode_hit_lists["rescue_2"][-200:],
                "rescue_3_pick_hits_by_test_decision": mode_hit_lists["rescue_3"][-200:],
                "random_pick_hits_by_test_decision": mode_hit_lists["random"][-200:],

                "model_pick_hits_by_test_decision": selected_history,
                "model_top20_hits_by_test_decision": selected_history,
                "random_top20_hits_by_test_decision": mode_hit_lists["random"][-200:],
                "random_top20_average_hits": mode_average_hits["random"],

                "regime_thresholds": regime_thresholds,
                "regime_mode_map": regime_mode_map,
                "regime_train_summary": regime_train_summary,
                "regime_test_summary": regime_test_output,

                "operation_names": operation_names,
                "zone_names": zone_names,
                "operation_transition_summary": operation_transition_summary,
                "zone_transition_summary": zone_transition_summary,

                "train_calibration": train_calibration,
                "test_calibration": test_calibration,
                "walk_forward_slices": walk_forward_slices,
                "walk_forward_mode_slices": walk_forward_mode_slices,
                "walk_forward_stability_summary": walk_forward_stability_summary,
                "feature_group_strength": feature_group_strength,

                "analysis_checklist": analysis_checklist,
                "calibration_analysis_summary": calibration_analysis_summary,
                "roi_first_mode_selection_summary": roi_first_mode_selection_summary,
                "combo_quality_bucket_summary": combo_quality_bucket_summary,
                "entropy_dispersion_summary": entropy_dispersion_summary,
                "operation_context_summary": operation_context_summary,
                "operation_transition_probability_summary": operation_transition_probability_summary,
                "zone_transition_probability_summary": zone_transition_probability_summary,
                "loss_win_audit_summary": loss_win_audit_summary,
                "near_miss_report": near_miss_report,
                "rescue_comparison_summary": rescue_comparison_summary,

                "latest_operation": operation_labels[latest_index],
                "latest_zone": zone_labels[latest_index],
                "latest_operation_streak_length": int(operation_streak_length[latest_index]),
                "latest_pattern_score": round(float(draw_pattern_score[latest_index]), 4),
                "latest_spread_score": round(float(draw_spread_score[latest_index]), 4),
                "latest_row_entropy": round(float(draw_row_entropy[latest_index]), 4),
                "latest_column_entropy": round(float(draw_column_entropy[latest_index]), 4),
                "latest_zone_entropy": round(float(draw_zone_entropy[latest_index]), 4),
                "latest_neighbor_density": round(float(draw_neighbor_density[latest_index]), 4),
                "latest_avg_pair_distance": round(float(draw_avg_pair_distance[latest_index]), 4),
                "latest_regime_score": round(float(latest_regime_score), 4),
                "latest_regime": latest_regime,
                "latest_selected_mode": latest_selected_mode,
                "latest_confidence": latest_confidence,
                "latest_number_explanations": latest_explanations,

                "confidence_summary": confidence_summary,
                "audit_examples_by_quality": audit_examples_by_quality,
                "confidence_play_threshold": confidence_play_threshold,
                "confidence_watch_threshold": confidence_watch_threshold,

                "latest_scores": latest_scores,
                "top_pick_latest_scores": selected_scores("regime_aware"),
                "top20_latest_scores": latest_scores[:20],
                "latest_raw_scores": selected_scores("raw"),
                "latest_spread_scores": selected_scores("spread"),
                "latest_hybrid_scores": selected_scores("hybrid"),
                "latest_relaxed_hybrid_scores": selected_scores("relaxed_hybrid"),
                "latest_miss_scores": selected_scores("miss"),
                "latest_regime_scores": selected_scores("regime_aware"),
                "latest_rescue_1_scores": selected_scores("rescue_1"),
                "latest_smart_rescue_1_scores": selected_scores("smart_rescue_1"),
                "latest_smart_rescue_info": latest_smart_rescue_info,
                "latest_safe_smart_rescue_1_scores": selected_scores("safe_smart_rescue_1"),
                "latest_safe_smart_rescue_info": latest_safe_smart_rescue_info,
                "latest_swap_model_1_scores": selected_scores("swap_model_1"),
                "latest_swap_model_1_info": latest_swap_model_1_info,
                "latest_rescue_2_scores": selected_scores("rescue_2"),
                "latest_rescue_3_scores": selected_scores("rescue_3"),

                "feature_importance": feature_importance,
                "created_at": timezone.now().isoformat(),
            },
        )

        log_done(f"Saved AI result ID: {result.id}")

        # ------------------------------------------------------------
        # Terminal summary
        # ------------------------------------------------------------

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO AI V8 finished."))
        self.stdout.write(f"AI Result ID: {result.id}")
        self.stdout.write(f"Train decision points: {result.train_draws}")
        self.stdout.write(f"Test decision points: {result.test_draws}")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Precision: {precision:.4f}")
        self.stdout.write(f"Recall: {recall:.4f}")

        self.stdout.write("")
        self.stdout.write("Hit-count metric:")
        self.stdout.write(f"Theoretical random top {pick} hits: {theoretical_baseline:.3f}")

        for mode in mode_names:
            self.stdout.write(
                f"{mode}: {mode_average_hits[mode]:.3f} ({mode_lifts[mode]:+.3f})"
            )

        self.stdout.write("")
        self.stdout.write("ROI metric:")

        for mode in mode_names:
            summary = mode_profit_summaries[mode]
            self.stdout.write(
                f"{mode}: ROI {summary['roi']:+.4f}% | "
                f"profit €{summary['total_profit']:.2f} | "
                f"cost €{summary['total_cost']:.2f} | "
                f"return €{summary['total_return']:.2f}"
            )

        self.stdout.write("")
        self.stdout.write(f"Best hit mode: {best_hit_mode}")
        self.stdout.write(f"Best ROI mode: {best_roi_mode}")

        self.stdout.write("")
        self.stdout.write("Walk-forward slices:")

        for row in walk_forward_slices:
            self.stdout.write(str(row))

        self.stdout.write("")
        self.stdout.write("Feature group strength:")
        self.stdout.write(str(feature_group_strength))

        self.stdout.write("")
        self.stdout.write("V8 swap model training summary:")
        self.stdout.write(str(swap_training_summary))

        self.stdout.write("")
        self.stdout.write("Calibration summary:")
        self.stdout.write(str(calibration_analysis_summary))

        self.stdout.write("")
        self.stdout.write("Walk-forward stability summary:")
        self.stdout.write(str(walk_forward_stability_summary))

        self.stdout.write("")
        self.stdout.write("Loss/win audit summary:")
        self.stdout.write(str(loss_win_audit_summary))

        self.stdout.write("")
        self.stdout.write("Near-miss report summary:")
        self.stdout.write(
            str(
                {
                    "total_near_miss_rounds": near_miss_report.get("total_near_miss_rounds"),
                    "by_hit_count": near_miss_report.get("by_hit_count"),
                    "average_best_missing_rank": near_miss_report.get("average_best_missing_rank"),
                    "recoverable_top_20_rate": near_miss_report.get("recoverable_top_20_rate"),
                    "recoverable_top_30_rate": near_miss_report.get("recoverable_top_30_rate"),
                    "best_missing_rank_buckets": near_miss_report.get("best_missing_rank_buckets"),
                }
            )
        )

        self.stdout.write("")
        self.stdout.write(
            f"Latest operation: {operation_labels[latest_index]} | "
            f"zone={zone_labels[latest_index]} | "
            f"regime={latest_regime} | "
            f"mode={latest_selected_mode}"
        )

        self.stdout.write(
            f"Latest confidence: {latest_confidence['confidence_score']} | "
            f"decision={latest_confidence['decision']}"
        )

        self.stdout.write(f"Reasons: {latest_confidence['reasons']}")

        self.stdout.write("")
        self.stdout.write("Smart rescue top exact contexts:")
        self.stdout.write(str(smart_rescue_context_summary[:10]))
        self.stdout.write("")
        self.stdout.write("Smart rescue top coarse contexts:")
        self.stdout.write(str(smart_rescue_coarse_summary[:10]))
        self.stdout.write("")
        self.stdout.write("Latest safe smart rescue info:")
        self.stdout.write(str(latest_safe_smart_rescue_info))

        self.stdout.write("")
        self.stdout.write("Latest V8 swap model info:")
        self.stdout.write(str(latest_swap_model_1_info))
        self.stdout.write("")
        self.stdout.write("Rescue comparison summary:")
        self.stdout.write(str(rescue_comparison_summary))

        self.stdout.write("")
        self.stdout.write(f"Top {pick} latest regime-aware picks:")

        for item in selected_scores("regime_aware"):
            self.stdout.write(
                f"#{item['rank']:02d} Number {item['number']:02d} | "
                f"{item['probability_percent']:.4f}% | "
                f"empirical lift {item['empirical_lift']:+.6f}"
            )
