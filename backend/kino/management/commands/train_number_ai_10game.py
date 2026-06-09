from collections import defaultdict
from math import comb

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
    help = "Train V4 operation-aware AI model to score KINO numbers for a future window"

    def add_arguments(self, parser):
        parser.add_argument("--horizon", type=int, default=10)
        parser.add_argument("--decision-step", type=int, default=5)
        parser.add_argument("--min-history", type=int, default=100)
        parser.add_argument("--test-ratio", type=float, default=0.2)
        parser.add_argument("--target-hits", type=int, default=3)
        parser.add_argument("--pick", type=int, default=12)

    def handle(self, *args, **options):
        horizon = options["horizon"]
        decision_step = options["decision_step"]
        min_history = options["min_history"]
        test_ratio = options["test_ratio"]
        target_hits = options["target_hits"]
        pick = options["pick"]

        def log_step(message):
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(f"▶ {message}"))

        def log_done(message):
            self.stdout.write(self.style.SUCCESS(f"✓ {message}"))

        # ------------------------------------------------------------
        # Basic helpers
        # ------------------------------------------------------------

        def number_row(number):
            return (number - 1) // 10 + 1

        def number_column(number):
            return (number - 1) % 10 + 1

        def row_numbers(row):
            return list(range((row - 1) * 10 + 1, row * 10 + 1))

        def column_numbers(column):
            return [column + row_index * 10 for row_index in range(0, 8)]

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

        def operation_from_scores(pattern_score, spread_score, shape_count):
            if pattern_score >= 4 or shape_count >= 3:
                return "heavy_pattern"

            if pattern_score >= 2:
                return "normal_pattern"

            if pattern_score == 0 and spread_score >= 16:
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
                    f"Not enough draws. Have {len(draws)}, "
                    f"need at least {min_history + horizon + 100}."
                )
            )
            return

        log_done(f"Loaded {len(draws):,} draws")

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Training KINO 10-game Number AI V4..."))
        self.stdout.write(f"Total draws: {len(draws):,}")
        self.stdout.write(f"Horizon: next {horizon} games")
        self.stdout.write(f"Decision step: every {decision_step} games")
        self.stdout.write(f"Pick size: top {pick} numbers")
        self.stdout.write(
            f"Target: number hits at least {target_hits} times inside next {horizon} games"
        )

        draw_sets = [set(draw.numbers) for draw in draws]
        draw_ids = [draw.draw_id for draw in draws]

        draw_index_by_id = {
            draw.draw_id: index
            for index, draw in enumerate(draws)
        }

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
        # Stored board pattern cache
        # ------------------------------------------------------------

        log_step("Loading board pattern events into feature cache...")

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

        # ------------------------------------------------------------
        # Stored shape cache
        # ------------------------------------------------------------

        log_step("Loading shape events into feature cache...")

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

        # ------------------------------------------------------------
        # Stored movement cache
        # ------------------------------------------------------------

        log_step("Loading shape movements into feature cache...")

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

        log_step("Building prefix matrices for stored analysis features...")

        board_row_events_prefix = make_prefix(board_row_events)
        board_column_events_prefix = make_prefix(board_column_events)
        board_number_hits_prefix = make_prefix(board_number_hits)

        shape_number_hits_prefix = make_prefix(shape_number_hits)
        shape_number_area_prefix = make_prefix(shape_number_area)
        shape_centers_prefix = make_prefix(shape_centers)

        movement_target_centers_prefix = make_prefix(movement_target_centers)
        movement_source_centers_completed_prefix = make_prefix(
            movement_source_centers_completed
        )

        board_total_events = (
            board_row_events.sum(axis=1)
            + board_column_events.sum(axis=1)
        ).astype(np.float32)

        shape_total_events = shape_centers.sum(axis=1).astype(np.float32)
        movement_total_events = movement_target_centers.sum(axis=1).astype(np.float32)

        board_total_prefix = make_1d_prefix(board_total_events)
        shape_total_prefix = make_1d_prefix(shape_total_events)
        movement_total_prefix = make_1d_prefix(movement_total_events)

        log_done("Stored analysis prefix matrices ready")

        # ------------------------------------------------------------
        # Operation/intensity vectors per draw
        # ------------------------------------------------------------

        log_step("Building operation/intensity vectors...")

        draw_pattern_score = np.zeros(len(draws), dtype=np.float32)
        draw_spread_score = np.zeros(len(draws), dtype=np.float32)
        draw_avg_row = np.zeros(len(draws), dtype=np.float32)
        draw_avg_col = np.zeros(len(draws), dtype=np.float32)
        draw_delta_row = np.zeros(len(draws), dtype=np.float32)
        draw_delta_col = np.zeros(len(draws), dtype=np.float32)
        draw_abs_movement = np.zeros(len(draws), dtype=np.float32)

        operation_labels = []
        zone_labels = []

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

            avg_row = sum(rows) / len(rows)
            avg_col = sum(columns) / len(columns)

            unique_rows = len(set(rows))
            unique_columns = len(set(columns))
            spread = unique_rows + unique_columns

            pattern_score = float(board_total_events[index] + shape_total_events[index])
            shape_count = float(shape_total_events[index])

            operation = operation_from_scores(
                pattern_score=pattern_score,
                spread_score=spread,
                shape_count=shape_count,
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
            draw_spread_score[index] = spread
            draw_avg_row[index] = avg_row
            draw_avg_col[index] = avg_col
            draw_delta_row[index] = delta_row
            draw_delta_col[index] = delta_col
            draw_abs_movement[index] = abs_movement
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

        operation_one_hot_prefix = make_prefix(operation_one_hot)
        zone_one_hot_prefix = make_prefix(zone_one_hot)

        log_done("Operation/intensity vectors ready")

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

        def extra_analysis_features(current_index, number):
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
                recent_from_prefix(
                    movement_source_centers_completed_prefix,
                    current_index,
                    number,
                    20,
                ),
                recent_from_prefix(
                    movement_source_centers_completed_prefix,
                    current_index,
                    number,
                    100,
                ),
            ]

        def operation_features(current_index):
            op_counts_last_10 = [
                recent_from_prefix(
                    operation_one_hot_prefix,
                    current_index,
                    op_index,
                    10,
                )
                for op_index in range(len(operation_names))
            ]

            zone_counts_last_10 = [
                recent_from_prefix(
                    zone_one_hot_prefix,
                    current_index,
                    zone_index,
                    10,
                )
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

                recent_from_1d_prefix(pattern_score_prefix, current_index, 3),
                recent_from_1d_prefix(pattern_score_prefix, current_index, 5),
                recent_from_1d_prefix(pattern_score_prefix, current_index, 10),

                recent_from_1d_prefix(spread_score_prefix, current_index, 3),
                recent_from_1d_prefix(spread_score_prefix, current_index, 5),
                recent_from_1d_prefix(spread_score_prefix, current_index, 10),

                recent_from_1d_prefix(abs_movement_prefix, current_index, 3),
                recent_from_1d_prefix(abs_movement_prefix, current_index, 5),
                recent_from_1d_prefix(abs_movement_prefix, current_index, 10),

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
            scatter_pressure = scatter_streak_length[current_index] * -2.0

            return (
                board_10 * 2.0
                + board_50 * 0.35
                + shape_10 * 1.0
                + shape_50 * 0.15
                + movement_20 * 1.5
                + movement_100 * 0.20
                + operation_pressure * 0.75
                + scatter_pressure
            )

        feature_names = [
            "number",
            "row",
            "column",
            "count_last_5",
            "count_last_10",
            "count_last_20",
            "count_last_50",
            "count_last_100",
            "ratio_last_10",
            "ratio_last_20",
            "ratio_last_50",
            "gap_since_seen",
            "appeared_current_draw",
            "appeared_previous_draw",
            "row_hits_current_draw",
            "column_hits_current_draw",

            "row_pattern_last_10",
            "row_pattern_last_50",
            "column_pattern_last_10",
            "column_pattern_last_50",

            "number_board_pattern_hit_last_10",
            "number_board_pattern_hit_last_50",

            "number_shape_hit_last_10",
            "number_shape_hit_last_50",
            "number_shape_area_last_10",
            "number_shape_area_last_50",

            "number_shape_center_last_20",
            "number_shape_center_last_100",

            "movement_target_center_last_20",
            "movement_target_center_last_100",
            "movement_source_center_completed_last_20",
            "movement_source_center_completed_last_100",

            "draw_pattern_score",
            "draw_spread_score",
            "draw_avg_row",
            "draw_avg_col",
            "draw_delta_row",
            "draw_delta_col",
            "draw_abs_movement",

            "pattern_score_last_3",
            "pattern_score_last_5",
            "pattern_score_last_10",

            "spread_score_last_3",
            "spread_score_last_5",
            "spread_score_last_10",

            "abs_movement_last_3",
            "abs_movement_last_5",
            "abs_movement_last_10",

            "operation_streak_length",
            "heavy_streak_length",
            "normal_streak_length",
            "light_streak_length",
            "scatter_streak_length",
            "quiet_streak_length",

            *[f"operation_last_10_{name}" for name in operation_names],
            *[f"zone_last_10_{name}" for name in zone_names],
        ]

        # ------------------------------------------------------------
        # Decision points and regime thresholds
        # ------------------------------------------------------------

        decision_indices = list(
            range(
                min_history,
                len(draws) - horizon,
                decision_step,
            )
        )

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

        regime_counter = defaultdict(int)

        for regime in decision_regime_by_index.values():
            regime_counter[regime] += 1

        self.stdout.write("")
        self.stdout.write("Regime thresholds based on training data:")
        self.stdout.write(f"q25: {regime_thresholds['q25']:.3f}")
        self.stdout.write(f"q50: {regime_thresholds['q50']:.3f}")
        self.stdout.write(f"q75: {regime_thresholds['q75']:.3f}")

        self.stdout.write("")
        self.stdout.write("Decision regime distribution:")
        for regime, count in sorted(regime_counter.items()):
            self.stdout.write(f"{regime}: {count:,}")

        # ------------------------------------------------------------
        # Build ML rows
        # ------------------------------------------------------------

        X_rows = []
        y_rows = []
        future_count_rows = []
        row_draw_indices = []

        total_decisions = len(decision_indices)

        log_step(f"Building ML rows for {total_decisions:,} decision points...")

        for decision_counter, current_index in enumerate(decision_indices, start=1):
            if decision_counter % 250 == 0:
                self.stdout.write(
                    f"  built {decision_counter:,}/{total_decisions:,} decision points "
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
                column_hits_current = len(
                    current_numbers.intersection(column_numbers(column))
                )

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

                features.extend(extra_analysis_features(current_index, number))
                features.extend(current_operation_features)

                target = 1 if next_horizon_count >= target_hits else 0

                X_rows.append(features)
                y_rows.append(target)
                future_count_rows.append(next_horizon_count)
                row_draw_indices.append(current_index)

        log_done(f"Built ML dataset: {len(X_rows):,} rows")

        log_step("Converting dataset to NumPy arrays...")

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int8)
        future_counts = np.array(future_count_rows, dtype=np.int16)
        row_draw_indices = np.array(row_draw_indices, dtype=np.int32)

        log_done(f"X shape: {X.shape}")

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

        self.stdout.write("")
        self.stdout.write(f"Training rows: {len(X_train):,}")
        self.stdout.write(f"Testing rows: {len(X_test):,}")

        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            self.stdout.write("")
            self.stdout.write(
                self.style.ERROR(
                    f"Training target has only one class: {unique_classes.tolist()}"
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    "Your target is impossible or too strict for this horizon."
                )
            )
            self.stdout.write("Try:")
            self.stdout.write("  horizon 1  -> --target-hits 1")
            self.stdout.write("  horizon 5  -> --target-hits 2")
            self.stdout.write("  horizon 10 -> --target-hits 3")
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

        log_step("Fitting 10-game model...")
        model.fit(X_train, y_train)
        log_done("Model fitting complete")

        # ------------------------------------------------------------
        # Pick strategy helpers
        # ------------------------------------------------------------

        def get_number_row(number):
            return (number - 1) // 10 + 1

        def get_number_column(number):
            return (number - 1) % 10 + 1

        def are_neighbors(first_number, second_number):
            first_row = get_number_row(first_number)
            first_col = get_number_column(first_number)

            second_row = get_number_row(second_number)
            second_col = get_number_column(second_number)

            return (
                abs(first_row - second_row) <= 1
                and abs(first_col - second_col) <= 1
            )

        def select_raw_pick(draw_probs, numbers, pick_size):
            ranked_indices = np.argsort(draw_probs)[::-1]
            return ranked_indices[:pick_size]

        def select_spread_pick(
            draw_probs,
            numbers,
            pick_size,
            max_per_row=2,
            max_per_column=2,
            max_neighbors=2,
        ):
            ranked_indices = np.argsort(draw_probs)[::-1]

            selected_indices = []
            row_counts = {}
            column_counts = {}

            for index in ranked_indices:
                number = int(numbers[index])
                row = get_number_row(number)
                column = get_number_column(number)

                if row_counts.get(row, 0) >= max_per_row:
                    continue

                if column_counts.get(column, 0) >= max_per_column:
                    continue

                neighbor_count = 0

                for selected_index in selected_indices:
                    selected_number = int(numbers[selected_index])

                    if are_neighbors(number, selected_number):
                        neighbor_count += 1

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
                row = get_number_row(number)
                column = get_number_column(number)

                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

            for index in ranked_indices:
                if index in selected_indices:
                    continue

                number = int(numbers[index])
                row = get_number_row(number)
                column = get_number_column(number)

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

        def select_mode_indices(mode, draw_probs, numbers, pick_size):
            if mode == "raw":
                return select_raw_pick(draw_probs, numbers, pick_size)

            if mode == "spread":
                return select_spread_pick(draw_probs, numbers, pick_size)

            if mode == "hybrid":
                return select_hybrid_pick(draw_probs, numbers, pick_size)

            return select_hybrid_pick(draw_probs, numbers, pick_size)

        def build_groups(draw_indices):
            groups = defaultdict(list)

            for position, draw_index in enumerate(draw_indices.tolist()):
                groups[int(draw_index)].append(position)

            return {
                draw_index: np.array(positions, dtype=np.int32)
                for draw_index, positions in groups.items()
            }

        train_groups = build_groups(train_draw_indices)
        test_groups = build_groups(test_draw_indices)

        # ------------------------------------------------------------
        # Predict probabilities
        # ------------------------------------------------------------

        log_step("Predicting train/test probabilities...")

        train_probabilities = model.predict_proba(X_train)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]

        test_predictions = (test_probabilities >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, zero_division=0)
        recall = recall_score(y_test, test_predictions, zero_division=0)

        log_done("Probabilities ready")

        # ------------------------------------------------------------
        # Learn best mode per regime using TRAIN ONLY
        # ------------------------------------------------------------

        log_step("Learning best pick mode per regime from training period...")

        train_mode_hits_by_regime = defaultdict(lambda: defaultdict(list))

        for counter, (draw_index, positions) in enumerate(train_groups.items(), start=1):
            if counter % 500 == 0:
                self.stdout.write(
                    f"  analyzed train pick modes for {counter:,}/{len(train_groups):,} decisions..."
                )

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")

            draw_probs = train_probabilities[positions]
            draw_future_counts = future_counts_train[positions]
            draw_features = X_train[positions]
            numbers = draw_features[:, 0].astype(int)

            for mode in ["raw", "spread", "hybrid"]:
                selected_indices = select_mode_indices(mode, draw_probs, numbers, pick)
                hits = int(draw_future_counts[selected_indices].sum())
                train_mode_hits_by_regime[regime][mode].append(hits)

        regime_mode_map = {}
        regime_train_summary = {}

        for regime in ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]:
            mode_averages = {}

            for mode in ["raw", "spread", "hybrid"]:
                values = train_mode_hits_by_regime[regime][mode]

                if values:
                    mode_averages[mode] = float(np.mean(values))
                else:
                    mode_averages[mode] = 0.0

            best_mode_for_regime = max(mode_averages, key=mode_averages.get)

            regime_mode_map[regime] = best_mode_for_regime
            regime_train_summary[regime] = {
                "best_mode": best_mode_for_regime,
                "raw": round(mode_averages["raw"], 4),
                "spread": round(mode_averages["spread"], 4),
                "hybrid": round(mode_averages["hybrid"], 4),
            }

        log_done("Regime mode map learned")

        self.stdout.write("")
        self.stdout.write("Regime mode map:")
        for regime, mode in regime_mode_map.items():
            self.stdout.write(f"{regime}: {mode} | {regime_train_summary[regime]}")

        # ------------------------------------------------------------
        # Test pick modes
        # ------------------------------------------------------------

        log_step("Testing raw/spread/hybrid/regime-aware pick modes...")

        raw_pick_total_hits = []
        spread_pick_total_hits = []
        hybrid_pick_total_hits = []
        regime_pick_total_hits = []
        random_pick_total_hits = []

        regime_test_summary = defaultdict(lambda: defaultdict(list))

        rng = np.random.default_rng(seed=42)

        unique_test_draw_indices = sorted(test_groups.keys())

        for counter, draw_index in enumerate(unique_test_draw_indices, start=1):
            if counter % 250 == 0:
                self.stdout.write(
                    f"  tested {counter:,}/{len(unique_test_draw_indices):,} decision points..."
                )

            positions = test_groups[draw_index]

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")
            selected_mode_for_regime = regime_mode_map.get(regime, "hybrid")

            draw_probs = test_probabilities[positions]
            draw_future_counts = future_counts_test[positions]
            draw_features = X_test[positions]
            numbers = draw_features[:, 0].astype(int)

            raw_indices = select_raw_pick(draw_probs, numbers, pick)
            spread_indices = select_spread_pick(draw_probs, numbers, pick)
            hybrid_indices = select_hybrid_pick(draw_probs, numbers, pick)
            regime_indices = select_mode_indices(
                selected_mode_for_regime,
                draw_probs,
                numbers,
                pick,
            )

            raw_hits = int(draw_future_counts[raw_indices].sum())
            spread_hits = int(draw_future_counts[spread_indices].sum())
            hybrid_hits = int(draw_future_counts[hybrid_indices].sum())
            regime_hits = int(draw_future_counts[regime_indices].sum())

            raw_pick_total_hits.append(raw_hits)
            spread_pick_total_hits.append(spread_hits)
            hybrid_pick_total_hits.append(hybrid_hits)
            regime_pick_total_hits.append(regime_hits)

            regime_test_summary[regime]["regime_aware"].append(regime_hits)
            regime_test_summary[regime]["raw"].append(raw_hits)
            regime_test_summary[regime]["spread"].append(spread_hits)
            regime_test_summary[regime]["hybrid"].append(hybrid_hits)

            random_indices = rng.choice(
                len(draw_features),
                size=pick,
                replace=False,
            )

            random_hits = int(draw_future_counts[random_indices].sum())
            random_pick_total_hits.append(random_hits)

        raw_pick_hits = float(np.mean(raw_pick_total_hits))
        spread_pick_hits = float(np.mean(spread_pick_total_hits))
        hybrid_pick_hits = float(np.mean(hybrid_pick_total_hits))
        regime_pick_hits = float(np.mean(regime_pick_total_hits))
        random_pick_hits = float(np.mean(random_pick_total_hits))

        theoretical_baseline = pick * horizon * 0.25

        raw_lift = raw_pick_hits - theoretical_baseline
        spread_lift = spread_pick_hits - theoretical_baseline
        hybrid_lift = hybrid_pick_hits - theoretical_baseline
        regime_lift = regime_pick_hits - theoretical_baseline
        random_lift = random_pick_hits - theoretical_baseline

        mode_scores = {
            "raw": raw_pick_hits,
            "spread": spread_pick_hits,
            "hybrid": hybrid_pick_hits,
            "regime_aware": regime_pick_hits,
        }

        best_mode = max(mode_scores, key=mode_scores.get)
        model_pick_hits = mode_scores[best_mode]
        lift = model_pick_hits - theoretical_baseline

        regime_test_output = {}

        for regime in ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]:
            regime_test_output[regime] = {}

            for mode in ["raw", "spread", "hybrid", "regime_aware"]:
                values = regime_test_summary[regime][mode]

                if values:
                    regime_test_output[regime][mode] = round(float(np.mean(values)), 4)
                else:
                    regime_test_output[regime][mode] = None

            regime_test_output[regime]["selected_mode"] = regime_mode_map.get(regime)

        log_done("Pick mode testing complete")

        # ------------------------------------------------------------
        # Baseline target probability
        # ------------------------------------------------------------

        baseline_target_probability = 0

        for hits in range(target_hits, horizon + 1):
            baseline_target_probability += (
                comb(horizon, hits)
                * (0.25 ** hits)
                * (0.75 ** (horizon - hits))
            )

        # ------------------------------------------------------------
        # Score latest draw
        # ------------------------------------------------------------

        log_step("Scoring latest draw with operation-aware regime mode...")

        latest_index = len(draws) - 1
        latest_numbers = draw_sets[latest_index]
        previous_numbers = draw_sets[latest_index - 1]

        latest_features = []
        latest_operation_features = operation_features(latest_index)

        for number in range(1, 81):
            row = number_row(number)
            column = number_column(number)

            count_last_5 = count_in_window(latest_index, number, 5)
            count_last_10 = count_in_window(latest_index, number, 10)
            count_last_20 = count_in_window(latest_index, number, 20)
            count_last_50 = count_in_window(latest_index, number, 50)
            count_last_100 = count_in_window(latest_index, number, 100)

            row_hits_current = len(latest_numbers.intersection(row_numbers(row)))
            column_hits_current = len(
                latest_numbers.intersection(column_numbers(column))
            )

            latest_row_features = [
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

            latest_row_features.extend(extra_analysis_features(latest_index, number))
            latest_row_features.extend(latest_operation_features)

            latest_features.append(latest_row_features)

        latest_X = np.array(latest_features, dtype=np.float32)
        latest_probabilities = model.predict_proba(latest_X)[:, 1]
        latest_numbers_array = np.array(range(1, 81), dtype=np.int32)

        latest_regime_score = current_pattern_pressure_score(latest_index)
        latest_regime = classify_regime(latest_regime_score)
        latest_selected_mode = regime_mode_map.get(latest_regime, "hybrid")

        latest_raw_indices = select_raw_pick(
            latest_probabilities,
            latest_numbers_array,
            pick,
        )

        latest_spread_indices = select_spread_pick(
            latest_probabilities,
            latest_numbers_array,
            pick,
        )

        latest_hybrid_indices = select_hybrid_pick(
            latest_probabilities,
            latest_numbers_array,
            pick,
        )

        latest_regime_indices = select_mode_indices(
            latest_selected_mode,
            latest_probabilities,
            latest_numbers_array,
            pick,
        )

        latest_scores = []

        for index, probability in enumerate(latest_probabilities):
            number = index + 1

            latest_scores.append(
                {
                    "number": number,
                    "row": number_row(number),
                    "column": number_column(number),
                    "probability": round(float(probability), 5),
                    "probability_percent": round(float(probability) * 100, 3),
                    "above_baseline": round(
                        (float(probability) - baseline_target_probability) * 100,
                        3,
                    ),
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

        def selected_scores_from_indices(indices):
            selected_numbers = [int(latest_numbers_array[index]) for index in indices]
            return [score_by_number[number] for number in selected_numbers]

        latest_raw_scores = selected_scores_from_indices(latest_raw_indices)
        latest_spread_scores = selected_scores_from_indices(latest_spread_indices)
        latest_hybrid_scores = selected_scores_from_indices(latest_hybrid_indices)
        latest_regime_scores = selected_scores_from_indices(latest_regime_indices)

        log_done(
            f"Latest draw scored | operation={operation_labels[latest_index]} "
            f"| regime={latest_regime} | selected_mode={latest_selected_mode}"
        )

        # ------------------------------------------------------------
        # Feature importance
        # ------------------------------------------------------------

        coefficients = model.named_steps["model"].coef_[0]

        feature_importance = sorted(
            [
                {
                    "feature": feature_names[index],
                    "coefficient": round(float(coef), 6),
                    "absolute_strength": round(abs(float(coef)), 6),
                }
                for index, coef in enumerate(coefficients)
            ],
            key=lambda item: item["absolute_strength"],
            reverse=True,
        )

        # ------------------------------------------------------------
        # Block history
        # ------------------------------------------------------------

        block_size = 250
        test_blocks = []

        for start in range(0, len(regime_pick_total_hits), block_size):
            block = regime_pick_total_hits[start:start + block_size]

            if not block:
                continue

            block_average = float(np.mean(block))

            test_blocks.append(
                {
                    "start": start,
                    "end": start + len(block) - 1,
                    "decisions": len(block),
                    "average_hits": round(block_average, 4),
                    "lift": round(block_average - theoretical_baseline, 4),
                }
            )

        # ------------------------------------------------------------
        # Save result
        # ------------------------------------------------------------

        log_step("Saving AI result...")

        selected_history = {
            "raw": raw_pick_total_hits[-200:],
            "spread": spread_pick_total_hits[-200:],
            "hybrid": hybrid_pick_total_hits[-200:],
            "regime_aware": regime_pick_total_hits[-200:],
        }[best_mode]

        result = KinoAIResult.objects.create(
            model_name="number_ai_10game_v4_operation_aware",
            train_draws=len(set(train_draw_indices.tolist())),
            test_draws=len(unique_test_draw_indices),
            baseline_top20_hits=theoretical_baseline,
            model_top20_hits=model_pick_hits,
            lift=lift,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            data={
                "pick": pick,
                "mode": "10_game_window",
                "feature_version": "v4_operation_intensity_vectors",
                "extra_features_enabled": True,
                "regime_aware_enabled": True,
                "operation_vectors_enabled": True,

                "horizon": horizon,
                "decision_step": decision_step,
                "target_hits": target_hits,
                "baseline_target_probability": round(
                    baseline_target_probability,
                    6,
                ),

                "latest_draw_id": draw_ids[-1],
                "split_draw_id": draw_ids[split_draw_index],
                "training_rows": int(len(X_train)),
                "testing_rows": int(len(X_test)),

                "operation_names": operation_names,
                "zone_names": zone_names,

                "latest_operation": operation_labels[latest_index],
                "latest_zone": zone_labels[latest_index],
                "latest_operation_streak_length": int(operation_streak_length[latest_index]),
                "latest_pattern_score": round(float(draw_pattern_score[latest_index]), 4),
                "latest_spread_score": round(float(draw_spread_score[latest_index]), 4),
                "latest_avg_row": round(float(draw_avg_row[latest_index]), 4),
                "latest_avg_col": round(float(draw_avg_col[latest_index]), 4),
                "latest_delta_row": round(float(draw_delta_row[latest_index]), 4),
                "latest_delta_col": round(float(draw_delta_col[latest_index]), 4),

                "regime_thresholds": regime_thresholds,
                "regime_mode_map": regime_mode_map,
                "regime_train_summary": regime_train_summary,
                "regime_test_summary": regime_test_output,
                "latest_regime_score": round(float(latest_regime_score), 4),
                "latest_regime": latest_regime,
                "latest_selected_mode": latest_selected_mode,

                "best_mode": best_mode,

                "raw_pick_average_hits": raw_pick_hits,
                "spread_pick_average_hits": spread_pick_hits,
                "hybrid_pick_average_hits": hybrid_pick_hits,
                "regime_pick_average_hits": regime_pick_hits,
                "random_pick_average_hits": random_pick_hits,

                "raw_lift": raw_lift,
                "spread_lift": spread_lift,
                "hybrid_lift": hybrid_lift,
                "regime_lift": regime_lift,
                "random_lift": random_lift,

                "raw_pick_hits_by_test_decision": raw_pick_total_hits[-200:],
                "spread_pick_hits_by_test_decision": spread_pick_total_hits[-200:],
                "hybrid_pick_hits_by_test_decision": hybrid_pick_total_hits[-200:],
                "regime_pick_hits_by_test_decision": regime_pick_total_hits[-200:],
                "random_pick_hits_by_test_decision": random_pick_total_hits[-200:],

                "model_pick_hits_by_test_decision": selected_history,

                "model_top20_hits_by_test_decision": selected_history,
                "random_top20_hits_by_test_decision": random_pick_total_hits[-200:],
                "random_top20_average_hits": random_pick_hits,

                "test_blocks": test_blocks,

                "latest_scores": latest_scores,
                "top_pick_latest_scores": latest_regime_scores,
                "top20_latest_scores": latest_scores[:20],

                "latest_raw_scores": latest_raw_scores,
                "latest_spread_scores": latest_spread_scores,
                "latest_hybrid_scores": latest_hybrid_scores,
                "latest_regime_scores": latest_regime_scores,

                "feature_importance": feature_importance,
                "created_at": timezone.now().isoformat(),
            },
        )

        log_done(f"Saved AI result ID: {result.id}")

        # ------------------------------------------------------------
        # Terminal summary
        # ------------------------------------------------------------

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO 10-game AI V4 training finished."))
        self.stdout.write(f"AI Result ID: {result.id}")
        self.stdout.write(f"Train decision points: {result.train_draws}")
        self.stdout.write(f"Test decision points: {result.test_draws}")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Precision: {precision:.4f}")
        self.stdout.write(f"Recall: {recall:.4f}")

        self.stdout.write("")
        self.stdout.write("Main 10-game test metric:")
        self.stdout.write(
            f"Theoretical random top {pick} hits: {theoretical_baseline:.3f}"
        )
        self.stdout.write(
            f"Simulated random top {pick} hits: {random_pick_hits:.3f}"
        )
        self.stdout.write("")
        self.stdout.write(f"Raw AI top {pick} hits: {raw_pick_hits:.3f} ({raw_lift:+.3f})")
        self.stdout.write(f"Spread AI top {pick} hits: {spread_pick_hits:.3f} ({spread_lift:+.3f})")
        self.stdout.write(f"Hybrid AI top {pick} hits: {hybrid_pick_hits:.3f} ({hybrid_lift:+.3f})")
        self.stdout.write(f"Regime-aware top {pick} hits: {regime_pick_hits:.3f} ({regime_lift:+.3f})")
        self.stdout.write("")
        self.stdout.write(f"Best overall mode: {best_mode}")
        self.stdout.write(
            f"Best mode lift vs simulated random: {model_pick_hits - random_pick_hits:+.3f}"
        )

        self.stdout.write("")
        self.stdout.write("Regime test summary:")
        for regime, values in regime_test_output.items():
            self.stdout.write(f"{regime}: {values}")

        self.stdout.write("")
        self.stdout.write(
            f"Latest operation: {operation_labels[latest_index]} "
            f"| zone={zone_labels[latest_index]} "
            f"| streak={int(operation_streak_length[latest_index])}"
        )
        self.stdout.write(
            f"Latest regime: {latest_regime} "
            f"| score={latest_regime_score:.3f} "
            f"| mode={latest_selected_mode}"
        )

        self.stdout.write("")
        self.stdout.write(f"Top {pick} latest operation-aware picks:")
        for item in latest_regime_scores:
            self.stdout.write(
                f"#{item['rank']:02d} Number {item['number']:02d} | "
                f"{item['probability_percent']:.3f}% | "
                f"above target baseline {item['above_baseline']:+.3f}%"
            )