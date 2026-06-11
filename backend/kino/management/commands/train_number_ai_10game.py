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
    help = "Train operation-aware KINO AI with ROI, confidence, components, and skip logic"

    def add_arguments(self, parser):
        parser.add_argument("--horizon", type=int, default=10)
        parser.add_argument("--decision-step", type=int, default=5)
        parser.add_argument("--min-history", type=int, default=100)
        parser.add_argument("--test-ratio", type=float, default=0.2)
        parser.add_argument("--target-hits", type=int, default=3)
        parser.add_argument("--pick", type=int, default=12)

        parser.add_argument(
            "--stake",
            type=float,
            default=1.0,
            help="Cost per combo per round. Example: 1 means €1 per round.",
        )

        parser.add_argument(
            "--payout-table",
            type=str,
            default="kino",
            choices=["kino", "bonus"],
            help="Use normal KINO or KINO BONUS 12-number payout table.",
        )

        parser.add_argument(
            "--confidence-play-threshold",
            type=float,
            default=70.0,
            help="Confidence score threshold for PLAY decision.",
        )

        parser.add_argument(
            "--confidence-watch-threshold",
            type=float,
            default=55.0,
            help="Confidence score threshold for WATCH decision.",
        )
        parser.add_argument(
            "--bonus-fee",
            type=float,
            default=1.0,
            help="Extra KINO BONUS charge per round. Example: 1 means €1 extra per round.",
        )

    def handle(self, *args, **options):
        horizon = options["horizon"]
        decision_step = options["decision_step"]
        min_history = options["min_history"]
        test_ratio = options["test_ratio"]
        target_hits = options["target_hits"]
        pick = options["pick"]
        stake = options["stake"]
        payout_table_name = options["payout_table"]
        confidence_play_threshold = options["confidence_play_threshold"]
        confidence_watch_threshold = options["confidence_watch_threshold"]
        bonus_fee = options["bonus_fee"]
        if pick != 12:
            raise ValueError(
                "This version only supports verified 12-number payout tables. "
                "Run with --pick 12."
            )

        # ------------------------------------------------------------
        # Payout tables
        # ------------------------------------------------------------

        KINO_12_PAYOUTS = {
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

        KINO_BONUS_12_PAYOUTS = {
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

        if payout_table_name == "kino":
            payout_table = KINO_12_PAYOUTS
        elif payout_table_name == "bonus":
            payout_table = KINO_BONUS_12_PAYOUTS
        else:
            raise ValueError(f"Unknown payout table: {payout_table_name}")

        # ------------------------------------------------------------
        # Logging helpers
        # ------------------------------------------------------------

        def log_step(message):
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(f"▶ {message}"))

        def log_done(message):
            self.stdout.write(self.style.SUCCESS(f"✓ {message}"))

        # ------------------------------------------------------------
        # Board helpers
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

        draw_sets = [set(draw.numbers) for draw in draws]
        draw_ids = [draw.draw_id for draw in draws]

        draw_index_by_id = {
            draw.draw_id: index
            for index, draw in enumerate(draws)
        }

        log_done(f"Loaded {len(draws):,} draws")

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Training KINO AI confidence system..."))
        self.stdout.write(f"Total draws: {len(draws):,}")
        self.stdout.write(f"Horizon: next {horizon} games")
        self.stdout.write(f"Decision step: every {decision_step} games")
        self.stdout.write(f"Pick size: {pick} numbers")
        self.stdout.write(f"Stake per combo per round: €{stake:.2f}")
        self.stdout.write(f"Cost per combo decision: €{stake * horizon:.2f}")
        self.stdout.write(f"Payout table: {payout_table_name}")
        self.stdout.write(
            f"Target: number hits at least {target_hits} times inside next {horizon} games"
        )

        # ------------------------------------------------------------
        # Basic prefix counts
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

        # ------------------------------------------------------------
        # Stored shape cache
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Stored movement cache
        # ------------------------------------------------------------

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

        log_done("Prefix matrices ready")

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

        def component_features_for_number(current_index, number):
            row = number_row(number)
            column = number_column(number)

            return {
                "hot_last_10": int(count_in_window(current_index, number, 10)),
                "hot_last_20": int(count_in_window(current_index, number, 20)),
                "hot_last_50": int(count_in_window(current_index, number, 50)),
                "gap": int(gap_since_seen(current_index, number)),

                "row_pattern_last_10": recent_from_prefix(board_row_events_prefix, current_index, row, 10),
                "column_pattern_last_10": recent_from_prefix(board_column_events_prefix, current_index, column, 10),
                "board_hit_last_10": recent_from_prefix(board_number_hits_prefix, current_index, number, 10),

                "shape_hit_last_10": recent_from_prefix(shape_number_hits_prefix, current_index, number, 10),
                "shape_area_last_10": recent_from_prefix(shape_number_area_prefix, current_index, number, 10),
                "shape_center_last_20": recent_from_prefix(shape_centers_prefix, current_index, number, 20),

                "movement_target_last_20": recent_from_prefix(movement_target_centers_prefix, current_index, number, 20),
                "movement_source_last_20": recent_from_prefix(
                    movement_source_centers_completed_prefix,
                    current_index,
                    number,
                    20,
                ),
            }

        def explain_number_components(current_index, number, probability, baseline_probability):
            features = component_features_for_number(current_index, number)
            components = []
            reasons = []

            if features["shape_area_last_10"] > 0 or features["shape_hit_last_10"] > 0:
                components.append("shape")
                reasons.append(
                    f"shape area/hit active "
                    f"({features['shape_area_last_10']}/{features['shape_hit_last_10']})"
                )

            if features["shape_center_last_20"] > 0:
                components.append("shape_center")
                reasons.append(f"recent shape center x{features['shape_center_last_20']}")

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
                    f"row/column/board pressure "
                    f"({features['row_pattern_last_10']}/"
                    f"{features['column_pattern_last_10']}/"
                    f"{features['board_hit_last_10']})"
                )

            if probability > baseline_probability * 1.20:
                components.append("high_confidence_probability")
                reasons.append("probability well above baseline")

            if features["hot_last_10"] >= 4:
                components.append("hot")
                reasons.append(f"hot last 10: {features['hot_last_10']}")

            if features["gap"] >= 15:
                components.append("cold_gap")
                reasons.append(f"gap since seen: {features['gap']}")

            if not components:
                components.append("filler")
                reasons.append("no strong component detected")

            return {
                "number": int(number),
                "components": components,
                "reasons": reasons,
                "features": features,
                "probability": round(float(probability), 6),
                "probability_percent": round(float(probability) * 100, 4),
            }

        # ------------------------------------------------------------
        # Feature names
        # ------------------------------------------------------------

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

        log_step("Fitting model...")
        model.fit(X_train, y_train)
        log_done("Model fitting complete")

        # ------------------------------------------------------------
        # Pick helpers
        # ------------------------------------------------------------

        def are_neighbors(first_number, second_number):
            first_row = number_row(first_number)
            first_col = number_column(first_number)

            second_row = number_row(second_number)
            second_col = number_column(second_number)

            return (
                abs(first_row - second_row) <= 1
                and abs(first_col - second_col) <= 1
            )

        def select_raw_pick(draw_probs, numbers, pick_size):
            ranked_indices = np.argsort(draw_probs)[::-1]
            return ranked_indices[:pick_size]

        def select_low_pick(draw_probs, numbers, pick_size):
            ranked_indices = np.argsort(draw_probs)
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
                row = number_row(number)
                column = number_column(number)

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

        def select_mode_indices(mode, draw_probs, numbers, pick_size):
            if mode == "raw":
                return select_raw_pick(draw_probs, numbers, pick_size)

            if mode == "spread":
                return select_spread_pick(draw_probs, numbers, pick_size)

            if mode == "hybrid":
                return select_hybrid_pick(draw_probs, numbers, pick_size)

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

        # ------------------------------------------------------------
        # Profit and component audit helpers
        # ------------------------------------------------------------
        normal_payout_table = KINO_12_PAYOUTS
        bonus_payout_table = KINO_BONUS_12_PAYOUTS


        def calculate_normal_payout_for_hits(hit_count):
            return float(normal_payout_table.get(hit_count, 0)) * stake


        def calculate_bonus_payout_for_hits(hit_count):
            return float(bonus_payout_table.get(hit_count, 0)) * stake

        def calculate_pick_profit(selected_numbers, current_index, component_map=None):
            total_cost = 0.0
            total_return = 0.0
            hit_distribution = defaultdict(int)
            bonus_hit_distribution = defaultdict(int)
            component_hit_distribution = defaultdict(lambda: defaultdict(int))
            round_details = []
            payout_source_distribution = defaultdict(int)
            selected_set = set(selected_numbers)

            start_index = current_index + 1
            end_index = min(current_index + horizon + 1, len(draws))

            for future_index in range(start_index, end_index):
                future_numbers = list(draws[future_index].numbers)
                future_numbers_set = set(future_numbers)

                bonus_number = future_numbers[-1] if future_numbers else None

                hit_numbers = sorted(selected_set.intersection(future_numbers_set))
                hit_count = len(hit_numbers)
                bonus_hit = bonus_number in selected_set

                if payout_table_name == "kino":
                    payout = calculate_normal_payout_for_hits(hit_count)
                    payout_source = "kino"

                elif payout_table_name == "bonus":
                    if bonus_hit:
                        payout = calculate_bonus_payout_for_hits(hit_count)
                        payout_source = "bonus"
                    else:
                        payout = calculate_normal_payout_for_hits(hit_count)
                        payout_source = "kino_fallback"

                else:
                    raise ValueError(f"Unknown payout table: {payout_table_name}")

                if payout_table_name == "bonus":
                    round_cost = stake + bonus_fee
                else:
                    round_cost = stake

                total_cost += round_cost
                total_return += payout
                hit_distribution[hit_count] += 1
                bonus_hit_distribution[str(bonus_hit)] += 1
                payout_source_distribution[payout_source] += 1

                if component_map:
                    for number in selected_numbers:
                        number_components = component_map.get(number, ["filler"])
                        did_hit = number in future_numbers_set

                        for component in number_components:
                            if did_hit:
                                component_hit_distribution[component]["hits"] += 1
                            else:
                                component_hit_distribution[component]["misses"] += 1

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
                "component_hit_distribution": {
                    component: dict(values)
                    for component, values in component_hit_distribution.items()
                },
                "round_details": round_details,
                "payout_source_distribution": dict(payout_source_distribution),
            }

        def summarize_profit(results):
            total_rounds = sum(item["rounds_played"] for item in results)
            total_cost = sum(item["cost"] for item in results)
            total_return = sum(item["return"] for item in results)
            total_profit = total_return - total_cost
            roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0.0

            hit_distribution = defaultdict(int)
            bonus_hit_distribution = defaultdict(int)
            component_hit_distribution = defaultdict(lambda: defaultdict(int))
            payout_source_distribution = defaultdict(int)

            for item in results:
                for source, count in item.get("payout_source_distribution", {}).items():
                    payout_source_distribution[source] += count
            for item in results:
                for hits, count in item["hit_distribution"].items():
                    hit_distribution[int(hits)] += count

                for value, count in item.get("bonus_hit_distribution", {}).items():
                    bonus_hit_distribution[value] += count

                for component, values in item.get("component_hit_distribution", {}).items():
                    component_hit_distribution[component]["hits"] += values.get("hits", 0)
                    component_hit_distribution[component]["misses"] += values.get("misses", 0)

            profitable_decisions = sum(1 for item in results if item["profit"] > 0)
            losing_decisions = sum(1 for item in results if item["profit"] < 0)
            break_even_decisions = sum(1 for item in results if item["profit"] == 0)

            zero_hit_rounds = hit_distribution.get(0, 0)

            paying_high_rounds = sum(
                count
                for hits, count in hit_distribution.items()
                if int(hits) >= 6
            )

            dead_zone_rounds = sum(
                count
                for hits, count in hit_distribution.items()
                if 1 <= int(hits) <= 5
            )

            paying_rounds = zero_hit_rounds + paying_high_rounds

            paying_round_rate = (
                (paying_rounds / total_rounds) * 100
                if total_rounds > 0
                else 0.0
            )

            dead_zone_rate = (
                (dead_zone_rounds / total_rounds) * 100
                if total_rounds > 0
                else 0.0
            )

            component_summary = {}

            for component, values in component_hit_distribution.items():
                hits = values.get("hits", 0)
                misses = values.get("misses", 0)
                total = hits + misses
                hit_rate = (hits / total) * 100 if total > 0 else 0.0

                component_summary[component] = {
                    "hits": hits,
                    "misses": misses,
                    "total": total,
                    "hit_rate": round(hit_rate, 4),
                }

            return {
                "stake_per_round": round(stake, 2),
                "rounds_per_combo": horizon,
                "cost_per_combo_decision": round(stake * horizon, 2),
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
                "zero_hit_rounds": zero_hit_rounds,
                "paying_high_rounds": paying_high_rounds,
                "dead_zone_rounds": dead_zone_rounds,
                "paying_rounds": paying_rounds,
                "paying_round_rate": round(paying_round_rate, 4),
                "dead_zone_rate": round(dead_zone_rate, 4),
                "component_summary": component_summary,
                "payout_source_distribution": dict(payout_source_distribution),
            }

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

        def calculate_confidence(
            current_index,
            selected_numbers,
            selected_probabilities,
            selected_components,
            regime,
            selected_mode,
            regime_train_summary,
        ):
            score = 0.0
            reasons = []

            avg_probability = float(np.mean(selected_probabilities))
            max_probability = float(np.max(selected_probabilities))
            min_probability = float(np.min(selected_probabilities))

            above_baseline_count = sum(
                1 for probability in selected_probabilities
                if probability > baseline_target_probability
            )

            strong_above_baseline_count = sum(
                1 for probability in selected_probabilities
                if probability > baseline_target_probability * 1.20
            )

            score += min(25.0, above_baseline_count * 1.5)
            score += min(15.0, strong_above_baseline_count * 2.0)

            if avg_probability > baseline_target_probability:
                score += 10.0
                reasons.append("average probability above target baseline")

            if max_probability > baseline_target_probability * 1.35:
                score += 7.0
                reasons.append("has at least one very strong number")

            all_components = []

            for item in selected_components:
                all_components.extend(item["components"])

            component_counts = defaultdict(int)

            for component in all_components:
                component_counts[component] += 1

            strong_component_count = (
                component_counts["shape"]
                + component_counts["shape_center"]
                + component_counts["movement_target"]
                + component_counts["board_pattern"]
            )

            if component_counts["shape"] >= 3:
                score += 10.0
                reasons.append("shape component has multiple numbers")

            if component_counts["movement_target"] >= 2:
                score += 8.0
                reasons.append("movement target component active")

            if component_counts["board_pattern"] >= 3:
                score += 7.0
                reasons.append("board pattern component active")

            if strong_component_count >= 6:
                score += 10.0
                reasons.append("many picks come from strong components")

            filler_count = component_counts["filler"]

            if filler_count >= 5:
                score -= 15.0
                reasons.append("too many filler numbers")

            pattern_score_last_5 = recent_from_1d_prefix(pattern_score_prefix, current_index, 5)
            spread_score_last_5 = recent_from_1d_prefix(spread_score_prefix, current_index, 5)
            movement_last_5 = recent_from_1d_prefix(abs_movement_prefix, current_index, 5)

            if pattern_score_last_5 >= 12:
                score += 8.0
                reasons.append("recent pattern pressure is strong")

            if movement_last_5 >= 3:
                score += 5.0
                reasons.append("recent movement is active")

            if selected_mode == "spread" and spread_score_last_5 >= 80:
                score += 7.0
                reasons.append("spread mode matches recent spread pressure")

            if selected_mode == "hybrid" and pattern_score_last_5 >= 8:
                score += 5.0
                reasons.append("hybrid mode matches pattern pressure")

            if selected_mode == "miss" and regime in ["spread_low", "light_pattern"]:
                score += 8.0
                reasons.append("miss mode allowed in weak-pattern regime")

            if regime_train_summary.get(regime):
                regime_data = regime_train_summary[regime]
                best_mode = regime_data.get("best_mode")

                if best_mode == selected_mode:
                    score += 10.0
                    reasons.append(f"selected mode matches train ROI winner for {regime}")

            operation = operation_labels[current_index]
            streak = operation_streak_length[current_index]

            if operation == "heavy_pattern" and streak >= 2 and selected_mode in ["raw", "hybrid"]:
                score += 8.0
                reasons.append("heavy pattern streak supports hit-seeking mode")

            if operation == "scatter_spread" and selected_mode in ["spread", "miss"]:
                score += 8.0
                reasons.append("scatter operation supports spread/miss mode")

            rows = [number_row(number) for number in selected_numbers]
            columns = [number_column(number) for number in selected_numbers]

            unique_rows = len(set(rows))
            unique_columns = len(set(columns))

            if unique_rows >= 6 and unique_columns >= 8:
                score += 5.0
                reasons.append("combo has good board coverage")

            if unique_rows <= 3:
                score -= 8.0
                reasons.append("combo is too row-clustered")

            if unique_columns <= 4:
                score -= 8.0
                reasons.append("combo is too column-clustered")

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
                "min_probability": round(min_probability, 6),
                "above_baseline_count": above_baseline_count,
                "strong_above_baseline_count": strong_above_baseline_count,
                "component_counts": dict(component_counts),
                "unique_rows": unique_rows,
                "unique_columns": unique_columns,
                "operation": operation,
                "operation_streak": int(streak),
                "regime": regime,
                "selected_mode": selected_mode,
            }

        # ------------------------------------------------------------
        # Groups
        # ------------------------------------------------------------

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
        # Learn best mode per regime by TRAIN ROI
        # ------------------------------------------------------------

        log_step("Learning best pick mode per regime by ROI from training period...")

        train_mode_hits_by_regime = defaultdict(lambda: defaultdict(list))
        train_mode_roi_by_regime = defaultdict(lambda: defaultdict(list))

        for counter, (draw_index, positions) in enumerate(train_groups.items(), start=1):
            if counter % 500 == 0:
                self.stdout.write(
                    f"  analyzed train ROI modes for {counter:,}/{len(train_groups):,} decisions..."
                )

            regime = decision_regime_by_index.get(draw_index, "normal_pattern")

            draw_probs = train_probabilities[positions]
            draw_future_counts = future_counts_train[positions]
            draw_features = X_train[positions]
            numbers = draw_features[:, 0].astype(int)

            for mode in ["raw", "spread", "hybrid", "miss"]:
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

            for mode in ["raw", "spread", "hybrid", "miss"]:
                hit_values = train_mode_hits_by_regime[regime][mode]
                roi_values = train_mode_roi_by_regime[regime][mode]

                mode_hit_averages[mode] = float(np.mean(hit_values)) if hit_values else 0.0
                mode_roi_averages[mode] = float(np.mean(roi_values)) if roi_values else -999999.0

            best_mode_for_regime = max(mode_roi_averages, key=mode_roi_averages.get)

            regime_mode_map[regime] = best_mode_for_regime
            regime_train_summary[regime] = {
                "best_mode": best_mode_for_regime,
                "selection_metric": "roi",
                "hit_averages": {
                    "raw": round(mode_hit_averages["raw"], 4),
                    "spread": round(mode_hit_averages["spread"], 4),
                    "hybrid": round(mode_hit_averages["hybrid"], 4),
                    "miss": round(mode_hit_averages["miss"], 4),
                },
                "roi_averages": {
                    "raw": round(mode_roi_averages["raw"], 4),
                    "spread": round(mode_roi_averages["spread"], 4),
                    "hybrid": round(mode_roi_averages["hybrid"], 4),
                    "miss": round(mode_roi_averages["miss"], 4),
                },
            }

        log_done("Regime mode map learned")

        self.stdout.write("")
        self.stdout.write("Regime ROI mode map:")
        for regime, data in regime_train_summary.items():
            self.stdout.write(f"{regime}: {data}")

        # ------------------------------------------------------------
        # Test modes
        # ------------------------------------------------------------

        log_step("Testing raw/spread/hybrid/miss/regime-aware modes with confidence...")

        raw_pick_total_hits = []
        spread_pick_total_hits = []
        hybrid_pick_total_hits = []
        miss_pick_total_hits = []
        regime_pick_total_hits = []
        random_pick_total_hits = []

        raw_profit_results = []
        spread_profit_results = []
        hybrid_profit_results = []
        miss_profit_results = []
        regime_profit_results = []
        random_profit_results = []

        regime_test_summary = defaultdict(lambda: defaultdict(list))
        confidence_audits = []

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
            miss_indices = select_low_pick(draw_probs, numbers, pick)

            regime_indices = select_mode_indices(
                selected_mode_for_regime,
                draw_probs,
                numbers,
                pick,
            )

            random_indices = rng.choice(
                len(draw_features),
                size=pick,
                replace=False,
            )

            raw_selected_numbers = [int(numbers[index]) for index in raw_indices]
            spread_selected_numbers = [int(numbers[index]) for index in spread_indices]
            hybrid_selected_numbers = [int(numbers[index]) for index in hybrid_indices]
            miss_selected_numbers = [int(numbers[index]) for index in miss_indices]
            regime_selected_numbers = [int(numbers[index]) for index in regime_indices]
            random_selected_numbers = [int(numbers[index]) for index in random_indices]

            raw_hits = int(draw_future_counts[raw_indices].sum())
            spread_hits = int(draw_future_counts[spread_indices].sum())
            hybrid_hits = int(draw_future_counts[hybrid_indices].sum())
            miss_hits = int(draw_future_counts[miss_indices].sum())
            regime_hits = int(draw_future_counts[regime_indices].sum())
            random_hits = int(draw_future_counts[random_indices].sum())

            raw_pick_total_hits.append(raw_hits)
            spread_pick_total_hits.append(spread_hits)
            hybrid_pick_total_hits.append(hybrid_hits)
            miss_pick_total_hits.append(miss_hits)
            regime_pick_total_hits.append(regime_hits)
            random_pick_total_hits.append(random_hits)

            regime_components = []
            regime_component_map = {}

            for local_index in regime_indices:
                number = int(numbers[local_index])
                probability = float(draw_probs[local_index])

                explanation = explain_number_components(
                    current_index=draw_index,
                    number=number,
                    probability=probability,
                    baseline_probability=baseline_target_probability,
                )

                regime_components.append(explanation)
                regime_component_map[number] = explanation["components"]

            confidence = calculate_confidence(
                current_index=draw_index,
                selected_numbers=regime_selected_numbers,
                selected_probabilities=[float(draw_probs[index]) for index in regime_indices],
                selected_components=regime_components,
                regime=regime,
                selected_mode=selected_mode_for_regime,
                regime_train_summary=regime_train_summary,
            )

            raw_profit_results.append(calculate_pick_profit(raw_selected_numbers, draw_index))
            spread_profit_results.append(calculate_pick_profit(spread_selected_numbers, draw_index))
            hybrid_profit_results.append(calculate_pick_profit(hybrid_selected_numbers, draw_index))
            miss_profit_results.append(calculate_pick_profit(miss_selected_numbers, draw_index))

            regime_profit_result = calculate_pick_profit(
                regime_selected_numbers,
                draw_index,
                component_map=regime_component_map,
            )

            regime_profit_results.append(regime_profit_result)
            random_profit_results.append(calculate_pick_profit(random_selected_numbers, draw_index))

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
                    "number_explanations": regime_components,
                    "profit_result": regime_profit_result,
                }
            )

            regime_test_summary[regime]["regime_aware"].append(regime_hits)
            regime_test_summary[regime]["raw"].append(raw_hits)
            regime_test_summary[regime]["spread"].append(spread_hits)
            regime_test_summary[regime]["hybrid"].append(hybrid_hits)
            regime_test_summary[regime]["miss"].append(miss_hits)

        # ------------------------------------------------------------
        # Summaries
        # ------------------------------------------------------------

        raw_pick_hits = float(np.mean(raw_pick_total_hits))
        spread_pick_hits = float(np.mean(spread_pick_total_hits))
        hybrid_pick_hits = float(np.mean(hybrid_pick_total_hits))
        miss_pick_hits = float(np.mean(miss_pick_total_hits))
        regime_pick_hits = float(np.mean(regime_pick_total_hits))
        random_pick_hits = float(np.mean(random_pick_total_hits))

        theoretical_baseline = pick * horizon * 0.25

        raw_lift = raw_pick_hits - theoretical_baseline
        spread_lift = spread_pick_hits - theoretical_baseline
        hybrid_lift = hybrid_pick_hits - theoretical_baseline
        miss_lift = miss_pick_hits - theoretical_baseline
        regime_lift = regime_pick_hits - theoretical_baseline
        random_lift = random_pick_hits - theoretical_baseline

        raw_profit_summary = summarize_profit(raw_profit_results)
        spread_profit_summary = summarize_profit(spread_profit_results)
        hybrid_profit_summary = summarize_profit(hybrid_profit_results)
        miss_profit_summary = summarize_profit(miss_profit_results)
        regime_profit_summary = summarize_profit(regime_profit_results)
        random_profit_summary = summarize_profit(random_profit_results)

        confidence_summary = summarize_confidence(confidence_audits)
        def compact_audit_item(item):
            profit_result = item["profit_result"]

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
                "cost": round(float(profit_result["cost"]), 2),
                "return": round(float(profit_result["return"]), 2),
                "profit": round(float(profit_result["profit"]), 2),
                "roi": round(float(profit_result["roi"]), 4),
                "hit_distribution": profit_result["hit_distribution"],
                "bonus_hit_distribution": profit_result.get("bonus_hit_distribution", {}),
                "round_details": profit_result.get("round_details", []),
                "number_explanations": item.get("number_explanations", []),
            }


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
                    key=lambda item: (
                        item["confidence_score"],
                        item["profit_result"]["profit"],
                    ),
                    reverse=True,
                )[:6]
            ],
            "high_confidence_losses": [
                compact_audit_item(item)
                for item in sorted(
                    high_confidence_losses,
                    key=lambda item: (
                        item["confidence_score"],
                        item["profit_result"]["profit"],
                    ),
                )[:6]
            ],
            "low_confidence_wins": [
                compact_audit_item(item)
                for item in sorted(
                    low_confidence_wins,
                    key=lambda item: (
                        item["confidence_score"],
                        -item["profit_result"]["profit"],
                    ),
                )[:6]
            ],
            "low_confidence_losses": [
                compact_audit_item(item)
                for item in sorted(
                    low_confidence_losses,
                    key=lambda item: (
                        item["confidence_score"],
                        item["profit_result"]["profit"],
                    ),
                )[:6]
            ],
        }
        mode_scores = {
            "raw": raw_pick_hits,
            "spread": spread_pick_hits,
            "hybrid": hybrid_pick_hits,
            "miss": miss_pick_hits,
            "regime_aware": regime_pick_hits,
        }

        best_mode = max(mode_scores, key=mode_scores.get)
        model_pick_hits = mode_scores[best_mode]
        lift = model_pick_hits - theoretical_baseline

        selected_history = {
            "raw": raw_pick_total_hits[-200:],
            "spread": spread_pick_total_hits[-200:],
            "hybrid": hybrid_pick_total_hits[-200:],
            "miss": miss_pick_total_hits[-200:],
            "regime_aware": regime_pick_total_hits[-200:],
        }[best_mode]

        regime_test_output = {}

        for regime in ["spread_low", "light_pattern", "normal_pattern", "heavy_pattern"]:
            regime_test_output[regime] = {}

            for mode in ["raw", "spread", "hybrid", "miss", "regime_aware"]:
                values = regime_test_summary[regime][mode]
                regime_test_output[regime][mode] = (
                    round(float(np.mean(values)), 4)
                    if values
                    else None
                )

            regime_test_output[regime]["selected_mode"] = regime_mode_map.get(regime)

        log_done("Testing complete")

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

        latest_raw_indices = select_raw_pick(latest_probabilities, latest_numbers_array, pick)
        latest_spread_indices = select_spread_pick(latest_probabilities, latest_numbers_array, pick)
        latest_hybrid_indices = select_hybrid_pick(latest_probabilities, latest_numbers_array, pick)
        latest_miss_indices = select_low_pick(latest_probabilities, latest_numbers_array, pick)
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
                    "probability": round(float(probability), 6),
                    "probability_percent": round(float(probability) * 100, 4),
                    "above_baseline": round(
                        (float(probability) - baseline_target_probability) * 100,
                        4,
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
        latest_miss_scores = selected_scores_from_indices(latest_miss_indices)
        latest_regime_scores = selected_scores_from_indices(latest_regime_indices)

        latest_regime_numbers = [
            int(latest_numbers_array[index])
            for index in latest_regime_indices
        ]

        latest_regime_components = []

        for local_index in latest_regime_indices:
            number = int(latest_numbers_array[local_index])
            probability = float(latest_probabilities[local_index])

            explanation = explain_number_components(
                current_index=latest_index,
                number=number,
                probability=probability,
                baseline_probability=baseline_target_probability,
            )

            latest_regime_components.append(explanation)

        latest_confidence = calculate_confidence(
            current_index=latest_index,
            selected_numbers=latest_regime_numbers,
            selected_probabilities=[float(latest_probabilities[index]) for index in latest_regime_indices],
            selected_components=latest_regime_components,
            regime=latest_regime,
            selected_mode=latest_selected_mode,
            regime_train_summary=regime_train_summary,
        )

        log_done(
            f"Latest draw scored | operation={operation_labels[latest_index]} "
            f"| regime={latest_regime} | mode={latest_selected_mode} "
            f"| confidence={latest_confidence['confidence_score']}"
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
        # Save result
        # ------------------------------------------------------------

        log_step("Saving AI result...")

        result = KinoAIResult.objects.create(
            model_name="number_ai_v6_confidence_roi_components",
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
                "mode": "confidence_roi_component_backtest",
                "feature_version": "v6_confidence_roi_components",
                "extra_features_enabled": True,
                "regime_aware_enabled": True,
                "operation_vectors_enabled": True,
                "profit_backtest_enabled": True,
                "component_audit_enabled": True,
                "confidence_gate_enabled": True,

                "horizon": horizon,
                "decision_step": decision_step,
                "target_hits": target_hits,
                "baseline_target_probability": round(baseline_target_probability, 6),

                "stake": stake,
                "payout_table": payout_table_name,
                "payout_table_values": payout_table,
                "cost_per_combo_decision": round(stake * horizon, 2),

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

                "latest_confidence": latest_confidence,
                "confidence_summary": confidence_summary,
                "audit_examples_by_quality": audit_examples_by_quality,
                "confidence_play_threshold": confidence_play_threshold,
                "confidence_watch_threshold": confidence_watch_threshold,
                "latest_number_explanations": latest_regime_components,

                "best_mode": best_mode,

                "raw_pick_average_hits": raw_pick_hits,
                "spread_pick_average_hits": spread_pick_hits,
                "hybrid_pick_average_hits": hybrid_pick_hits,
                "miss_pick_average_hits": miss_pick_hits,
                "regime_pick_average_hits": regime_pick_hits,
                "random_pick_average_hits": random_pick_hits,

                "raw_lift": raw_lift,
                "spread_lift": spread_lift,
                "hybrid_lift": hybrid_lift,
                "miss_lift": miss_lift,
                "regime_lift": regime_lift,
                "random_lift": random_lift,

                "raw_profit_summary": raw_profit_summary,
                "spread_profit_summary": spread_profit_summary,
                "hybrid_profit_summary": hybrid_profit_summary,
                "miss_profit_summary": miss_profit_summary,
                "regime_profit_summary": regime_profit_summary,
                "random_profit_summary": random_profit_summary,

                "raw_pick_hits_by_test_decision": raw_pick_total_hits[-200:],
                "spread_pick_hits_by_test_decision": spread_pick_total_hits[-200:],
                "hybrid_pick_hits_by_test_decision": hybrid_pick_total_hits[-200:],
                "miss_pick_hits_by_test_decision": miss_pick_total_hits[-200:],
                "regime_pick_hits_by_test_decision": regime_pick_total_hits[-200:],
                "random_pick_hits_by_test_decision": random_pick_total_hits[-200:],

                "model_pick_hits_by_test_decision": selected_history,
                "model_top20_hits_by_test_decision": selected_history,
                "random_top20_hits_by_test_decision": random_pick_total_hits[-200:],
                "random_top20_average_hits": random_pick_hits,

                "latest_scores": latest_scores,
                "top_pick_latest_scores": latest_regime_scores,
                "top20_latest_scores": latest_scores[:20],

                "latest_raw_scores": latest_raw_scores,
                "latest_spread_scores": latest_spread_scores,
                "latest_hybrid_scores": latest_hybrid_scores,
                "latest_miss_scores": latest_miss_scores,
                "latest_regime_scores": latest_regime_scores,



                "confidence_audit_examples": sorted(
                    confidence_audits,
                    key=lambda item: item["confidence_score"],
                    reverse=True,
                )[:50],

                "feature_importance": feature_importance,
                "created_at": timezone.now().isoformat(),
            },
        )

        log_done(f"Saved AI result ID: {result.id}")

        # ------------------------------------------------------------
        # Terminal summary
        # ------------------------------------------------------------

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO AI confidence training finished."))
        self.stdout.write(f"AI Result ID: {result.id}")
        self.stdout.write(f"Train decision points: {result.train_draws}")
        self.stdout.write(f"Test decision points: {result.test_draws}")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Precision: {precision:.4f}")
        self.stdout.write(f"Recall: {recall:.4f}")

        self.stdout.write("")
        self.stdout.write("Hit-count metric:")
        self.stdout.write(f"Theoretical random top {pick} hits: {theoretical_baseline:.3f}")
        self.stdout.write(f"Simulated random top {pick} hits: {random_pick_hits:.3f}")
        self.stdout.write("")
        self.stdout.write(f"Raw AI top {pick} hits: {raw_pick_hits:.3f} ({raw_lift:+.3f})")
        self.stdout.write(f"Spread AI top {pick} hits: {spread_pick_hits:.3f} ({spread_lift:+.3f})")
        self.stdout.write(f"Hybrid AI top {pick} hits: {hybrid_pick_hits:.3f} ({hybrid_lift:+.3f})")
        self.stdout.write(f"Miss AI top {pick} hits: {miss_pick_hits:.3f} ({miss_lift:+.3f})")
        self.stdout.write(f"Regime-aware top {pick} hits: {regime_pick_hits:.3f} ({regime_lift:+.3f})")
        self.stdout.write("")
        self.stdout.write(f"Best overall hit-count mode: {best_mode}")

        self.stdout.write("")
        self.stdout.write("Profit / ROI metric:")
        self.stdout.write(f"Payout table: {payout_table_name}")
        self.stdout.write(f"Stake per combo per round: €{stake:.2f}")
        self.stdout.write(f"Rounds per combo decision: {horizon}")
        self.stdout.write(f"Cost per combo decision: €{stake * horizon:.2f}")

        for name, summary in [
            ("Raw", raw_profit_summary),
            ("Spread", spread_profit_summary),
            ("Hybrid", hybrid_profit_summary),
            ("Miss/Low-probability", miss_profit_summary),
            ("Regime-aware", regime_profit_summary),
            ("Random", random_profit_summary),
        ]:
            self.stdout.write("")
            self.stdout.write(f"{name}:")
            self.stdout.write(f"  Total combo decisions: {summary['total_combo_decisions']}")
            self.stdout.write(f"  Total rounds played: {summary['total_rounds_played']}")
            self.stdout.write(f"  Cost: €{summary['total_cost']:.2f}")
            self.stdout.write(f"  Return: €{summary['total_return']:.2f}")
            self.stdout.write(f"  Profit: €{summary['total_profit']:.2f}")
            self.stdout.write(f"  ROI: {summary['roi']:+.4f}%")
            self.stdout.write(f"  Paying rounds: {summary['paying_rounds']}")
            self.stdout.write(f"  Paying round rate: {summary['paying_round_rate']:+.4f}%")
            self.stdout.write(f"  Dead-zone rounds: {summary['dead_zone_rounds']}")
            self.stdout.write(f"  Dead-zone rate: {summary['dead_zone_rate']:+.4f}%")
            self.stdout.write(f"  Bonus hit distribution: {summary.get('bonus_hit_distribution', {})}")
            self.stdout.write(f"  Hit distribution: {summary['hit_distribution']}")
            self.stdout.write(f"  Component summary: {summary.get('component_summary', {})}")

        self.stdout.write("")
        self.stdout.write("Confidence gate summary:")
        for threshold, values in confidence_summary["thresholds"].items():
            self.stdout.write(f"  Confidence >= {threshold}: {values}")

        self.stdout.write("")
        self.stdout.write("Confidence bucket summary:")
        for bucket, values in confidence_summary["buckets"].items():
            self.stdout.write(f"  {bucket}: {values}")

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
        self.stdout.write(
            f"Latest confidence: {latest_confidence['confidence_score']} "
            f"| decision={latest_confidence['decision']}"
        )
        self.stdout.write(f"Confidence reasons: {latest_confidence['reasons']}")

        self.stdout.write("")
        self.stdout.write(f"Top {pick} latest confidence-aware picks:")
        for item in latest_regime_scores:
            self.stdout.write(
                f"#{item['rank']:02d} Number {item['number']:02d} | "
                f"{item['probability_percent']:.4f}% | "
                f"above target baseline {item['above_baseline']:+.4f}%"
            )

        self.stdout.write("")
        self.stdout.write("Latest number explanations:")
        for item in latest_regime_components:
            self.stdout.write(
                f"Number {item['number']:02d} | "
                f"components={item['components']} | "
                f"reasons={item['reasons']}"
            )