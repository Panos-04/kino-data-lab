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
    help = "Train AI model to score KINO numbers for a future 10-game window"

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

        log_done(f"Loaded {len(draws)} draws")

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Training KINO 10-game Number AI V2..."))
        self.stdout.write(f"Total draws: {len(draws)}")
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
        # Basic number prefix counts
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

        def recent_from_prefix(prefix, current_index, column_index, window_size):
            start_index = max(0, current_index - window_size + 1)
            end_index = current_index + 1

            return int(prefix[end_index][column_index] - prefix[start_index][column_index])

        # ------------------------------------------------------------
        # Stored analysis feature matrices
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

        log_step("Loading shape movements into feature cache...")

        # Completed movements only.
        # We attach movement features to the TO draw, because before the TO draw happens,
        # the movement is not known yet. This avoids future leakage.
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

        log_done("Stored analysis prefix matrices ready")

        # ------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------

        def number_row(number):
            return (number - 1) // 10 + 1

        def number_column(number):
            return (number - 1) % 10 + 1

        def row_numbers(row):
            return list(range((row - 1) * 10 + 1, row * 10 + 1))

        def column_numbers(column):
            return [column + row_index * 10 for row_index in range(0, 8)]

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
                # Row/column pattern pressure
                recent_from_prefix(board_row_events_prefix, current_index, row, 10),
                recent_from_prefix(board_row_events_prefix, current_index, row, 50),
                recent_from_prefix(board_column_events_prefix, current_index, column, 10),
                recent_from_prefix(board_column_events_prefix, current_index, column, 50),

                # Did this number participate in stored row/column events?
                recent_from_prefix(board_number_hits_prefix, current_index, number, 10),
                recent_from_prefix(board_number_hits_prefix, current_index, number, 50),

                # Shape involvement
                recent_from_prefix(shape_number_hits_prefix, current_index, number, 10),
                recent_from_prefix(shape_number_hits_prefix, current_index, number, 50),
                recent_from_prefix(shape_number_area_prefix, current_index, number, 10),
                recent_from_prefix(shape_number_area_prefix, current_index, number, 50),

                # Was this number a shape center recently?
                recent_from_prefix(shape_centers_prefix, current_index, number, 20),
                recent_from_prefix(shape_centers_prefix, current_index, number, 100),

                # Completed movement-path information
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
        ]

        # ------------------------------------------------------------
        # Build ML rows
        # ------------------------------------------------------------

        X_rows = []
        y_rows = []
        future_count_rows = []
        row_draw_indices = []

        decision_indices = list(
            range(
                min_history,
                len(draws) - horizon,
                decision_step,
            )
        )

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

        split_draw_index = int(len(draws) * (1 - test_ratio))

        train_mask = row_draw_indices < split_draw_index
        test_mask = row_draw_indices >= split_draw_index

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        future_counts_test = future_counts[test_mask]
        test_draw_indices = row_draw_indices[test_mask]

        self.stdout.write("")
        self.stdout.write(f"Training rows: {len(X_train):,}")
        self.stdout.write(f"Testing rows: {len(X_test):,}")

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
            self.stdout.write("")
            self.stdout.write("Try one of these:")
            self.stdout.write("  horizon 1  -> use --target-hits 1")
            self.stdout.write("  horizon 5  -> use --target-hits 2")
            self.stdout.write("  horizon 10 -> use --target-hits 3")
            return

        log_step("Fitting 10-game model...")
        model.fit(X_train, y_train)
        log_done("Model fitting complete")

        # ------------------------------------------------------------
        # Test model
        # ------------------------------------------------------------

        log_step("Testing 10-game model...")

        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, zero_division=0)
        recall = recall_score(y_test, test_predictions, zero_division=0)

        unique_test_draw_indices = sorted(set(test_draw_indices.tolist()))

        raw_pick_total_hits = []
        spread_pick_total_hits = []
        hybrid_pick_total_hits = []
        random_pick_total_hits = []


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


        def select_raw_pick(draw_probs, numbers, pick):
            ranked_indices = np.argsort(draw_probs)[::-1]
            return ranked_indices[:pick]


        def select_spread_pick(
            draw_probs,
            numbers,
            pick,
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

                if len(selected_indices) >= pick:
                    break

            # Fallback: if constraints are too strict, fill remaining spots with best unused numbers.
            if len(selected_indices) < pick:
                selected_set = set(selected_indices)

                for index in ranked_indices:
                    if index in selected_set:
                        continue

                    selected_indices.append(index)
                    selected_set.add(index)

                    if len(selected_indices) >= pick:
                        break

            return np.array(selected_indices[:pick])


        def select_hybrid_pick(draw_probs, numbers, pick):
            ranked_indices = np.argsort(draw_probs)[::-1]

            # Keep strongest 4 raw picks.
            locked_count = min(4, pick)
            locked_indices = list(ranked_indices[:locked_count])

            selected_indices = locked_indices[:]

            row_counts = {}
            column_counts = {}

            for index in selected_indices:
                number = int(numbers[index])
                row = get_number_row(number)
                column = get_number_column(number)

                row_counts[row] = row_counts.get(row, 0) + 1
                column_counts[column] = column_counts.get(column, 0) + 1

            # Fill the rest with spread logic.
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

                if len(selected_indices) >= pick:
                    break

            # Fallback fill.
            if len(selected_indices) < pick:
                selected_set = set(selected_indices)

                for index in ranked_indices:
                    if index in selected_set:
                        continue

                    selected_indices.append(index)
                    selected_set.add(index)

                    if len(selected_indices) >= pick:
                        break

            return np.array(selected_indices[:pick])

        rng = np.random.default_rng(seed=42)

        for counter, draw_index in enumerate(unique_test_draw_indices, start=1):
            if counter % 250 == 0:
                self.stdout.write(
                    f"  tested {counter:,}/{len(unique_test_draw_indices):,} decision points..."
                )

            mask = test_draw_indices == draw_index

            draw_probs = test_probabilities[mask]
            draw_future_counts = future_counts_test[mask]
            draw_features = X_test[mask]

            numbers = draw_features[:, 0].astype(int)

            raw_indices = select_raw_pick(draw_probs, numbers, pick)
            spread_indices = select_spread_pick(draw_probs, numbers, pick)
            hybrid_indices = select_hybrid_pick(draw_probs, numbers, pick)

            raw_hits = int(draw_future_counts[raw_indices].sum())
            spread_hits = int(draw_future_counts[spread_indices].sum())
            hybrid_hits = int(draw_future_counts[hybrid_indices].sum())

            raw_pick_total_hits.append(raw_hits)
            spread_pick_total_hits.append(spread_hits)
            hybrid_pick_total_hits.append(hybrid_hits)

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
        random_pick_hits = float(np.mean(random_pick_total_hits))

        theoretical_baseline = pick * horizon * 0.25

        raw_lift = raw_pick_hits - theoretical_baseline
        spread_lift = spread_pick_hits - theoretical_baseline
        hybrid_lift = hybrid_pick_hits - theoretical_baseline
        random_lift = random_pick_hits - theoretical_baseline

        # Keep model_pick_hits as the best-performing AI mode for existing DB/frontend fields.
        mode_scores = {
            "raw": raw_pick_hits,
            "spread": spread_pick_hits,
            "hybrid": hybrid_pick_hits,
        }

        best_mode = max(mode_scores, key=mode_scores.get)
        model_pick_hits = mode_scores[best_mode]
        lift = model_pick_hits - theoretical_baseline

        # Baseline probability that a number hits at least target_hits times in horizon games.
        baseline_target_probability = 0

        for hits in range(target_hits, horizon + 1):
            baseline_target_probability += (
                comb(horizon, hits)
                * (0.25 ** hits)
                * (0.75 ** (horizon - hits))
            )

        log_done("Testing complete")

        # ------------------------------------------------------------
        # Score latest draw
        # ------------------------------------------------------------

        log_step("Scoring latest draw...")

        latest_index = len(draws) - 1
        latest_numbers = draw_sets[latest_index]
        previous_numbers = draw_sets[latest_index - 1]

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

            latest_features.append(latest_row_features)

        latest_X = np.array(latest_features, dtype=np.float32)
        latest_probabilities = model.predict_proba(latest_X)[:, 1]

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

        log_done("Latest draw scored")

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

        for start in range(0, len(hybrid_pick_total_hits), block_size):
            block = hybrid_pick_total_hits[start:start + block_size]

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

        result = KinoAIResult.objects.create(
            model_name="number_ai_10game_v2",
            train_draws=len(set(row_draw_indices[train_mask].tolist())),
            test_draws=len(unique_test_draw_indices),
            baseline_top20_hits=theoretical_baseline,
            model_top20_hits=model_pick_hits,
            lift=lift,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            data={
                "best_mode": best_mode,

                "raw_pick_average_hits": raw_pick_hits,
                "spread_pick_average_hits": spread_pick_hits,
                "hybrid_pick_average_hits": hybrid_pick_hits,
                "random_pick_average_hits": random_pick_hits,

                "raw_lift": raw_lift,
                "spread_lift": spread_lift,
                "hybrid_lift": hybrid_lift,
                "random_lift": random_lift,

                "raw_pick_hits_by_test_decision": raw_pick_total_hits[-200:],
                "spread_pick_hits_by_test_decision": spread_pick_total_hits[-200:],
                "hybrid_pick_hits_by_test_decision": hybrid_pick_total_hits[-200:],
                "random_pick_hits_by_test_decision": random_pick_total_hits[-200:],

                # Main chart should show best mode / hybrid mode.
                "model_pick_hits_by_test_decision": (
                    raw_pick_total_hits[-200:]
                    if best_mode == "raw"
                    else spread_pick_total_hits[-200:]
                    if best_mode == "spread"
                    else hybrid_pick_total_hits[-200:]
                ),

                # Backward-compatible keys.
                "model_top20_hits_by_test_decision": (
                    raw_pick_total_hits[-200:]
                    if best_mode == "raw"
                    else spread_pick_total_hits[-200:]
                    if best_mode == "spread"
                    else hybrid_pick_total_hits[-200:]
                ),
                "random_top20_hits_by_test_decision": random_pick_total_hits[-200:],
                "random_top20_average_hits": random_pick_hits,
            },
        )

        log_done(f"Saved AI result ID: {result.id}")

        # ------------------------------------------------------------
        # Terminal summary
        # ------------------------------------------------------------

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO 10-game AI training finished."))
        self.stdout.write(f"AI Result ID: {result.id}")
        self.stdout.write(f"Train decision points: {result.train_draws}")
        self.stdout.write(f"Test decision points: {result.test_draws}")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Precision: {precision:.4f}")
        self.stdout.write(f"Recall: {recall:.4f}")

        self.stdout.write(f"Theoretical random top {pick} hits: {theoretical_baseline:.3f}")
        self.stdout.write(f"Simulated random top {pick} hits: {random_pick_hits:.3f}")
        self.stdout.write("")
        self.stdout.write(f"Raw AI top {pick} hits: {raw_pick_hits:.3f} ({raw_lift:+.3f})")
        self.stdout.write(f"Spread AI top {pick} hits: {spread_pick_hits:.3f} ({spread_lift:+.3f})")
        self.stdout.write(f"Hybrid AI top {pick} hits: {hybrid_pick_hits:.3f} ({hybrid_lift:+.3f})")
        self.stdout.write("")
        self.stdout.write(f"Best mode: {best_mode}")
        self.stdout.write(f"Best mode lift vs random simulation: {model_pick_hits - random_pick_hits:+.3f}")
        self.stdout.write("")
        self.stdout.write(f"Top {pick} latest 10-game number scores:")
        for item in latest_scores[:pick]:
            self.stdout.write(
                f"#{item['rank']:02d} Number {item['number']:02d} | "
                f"{item['probability_percent']:.3f}% | "
                f"above target baseline {item['above_baseline']:+.3f}%"
            )