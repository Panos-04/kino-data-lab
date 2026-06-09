import os
from pathlib import Path

import joblib
import numpy as np

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kino.models import KinoDraw, KinoAIResult


class Command(BaseCommand):
    help = "Train a simple ML model to score KINO numbers for next-draw appearance"

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-history",
            type=int,
            default=100,
            help="Minimum previous draws required before creating training examples",
        )

        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.2,
            help="Last percentage of draws used for time-based testing",
        )

        parser.add_argument(
            "--save-model",
            action="store_true",
            help="Save trained sklearn model to backend/ml_models",
        )

    def handle(self, *args, **options):
        min_history = options["min_history"]
        test_ratio = options["test_ratio"]
        save_model = options["save_model"]

        draws = list(KinoDraw.objects.order_by("draw_time", "draw_id"))

        if len(draws) < min_history + 100:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough draws. Have {len(draws)}, need at least {min_history + 100}."
                )
            )
            return

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Training KINO Number AI v1..."))
        self.stdout.write(f"Total draws: {len(draws)}")
        self.stdout.write(f"Min history: {min_history}")
        self.stdout.write(f"Test ratio: {test_ratio}")

        draw_sets = [set(draw.numbers) for draw in draws]
        draw_ids = [draw.draw_id for draw in draws]

        # prefix_counts[i][n] = how many times number n appeared before draw index i
        prefix_counts = np.zeros((len(draws) + 1, 81), dtype=np.int32)

        for index, numbers in enumerate(draw_sets):
            prefix_counts[index + 1] = prefix_counts[index]

            for number in numbers:
                prefix_counts[index + 1][number] += 1

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
        ]

        X_rows = []
        y_rows = []
        row_draw_indices = []

        last_seen = [-1] * 81

        def count_in_window(current_index, number, window_size):
            start_index = max(0, current_index - window_size + 1)
            end_index = current_index + 1

            return (
                prefix_counts[end_index][number]
                - prefix_counts[start_index][number]
            )

        def number_row(number):
            return (number - 1) // 10 + 1

        def number_column(number):
            return (number - 1) % 10 + 1

        def row_numbers(row):
            return list(range((row - 1) * 10 + 1, row * 10 + 1))

        def column_numbers(column):
            return [column + row_index * 10 for row_index in range(0, 8)]

        for current_index in range(0, len(draws) - 1):
            current_numbers = draw_sets[current_index]

            for number in current_numbers:
                last_seen[number] = current_index

            if current_index < min_history:
                continue

            previous_numbers = draw_sets[current_index - 1]
            next_numbers = draw_sets[current_index + 1]

            for number in range(1, 81):
                row = number_row(number)
                column = number_column(number)

                count_last_5 = count_in_window(current_index, number, 5)
                count_last_10 = count_in_window(current_index, number, 10)
                count_last_20 = count_in_window(current_index, number, 20)
                count_last_50 = count_in_window(current_index, number, 50)
                count_last_100 = count_in_window(current_index, number, 100)

                if last_seen[number] == -1:
                    gap_since_seen = 999
                else:
                    gap_since_seen = current_index - last_seen[number]

                row_hits_current = len(
                    current_numbers.intersection(row_numbers(row))
                )

                column_hits_current = len(
                    current_numbers.intersection(column_numbers(column))
                )

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
                    gap_since_seen,
                    1 if number in current_numbers else 0,
                    1 if number in previous_numbers else 0,
                    row_hits_current,
                    column_hits_current,
                ]

                target = 1 if number in next_numbers else 0

                X_rows.append(features)
                y_rows.append(target)
                row_draw_indices.append(current_index)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int8)
        row_draw_indices = np.array(row_draw_indices, dtype=np.int32)

        split_draw_index = int(len(draws) * (1 - test_ratio))

        train_mask = row_draw_indices < split_draw_index
        test_mask = row_draw_indices >= split_draw_index

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]
        test_draw_indices = row_draw_indices[test_mask]

        self.stdout.write("")
        self.stdout.write(f"Training rows: {len(X_train)}")
        self.stdout.write(f"Testing rows: {len(X_test)}")

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Fitting model..."))
        model.fit(X_train, y_train)

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Testing model..."))

        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, zero_division=0)
        recall = recall_score(y_test, test_predictions, zero_division=0)

        # Important metric:
        # For each historical test draw, choose the model's top 20 numbers.
        # Count how many actually hit in the next draw.
        top20_hits = []
        block_size = 250
        test_blocks = []

        for start in range(0, len(top20_hits), block_size):
            block = top20_hits[start:start + block_size]

            if not block:
                continue

            block_average = float(np.mean(block))

            test_blocks.append({
                "start": start,
                "end": start + len(block) - 1,
                "draws": len(block),
                "average_hits": round(block_average, 4),
                "lift": round(block_average - 5.0, 4),
            })
        unique_test_draw_indices = sorted(set(test_draw_indices.tolist()))

        for draw_index in unique_test_draw_indices:
            mask = test_draw_indices == draw_index

            draw_probs = test_probabilities[mask]
            draw_targets = y_test[mask]
            draw_features = X_test[mask]

            # number is first feature
            numbers = draw_features[:, 0].astype(int)

            ranked_indices = np.argsort(draw_probs)[::-1]
            top_indices = ranked_indices[:20]

            selected_numbers = numbers[top_indices]
            selected_targets = draw_targets[top_indices]

            hits = int(selected_targets.sum())
            top20_hits.append(hits)

        model_top20_hits = float(np.mean(top20_hits)) if top20_hits else 0.0
        baseline_top20_hits = 5.0
        lift = model_top20_hits - baseline_top20_hits

        # Score latest draw for actual next-draw candidate probabilities
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

            seen_indices = [
                index
                for index in range(latest_index, max(-1, latest_index - 300), -1)
                if number in draw_sets[index]
            ]

            if seen_indices:
                gap_since_seen = latest_index - seen_indices[0]
            else:
                gap_since_seen = 999

            row_hits_current = len(
                latest_numbers.intersection(row_numbers(row))
            )

            column_hits_current = len(
                latest_numbers.intersection(column_numbers(column))
            )

            latest_features.append([
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
                gap_since_seen,
                1 if number in latest_numbers else 0,
                1 if number in previous_numbers else 0,
                row_hits_current,
                column_hits_current,
            ])

        latest_X = np.array(latest_features, dtype=np.float32)
        latest_probabilities = model.predict_proba(latest_X)[:, 1]

        latest_scores = []

        for index, probability in enumerate(latest_probabilities):
            number = index + 1
            row = number_row(number)
            column = number_column(number)

            latest_scores.append({
                "number": number,
                "row": row,
                "column": column,
                "probability": round(float(probability), 5),
                "probability_percent": round(float(probability) * 100, 3),
                "above_baseline": round((float(probability) - 0.25) * 100, 3),
                "count_last_10": int(count_in_window(latest_index, number, 10)),
                "count_last_20": int(count_in_window(latest_index, number, 20)),
                "count_last_50": int(count_in_window(latest_index, number, 50)),
            })

        latest_scores = sorted(
            latest_scores,
            key=lambda item: item["probability"],
            reverse=True,
        )

        for rank, item in enumerate(latest_scores, start=1):
            item["rank"] = rank

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

        model_path = None

        if save_model:
            model_dir = Path(settings.BASE_DIR) / "ml_models"
            os.makedirs(model_dir, exist_ok=True)

            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"kino_number_ai_v1_{timestamp}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "created_at": timezone.now().isoformat(),
                },
                model_path,
            )

            model_path = str(model_path)
        random_top20_hits = []

        rng = np.random.default_rng(seed=42)

        for draw_index in unique_test_draw_indices:
            mask = test_draw_indices == draw_index

            draw_targets = y_test[mask]

            random_indices = rng.choice(
                len(draw_targets),
                size=20,
                replace=False,
            )

            random_hits = int(draw_targets[random_indices].sum())
            random_top20_hits.append(random_hits)

        random_top20_average_hits = float(np.mean(random_top20_hits))
        result = KinoAIResult.objects.create(
            model_name="number_ai_v1",
            train_draws=len(set(row_draw_indices[train_mask].tolist())),
            test_draws=len(unique_test_draw_indices),
            baseline_top20_hits=baseline_top20_hits,
            model_top20_hits=model_top20_hits,
            lift=lift,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            data={
                "latest_draw_id": draw_ids[-1],
                "split_draw_id": draw_ids[split_draw_index],
                "training_rows": int(len(X_train)),
                "testing_rows": int(len(X_test)),
                "top20_hits_by_test_draw": top20_hits[-200:],
                "latest_scores": latest_scores,
                "top20_latest_scores": latest_scores[:20],
                "feature_importance": feature_importance,
                "model_path": model_path,
                "created_at": timezone.now().isoformat(),
                "random_top20_hits_by_test_draw": random_top20_hits[-200:],
                "random_top20_average_hits": random_top20_average_hits,
                "test_blocks": test_blocks,
            },
        )



        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO Number AI training finished."))
        self.stdout.write(f"AI Result ID: {result.id}")
        self.stdout.write(f"Train draws: {result.train_draws}")
        self.stdout.write(f"Test draws: {result.test_draws}")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Precision: {precision:.4f}")
        self.stdout.write(f"Recall: {recall:.4f}")
        self.stdout.write("")
        self.stdout.write("Main test metric:")
        self.stdout.write(f"Random baseline top 20 hits: {baseline_top20_hits:.3f}")
        self.stdout.write(f"Model top 20 average hits: {model_top20_hits:.3f}")
        self.stdout.write(f"Lift above baseline: {lift:+.3f}")
        self.stdout.write(f"Random simulated top 20 hits: {random_top20_average_hits:.3f}")
        self.stdout.write("")
        self.stdout.write("Top 20 latest number scores:")
        for item in latest_scores[:20]:
            self.stdout.write(
                f"#{item['rank']:02d} Number {item['number']:02d} | "
                f"{item['probability_percent']:.3f}% | "
                f"above baseline {item['above_baseline']:+.3f}%"
            )