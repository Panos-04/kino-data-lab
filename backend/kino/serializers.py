from collections import Counter

from rest_framework import serializers
from .models import KinoDraw, KinoWindowAnalysis, KinoWindowNumber


class KinoWindowNumberSerializer(serializers.ModelSerializer):
    first_half_count = serializers.SerializerMethodField()
    second_half_count = serializers.SerializerMethodField()

    class Meta:
        model = KinoWindowNumber
        fields = [
            "number",
            "count",
            "percentage",
            "first_half_count",
            "second_half_count",
        ]

    def _get_split_counts(self, obj):
        cache_key = f"split_counts_{obj.analysis_id}"

        if cache_key in self.context:
            return self.context[cache_key]

        analysis = obj.analysis

        draws = list(
            KinoDraw.objects
            .filter(
                draw_time__gte=analysis.start_time,
                draw_time__lte=analysis.end_time,
            )
            .order_by("draw_time")
        )

        half = len(draws) // 2

        first_half = draws[:half]
        second_half = draws[half:]

        first_counter = Counter()
        second_counter = Counter()

        for draw in first_half:
            first_counter.update(draw.numbers)

        for draw in second_half:
            second_counter.update(draw.numbers)

        split_counts = {}

        for number in range(1, 81):
            split_counts[number] = {
                "first": first_counter.get(number, 0),
                "second": second_counter.get(number, 0),
            }

        self.context[cache_key] = split_counts
        return split_counts

    def get_first_half_count(self, obj):
        return self._get_split_counts(obj)[obj.number]["first"]

    def get_second_half_count(self, obj):
        return self._get_split_counts(obj)[obj.number]["second"]


class KinoWindowAnalysisSerializer(serializers.ModelSerializer):
    numbers = KinoWindowNumberSerializer(many=True, read_only=True)
    start_draw_id = serializers.IntegerField(source="start_draw.draw_id", read_only=True)
    end_draw_id = serializers.IntegerField(source="end_draw.draw_id", read_only=True)

    result_start_draw_id = serializers.SerializerMethodField()
    result_end_draw_id = serializers.SerializerMethodField()
    result_numbers = serializers.SerializerMethodField()

    class Meta:
        model = KinoWindowAnalysis
        fields = [
            "id",
            "window_size",
            "step_size",
            "start_draw_id",
            "end_draw_id",
            "start_time",
            "end_time",
            "numbers",
            "result_start_draw_id",
            "result_end_draw_id",
            "result_numbers",
        ]

    def _get_result_draws(self, obj):
        result_size = self.context.get("result_size", 10)

        return list(
            KinoDraw.objects
            .filter(draw_time__gt=obj.end_time)
            .order_by("draw_time")[:result_size]
        )

    def get_result_start_draw_id(self, obj):
        result_draws = self._get_result_draws(obj)

        if not result_draws:
            return None

        return result_draws[0].draw_id

    def get_result_end_draw_id(self, obj):
        result_draws = self._get_result_draws(obj)

        if not result_draws:
            return None

        return result_draws[-1].draw_id

    def get_result_numbers(self, obj):
        result_draws = self._get_result_draws(obj)
        result_size = len(result_draws)

        counter = Counter()

        for draw in result_draws:
            counter.update(draw.numbers)

        results = []

        for number in range(1, 81):
            count = counter.get(number, 0)
            percentage = round((count / result_size) * 100, 2) if result_size else 0

            results.append({
                "number": number,
                "count": count,
                "percentage": percentage,
            })

        return results