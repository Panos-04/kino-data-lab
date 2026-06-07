from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import KinoWindowAnalysis
from .serializers import KinoWindowAnalysisSerializer
from .models import KinoDraw, KinoWindowAnalysis
from .services.window_relations import build_window_relations
from collections import Counter

@api_view(["GET"])
def window_analysis_list(request):
    window_size = request.GET.get("window_size")
    step_size = request.GET.get("step_size")
    limit = int(request.GET.get("limit", 50))

    queryset = KinoWindowAnalysis.objects.prefetch_related("numbers").order_by("start_time")

    if window_size:
        queryset = queryset.filter(window_size=window_size)

    if step_size:
        queryset = queryset.filter(step_size=step_size)

    queryset = queryset[:limit]

    serializer = KinoWindowAnalysisSerializer(queryset, many=True)

    return Response({
        "count": len(serializer.data),
        "results": serializer.data,
    })

@api_view(["GET"])
def window_relations_detail(request, window_id):
    top_limit = int(request.GET.get("top", 10))

    try:
        analysis = (
            KinoWindowAnalysis.objects
            .prefetch_related("numbers")
            .get(id=window_id)
        )
    except KinoWindowAnalysis.DoesNotExist:
        return Response(
            {"detail": "Window analysis not found."},
            status=404,
        )

    draws = list(
        KinoDraw.objects
        .filter(
            draw_time__gte=analysis.start_time,
            draw_time__lte=analysis.end_time,
        )
        .order_by("draw_time")
    )

    relations = build_window_relations(
        analysis=analysis,
        draws=draws,
        top_limit=top_limit,
        max_workers=6,
    )

    return Response({
        "window_id": analysis.id,
        "window_size": analysis.window_size,
        "step_size": analysis.step_size,
        "start_draw_id": analysis.start_draw.draw_id,
        "end_draw_id": analysis.end_draw.draw_id,
        "relations": relations,
    })

@api_view(["GET"])
def number_relations_detail(request, window_id, number):
    top_limit = int(request.GET.get("top", 15))

    try:
        analysis = KinoWindowAnalysis.objects.get(id=window_id)
    except KinoWindowAnalysis.DoesNotExist:
        return Response(
            {"detail": "Window analysis not found."},
            status=404,
        )

    draws = list(
        KinoDraw.objects
        .filter(
            draw_time__gte=analysis.start_time,
            draw_time__lte=analysis.end_time,
        )
        .order_by("draw_time")
    )

    if not draws:
        return Response(
            {"detail": "No draws found for this window."},
            status=404,
        )

    half = len(draws) // 2
    first_half_draws = draws[:half]
    second_half_draws = draws[half:]

    def count_connections(draw_list):
        counter = Counter()
        anchor_hits = 0

        for draw in draw_list:
            draw_numbers = draw.numbers

            if number not in draw_numbers:
                continue

            anchor_hits += 1

            for related_number in draw_numbers:
                if related_number == number:
                    continue

                counter[related_number] += 1

        return anchor_hits, counter

    total_anchor_hits, total_counter = count_connections(draws)
    first_anchor_hits, first_counter = count_connections(first_half_draws)
    second_anchor_hits, second_counter = count_connections(second_half_draws)

    related_numbers = []

    for related_number, total_count in total_counter.most_common(top_limit):
        first_count = first_counter.get(related_number, 0)
        second_count = second_counter.get(related_number, 0)

        related_numbers.append({
            "number": related_number,
            "total_count": total_count,
            "first_half_count": first_count,
            "second_half_count": second_count,
            "change": second_count - first_count,
        })

    return Response({
        "window_id": analysis.id,
        "window_size": analysis.window_size,
        "step_size": analysis.step_size,
        "start_draw_id": analysis.start_draw.draw_id,
        "end_draw_id": analysis.end_draw.draw_id,
        "selected_number": number,
        "anchor_appearances": total_anchor_hits,
        "first_half_anchor_appearances": first_anchor_hits,
        "second_half_anchor_appearances": second_anchor_hits,
        "related_numbers": related_numbers,
    })

from collections import Counter


@api_view(["GET"])
def general_relations_detail(request, window_id):
    top_limit = int(request.GET.get("top", 20))
    bottom_limit = int(request.GET.get("bottom", 20))

    try:
        analysis = (
            KinoWindowAnalysis.objects
            .prefetch_related("numbers")
            .get(id=window_id)
        )
    except KinoWindowAnalysis.DoesNotExist:
        return Response(
            {"detail": "Window analysis not found."},
            status=404,
        )

    draws = list(
        KinoDraw.objects
        .filter(
            draw_time__gte=analysis.start_time,
            draw_time__lte=analysis.end_time,
        )
        .order_by("draw_time")
    )

    if not draws:
        return Response(
            {"detail": "No draws found for this window."},
            status=404,
        )

    window_numbers = list(analysis.numbers.all())
    expected = analysis.window_size * 0.25

    hot_numbers = sorted(
        window_numbers,
        key=lambda item: item.count,
        reverse=True,
    )[:5]

    cold_numbers = sorted(
        window_numbers,
        key=lambda item: item.count,
    )[:5]

    selected = {item.number for item in hot_numbers + cold_numbers}

    middle_numbers = sorted(
        [item for item in window_numbers if item.number not in selected],
        key=lambda item: abs(item.count - expected),
    )[:10]

    anchors = []

    for item in hot_numbers:
        anchors.append({
            "number": item.number,
            "type": "hot",
            "heat": item.count,
        })

    for item in cold_numbers:
        anchors.append({
            "number": item.number,
            "type": "cold",
            "heat": item.count,
        })

    for item in middle_numbers:
        anchors.append({
            "number": item.number,
            "type": "middle",
            "heat": item.count,
        })

    half = len(draws) // 2
    first_half_draws = draws[:half]
    second_half_draws = draws[half:]

    def count_connections(anchor_number, draw_list):
        counter = Counter()
        anchor_hits = 0

        for draw in draw_list:
            draw_numbers = draw.numbers

            if anchor_number not in draw_numbers:
                continue

            anchor_hits += 1

            for related_number in draw_numbers:
                if related_number == anchor_number:
                    continue

                counter[related_number] += 1

        return anchor_hits, counter

    anchor_results = []

    for anchor in anchors:
        anchor_number = anchor["number"]

        total_hits, total_counter = count_connections(anchor_number, draws)
        first_hits, first_counter = count_connections(anchor_number, first_half_draws)
        second_hits, second_counter = count_connections(anchor_number, second_half_draws)

        all_related = []

        for related_number in range(1, 81):
            if related_number == anchor_number:
                continue

            total_count = total_counter.get(related_number, 0)
            first_count = first_counter.get(related_number, 0)
            second_count = second_counter.get(related_number, 0)

            all_related.append({
                "number": related_number,
                "total_count": total_count,
                "first_half_count": first_count,
                "second_half_count": second_count,
                "change": second_count - first_count,
            })

        strongest = sorted(
            all_related,
            key=lambda item: (-item["total_count"], item["number"])
        )[:top_limit]

        weakest = sorted(
            all_related,
            key=lambda item: (item["total_count"], item["number"])
        )[:bottom_limit]

        anchor_results.append({
            "anchor_number": anchor_number,
            "anchor_type": anchor["type"],
            "anchor_heat": anchor["heat"],
            "anchor_appearances": total_hits,
            "first_half_anchor_appearances": first_hits,
            "second_half_anchor_appearances": second_hits,
            "strongest_connections": strongest,
            "weakest_connections": weakest,
        })

    return Response({
        "window_id": analysis.id,
        "window_size": analysis.window_size,
        "step_size": analysis.step_size,
        "start_draw_id": analysis.start_draw.draw_id,
        "end_draw_id": analysis.end_draw.draw_id,
        "expected_heat": expected,
        "anchors": anchor_results,
    })


def combo_baseline_distribution(pick_count):
    """
    Exact hypergeometric baseline for KINO:
    20 winning numbers from 80 total.
    """
    from math import comb

    total = comb(80, pick_count)
    distribution = {}

    for hits in range(0, pick_count + 1):
        if hits > 20 or pick_count - hits > 60:
            probability = 0
        else:
            probability = (
                comb(20, hits) * comb(60, pick_count - hits)
            ) / total

        distribution[hits] = probability * 100

    return distribution


@api_view(["GET"])
def combo_test_api(request):
    strategy = request.GET.get("strategy", "cold")
    window_size = int(request.GET.get("window", 20))
    step_size = int(request.GET.get("step", 10))
    pick_count = int(request.GET.get("pick", 5))
    future_size = int(request.GET.get("future", 1))

    if strategy not in ["cold", "hot", "middle"]:
        return Response(
            {"detail": "Invalid strategy. Use cold, hot, or middle."},
            status=400,
        )

    windows = (
        KinoWindowAnalysis.objects
        .filter(window_size=window_size, step_size=step_size)
        .prefetch_related("numbers")
        .order_by("start_time")
    )

    hit_distribution = Counter()
    tested_draws = 0
    skipped_windows = 0
    best_results = []

    for window in windows:
        numbers = list(window.numbers.all())
        expected = window.window_size * 0.25

        if strategy == "cold":
            selected = sorted(
                numbers,
                key=lambda item: (item.count, item.number)
            )[:pick_count]

        elif strategy == "hot":
            selected = sorted(
                numbers,
                key=lambda item: (-item.count, item.number)
            )[:pick_count]

        else:
            selected = sorted(
                numbers,
                key=lambda item: (abs(item.count - expected), item.number)
            )[:pick_count]

        combo = [item.number for item in selected]

        future_draws = list(
            KinoDraw.objects
            .filter(draw_time__gt=window.end_time)
            .order_by("draw_time")[:future_size]
        )

        if len(future_draws) < future_size:
            skipped_windows += 1
            continue

        for draw in future_draws:
            hit_numbers = sorted(set(combo).intersection(draw.numbers))
            hit_count = len(hit_numbers)

            hit_distribution[hit_count] += 1
            tested_draws += 1

            if hit_count >= 4:
                best_results.append({
                    "window_id": window.id,
                    "draw_id": draw.draw_id,
                    "combo": combo,
                    "draw_numbers": draw.numbers,
                    "hit_count": hit_count,
                    "hit_numbers": hit_numbers,
                })

    baseline = combo_baseline_distribution(pick_count)

    distribution = []

    for hits in range(0, pick_count + 1):
        count = hit_distribution[hits]
        percentage = (count / tested_draws) * 100 if tested_draws else 0

        distribution.append({
            "hits": hits,
            "count": count,
            "percentage": round(percentage, 3),
            "baseline_percentage": round(baseline[hits], 3),
            "difference": round(percentage - baseline[hits], 3),
        })

    four_plus_count = sum(
        hit_distribution[hits]
        for hits in range(4, pick_count + 1)
    )

    four_plus_rate = (
        (four_plus_count / tested_draws) * 100
        if tested_draws
        else 0
    )

    four_plus_baseline = sum(
        baseline[hits]
        for hits in range(4, pick_count + 1)
    )

    return Response({
        "strategy": strategy,
        "window_size": window_size,
        "step_size": step_size,
        "pick_count": pick_count,
        "future_size": future_size,
        "tested_draws": tested_draws,
        "skipped_windows": skipped_windows,
        "distribution": distribution,
        "four_plus": {
            "count": four_plus_count,
            "percentage": round(four_plus_rate, 3),
            "baseline_percentage": round(four_plus_baseline, 3),
            "difference": round(four_plus_rate - four_plus_baseline, 3),
        },
        "best_results": best_results[:20],
    })

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

@api_view(["GET"])
def pattern_test_api(request):
    row_threshold = int(request.GET.get("row_threshold", 6))
    column_threshold = int(request.GET.get("column_threshold", 5))
    limit = int(request.GET.get("limit", 20))

    draws = list(KinoDraw.objects.order_by("draw_time"))

    if not draws:
        return Response({
            "total_draws": 0,
            "row_patterns": [],
            "column_patterns": [],
            "row_summary": [],
            "column_summary": [],
            "streaks": [],
        })

    def count_hits(draw_numbers, group_numbers):
        draw_set = set(draw_numbers)
        hit_numbers = sorted(draw_set.intersection(group_numbers))

        return {
            "count": len(hit_numbers),
            "numbers": hit_numbers,
        }

    from collections import Counter, defaultdict

    row_patterns = []
    column_patterns = []
    row_counter = Counter()
    column_counter = Counter()

    current_streaks = defaultdict(int)
    best_streaks = defaultdict(int)

    all_keys = (
        [("row", row_id) for row_id in ROWS.keys()] +
        [("column", col_id) for col_id in COLUMNS.keys()]
    )

    for draw in draws:
        active_keys = set()

        for row_id, row_numbers in ROWS.items():
            result = count_hits(draw.numbers, row_numbers)

            if result["count"] >= row_threshold:
                key = ("row", row_id)
                active_keys.add(key)
                row_counter[row_id] += 1

                row_patterns.append({
                    "draw_id": draw.draw_id,
                    "draw_time": draw.draw_time,
                    "type": "row",
                    "group": row_id,
                    "hit_count": result["count"],
                    "hit_numbers": result["numbers"],
                    "draw_numbers": draw.numbers,
                })

        for column_id, column_numbers in COLUMNS.items():
            result = count_hits(draw.numbers, column_numbers)

            if result["count"] >= column_threshold:
                key = ("column", column_id)
                active_keys.add(key)
                column_counter[column_id] += 1

                column_patterns.append({
                    "draw_id": draw.draw_id,
                    "draw_time": draw.draw_time,
                    "type": "column",
                    "group": column_id,
                    "hit_count": result["count"],
                    "hit_numbers": result["numbers"],
                    "draw_numbers": draw.numbers,
                })

        for key in all_keys:
            if key in active_keys:
                current_streaks[key] += 1
                best_streaks[key] = max(best_streaks[key], current_streaks[key])
            else:
                current_streaks[key] = 0

    total_draws = len(draws)

    row_summary = [
        {
            "group": row_id,
            "count": count,
            "percentage": round((count / total_draws) * 100, 3),
        }
        for row_id, count in row_counter.most_common()
    ]

    column_summary = [
        {
            "group": column_id,
            "count": count,
            "percentage": round((count / total_draws) * 100, 3),
        }
        for column_id, count in column_counter.most_common()
    ]

    streaks = []

    for key, streak in best_streaks.items():
        if streak <= 1:
            continue

        pattern_type, group_id = key

        streaks.append({
            "type": pattern_type,
            "group": group_id,
            "streak": streak,
        })

    streaks.sort(key=lambda item: item["streak"], reverse=True)

    return Response({
        "total_draws": total_draws,
        "row_threshold": row_threshold,
        "column_threshold": column_threshold,
        "row_pattern_count": len(row_patterns),
        "column_pattern_count": len(column_patterns),
        "row_pattern_percentage": round((len(row_patterns) / total_draws) * 100, 3),
        "column_pattern_percentage": round((len(column_patterns) / total_draws) * 100, 3),
        "row_summary": row_summary,
        "column_summary": column_summary,
        "streaks": streaks[:20],
        "row_patterns": row_patterns[:limit],
        "column_patterns": column_patterns[:limit],
    })