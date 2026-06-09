from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import KinoWindowAnalysis
from .serializers import KinoWindowAnalysisSerializer
from .models import KinoDraw, KinoWindowAnalysis, KinoShapeMovement
from .services.window_relations import build_window_relations
from collections import Counter, defaultdict
from .services.shape_detector import detect_shape, detect_all_shapes
from django.db.models import Count, Avg, Min, Max
from .models import KinoBoardPatternEvent
from .models import KinoAIResult

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
    def build_gap_summary(patterns):
        grouped = defaultdict(list)

        for pattern in patterns:
            key = (pattern["type"], pattern["group"])
            grouped[key].append(pattern)

        summaries = []

        for (pattern_type, group_id), items in grouped.items():
            items = sorted(items, key=lambda item: item["draw_time"])

            gaps = []

            for index in range(1, len(items)):
                previous = items[index - 1]
                current = items[index]

                draw_gap = current["draw_id"] - previous["draw_id"]

                gaps.append({
                    "from_draw_id": previous["draw_id"],
                    "to_draw_id": current["draw_id"],
                    "gap": draw_gap,
                })

            if gaps:
                gap_values = [item["gap"] for item in gaps]

                summaries.append({
                    "type": pattern_type,
                    "group": group_id,
                    "events": len(items),
                    "repeat_count": len(gaps),
                    "min_gap": min(gap_values),
                    "max_gap": max(gap_values),
                    "avg_gap": round(sum(gap_values) / len(gap_values), 2),
                    "examples": gaps[:10],
                })
            else:
                summaries.append({
                    "type": pattern_type,
                    "group": group_id,
                    "events": len(items),
                    "repeat_count": 0,
                    "min_gap": None,
                    "max_gap": None,
                    "avg_gap": None,
                    "examples": [],
                })

        summaries.sort(
            key=lambda item: (
                item["avg_gap"] if item["avg_gap"] is not None else 999999,
                -item["events"],
            )
        )

        return summaries
    

    def build_repeat_rate_summary(patterns, windows=(1, 5, 10, 20, 50, 100)):
        grouped = defaultdict(list)

        for pattern in patterns:
            key = (pattern["type"], pattern["group"])
            grouped[key].append(pattern)

        summaries = []

        for (pattern_type, group_id), items in grouped.items():
            items = sorted(items, key=lambda item: item["draw_id"])
            draw_ids = [item["draw_id"] for item in items]

            repeat_rates = []

            for window_size in windows:
                repeat_count = 0

                # Last event cannot be tested properly because no future repeat after it may exist in DB
                testable_events = max(len(draw_ids) - 1, 0)

                for index, draw_id in enumerate(draw_ids[:-1]):
                    future_draws = draw_ids[index + 1:]

                    repeated = any(
                        0 < future_draw_id - draw_id <= window_size
                        for future_draw_id in future_draws
                    )

                    if repeated:
                        repeat_count += 1

                rate = (
                    round((repeat_count / testable_events) * 100, 3)
                    if testable_events
                    else 0
                )

                repeat_rates.append({
                    "within_games": window_size,
                    "repeat_count": repeat_count,
                    "tested_events": testable_events,
                    "repeat_rate": rate,
                })

            summaries.append({
                "type": pattern_type,
                "group": group_id,
                "events": len(items),
                "repeat_rates": repeat_rates,
            })

        summaries.sort(
            key=lambda item: item["repeat_rates"][2]["repeat_rate"] if len(item["repeat_rates"]) > 2 else 0,
            reverse=True,
        )

        return summaries
    
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

    row_gap_summary = build_gap_summary(row_patterns)
    column_gap_summary = build_gap_summary(column_patterns)
    row_repeat_summary = build_repeat_rate_summary(row_patterns)
    column_repeat_summary = build_repeat_rate_summary(column_patterns)
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
        "row_gap_summary": row_gap_summary,
        "column_gap_summary": column_gap_summary,
        "row_patterns": row_patterns[:limit],
        "column_patterns": column_patterns[:limit],
        "row_repeat_summary": row_repeat_summary,
        "column_repeat_summary": column_repeat_summary,
    })

@api_view(["GET"])
def shape_pattern_test_api(request):
    shape = request.GET.get("shape", "all")
    min_hits_raw = request.GET.get("min_hits")
    limit = int(request.GET.get("limit", 20))

    min_hits = int(min_hits_raw) if min_hits_raw else None

    valid_shapes = [
        "all",
        "cross",
        "box_2x2",
        "l_shape",
        "vertical_4",
        "horizontal_4",
        "diagonal_down_4",
        "diagonal_up_4",
    ]

    if shape not in valid_shapes:
        return Response(
            {"detail": "Invalid shape."},
            status=400,
        )

    draws = list(KinoDraw.objects.order_by("draw_time"))

    shape_events = []
    shape_counter = Counter()
    center_counter = Counter()
    hit_count_counter = Counter()
    draws_with_shape = set()
    events_per_draw = Counter()

    for draw in draws:
        if shape == "all":
            events = detect_all_shapes(draw.numbers)
        else:
            events = detect_shape(
                draw_numbers=draw.numbers,
                shape_name=shape,
                min_hits=min_hits,
            )

        if events:
            draws_with_shape.add(draw.draw_id)

        events_per_draw[draw.draw_id] = len(events)

        for event in events:
            event_data = {
                **event,
                "draw_id": draw.draw_id,
                "draw_time": draw.draw_time,
                "draw_numbers": draw.numbers,
            }

            shape_events.append(event_data)
            shape_counter[event["shape"]] += 1
            center_counter[(event["shape"], event["center_number"])] += 1
            hit_count_counter[(event["shape"], event["hit_count"])] += 1

    total_draws = len(draws)
    total_events = len(shape_events)

    shape_summary = []

    for shape_name, count in shape_counter.most_common():
        draws_for_shape = {
            event["draw_id"]
            for event in shape_events
            if event["shape"] == shape_name
        }

        shape_summary.append({
            "shape": shape_name,
            "events": count,
            "draws_with_shape": len(draws_for_shape),
            "draw_percentage": round((len(draws_for_shape) / total_draws) * 100, 3) if total_draws else 0,
            "avg_events_per_draw": round(count / total_draws, 3) if total_draws else 0,
        })

    center_summary = [
        {
            "shape": shape_name,
            "center_number": center_number,
            "events": count,
        }
        for (shape_name, center_number), count in center_counter.most_common(30)
    ]

    hit_count_summary = defaultdict(list)

    for (shape_name, hit_count), count in hit_count_counter.items():
        hit_count_summary[shape_name].append({
            "hit_count": hit_count,
            "events": count,
        })

    hit_count_summary = {
        shape_name: sorted(rows, key=lambda row: row["hit_count"], reverse=True)
        for shape_name, rows in hit_count_summary.items()
    }

    most_events_in_one_draw = max(events_per_draw.values()) if events_per_draw else 0

    return Response({
        "shape": shape,
        "min_hits": min_hits,
        "total_draws": total_draws,
        "total_events": total_events,
        "draws_with_any_shape": len(draws_with_shape),
        "draws_with_any_shape_percentage": round((len(draws_with_shape) / total_draws) * 100, 3) if total_draws else 0,
        "avg_events_per_draw": round(total_events / total_draws, 3) if total_draws else 0,
        "most_events_in_one_draw": most_events_in_one_draw,
        "shape_summary": shape_summary,
        "center_summary": center_summary,
        "hit_count_summary": hit_count_summary,
        "examples": shape_events[:limit],
    })

@api_view(["GET"])
def shape_movements_api(request):
    shape = request.GET.get("shape", "cross")
    mode = request.GET.get("mode", "one-to-one")
    min_hits = int(request.GET.get("min_hits", 4))
    future_window = int(request.GET.get("future", 10))
    limit = int(request.GET.get("limit", 30))

    queryset = KinoShapeMovement.objects.filter(
        shape=shape,
        mode=mode,
        min_hits=min_hits,
        future_window=future_window,
    )

    total_movements = queryset.count()

    vector_summary = list(
        queryset.values("delta_row", "delta_col")
        .annotate(count=Count("id"))
        .order_by("-count")[:limit]
    )

    gap_summary = list(
        queryset.values("gap")
        .annotate(count=Count("id"))
        .order_by("gap")
    )

    center_summary = list(
        queryset.values("from_center", "to_center")
        .annotate(count=Count("id"))
        .order_by("-count")[:limit]
    )

    examples = list(
        queryset.order_by("-id")
        .values(
            "id",
            "from_draw_id",
            "to_draw_id",
            "from_center",
            "to_center",
            "delta_row",
            "delta_col",
            "gap",
            "overlap_score",
            "distance_score",
        )[:limit]
    )

    def with_percentage(rows):
        output = []

        for row in rows:
            count = row["count"]
            percentage = (count / total_movements) * 100 if total_movements else 0

            output.append({
                **row,
                "percentage": round(percentage, 3),
            })

        return output

    return Response({
        "shape": shape,
        "mode": mode,
        "min_hits": min_hits,
        "future_window": future_window,
        "total_movements": total_movements,
        "vector_summary": with_percentage(vector_summary),
        "gap_summary": with_percentage(gap_summary),
        "center_summary": with_percentage(center_summary),
        "examples": examples,
    })

@api_view(["GET"])
def board_pattern_events_api(request):
    row_threshold = int(request.GET.get("row_threshold", 6))
    column_threshold = int(request.GET.get("column_threshold", 5))
    limit = int(request.GET.get("limit", 30))

    queryset = KinoBoardPatternEvent.objects.filter(
        threshold__in=[row_threshold, column_threshold]
    ).select_related("draw")

    total_events = queryset.count()

    row_events = queryset.filter(pattern_type="row")
    column_events = queryset.filter(pattern_type="column")

    row_summary = list(
        row_events.values("group_number")
        .annotate(
            count=Count("id"),
            avg_hits=Avg("hit_count"),
            min_hits=Min("hit_count"),
            max_hits=Max("hit_count"),
        )
        .order_by("-count")
    )

    column_summary = list(
        column_events.values("group_number")
        .annotate(
            count=Count("id"),
            avg_hits=Avg("hit_count"),
            min_hits=Min("hit_count"),
            max_hits=Max("hit_count"),
        )
        .order_by("-count")
    )

    hit_count_summary = list(
        queryset.values("pattern_type", "hit_count")
        .annotate(count=Count("id"))
        .order_by("pattern_type", "-hit_count")
    )

    recent_events_raw = list(
        queryset.order_by("-draw__draw_time")
        .values(
            "id",
            "draw__draw_id",
            "draw__draw_time",
            "pattern_type",
            "group_number",
            "group_numbers",
            "hit_numbers",
            "hit_count",
            "threshold",
        )[:limit]
    )

    recent_events = []

    for event in recent_events_raw:
        recent_events.append({
            "id": event["id"],
            "draw_id": event["draw__draw_id"],
            "draw_time": event["draw__draw_time"],
            "pattern_type": event["pattern_type"],
            "group_number": event["group_number"],
            "group_numbers": event["group_numbers"],
            "hit_numbers": event["hit_numbers"],
            "hit_count": event["hit_count"],
            "threshold": event["threshold"],
        })

    def add_percentage(rows, denominator):
        output = []

        for row in rows:
            count = row["count"]
            percentage = (count / denominator) * 100 if denominator else 0

            output.append({
                **row,
                "percentage": round(percentage, 3),
            })

        return output

    return Response({
        "row_threshold": row_threshold,
        "column_threshold": column_threshold,
        "total_events": total_events,
        "row_event_count": row_events.count(),
        "column_event_count": column_events.count(),
        "row_summary": add_percentage(row_summary, row_events.count()),
        "column_summary": add_percentage(column_summary, column_events.count()),
        "hit_count_summary": add_percentage(hit_count_summary, total_events),
        "recent_events": recent_events,
    })

@api_view(["GET"])
def ai_results_api(request):
    result = KinoAIResult.objects.order_by("-created_at").first()

    if result is None:
        return Response({
            "has_result": False,
            "message": "No AI result found. Run python manage.py train_number_ai first.",
        })

    return Response({
        "has_result": True,
        "id": result.id,
        "model_name": result.model_name,
        "train_draws": result.train_draws,
        "test_draws": result.test_draws,
        "baseline_top20_hits": result.baseline_top20_hits,
        "model_top20_hits": result.model_top20_hits,
        "lift": result.lift,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "created_at": result.created_at,
        "data": result.data,
    })