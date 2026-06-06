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