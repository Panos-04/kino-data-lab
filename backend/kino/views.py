from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import KinoWindowAnalysis
from .serializers import KinoWindowAnalysisSerializer
from .models import KinoDraw, KinoWindowAnalysis
from .services.window_relations import build_window_relations

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