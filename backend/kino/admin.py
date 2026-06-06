from django.contrib import admin
from .models import KinoDraw, KinoWindowAnalysis, KinoWindowNumber


@admin.register(KinoDraw)
class KinoDrawAdmin(admin.ModelAdmin):
    list_display = ("draw_id", "draw_time", "numbers")
    search_fields = ("draw_id",)
    ordering = ("-draw_time",)


class KinoWindowNumberInline(admin.TabularInline):
    model = KinoWindowNumber
    extra = 0
    readonly_fields = ("number", "count", "percentage")
    can_delete = False


@admin.register(KinoWindowAnalysis)
class KinoWindowAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "window_size",
        "step_size",
        "start_draw",
        "end_draw",
        "start_time",
        "end_time",
    )
    list_filter = ("window_size", "step_size")
    search_fields = ("start_draw__draw_id", "end_draw__draw_id")
    inlines = [KinoWindowNumberInline]


@admin.register(KinoWindowNumber)
class KinoWindowNumberAdmin(admin.ModelAdmin):
    list_display = ("analysis", "number", "count", "percentage")
    list_filter = ("number",)
    search_fields = ("number",)