from django.urls import path
from .views import (
    window_analysis_list,
    number_relations_detail,
    general_relations_detail,
    combo_test_api,
    pattern_test_api,
    shape_movements_api,
    board_pattern_events_api,
)
from .views import shape_pattern_test_api
urlpatterns = [
    path("windows/", window_analysis_list, name="window-analysis-list"),
    path(
        "windows/<int:window_id>/relations/<int:number>/",
        number_relations_detail,
        name="number-relations-detail",
    ),
    path(
        "windows/<int:window_id>/general-relations/",
        general_relations_detail,
        name="general-relations-detail",
    ),
    path("combo-test/", combo_test_api, name="combo-test-api"),
    path("pattern-test/", pattern_test_api, name="pattern-test-api"),
    path("shape-pattern-test/", shape_pattern_test_api, name="shape-pattern-test-api"),
    path("shape-movements/", shape_movements_api, name="shape-movements-api"),
    path("board-pattern-events/", board_pattern_events_api, name="board-pattern-events-api"),
]