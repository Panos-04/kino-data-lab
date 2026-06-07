from django.urls import path
from .views import (
    window_analysis_list,
    number_relations_detail,
    general_relations_detail,
    combo_test_api,
)

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
]