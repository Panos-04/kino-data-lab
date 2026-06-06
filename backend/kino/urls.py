from django.urls import path
from .views import window_analysis_list, window_relations_detail

urlpatterns = [
    path("windows/", window_analysis_list, name="window-analysis-list"),
    path("windows/<int:window_id>/relations/", window_relations_detail, name="window-relations-detail"),
]