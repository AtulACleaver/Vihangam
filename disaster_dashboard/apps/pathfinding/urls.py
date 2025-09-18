from django.urls import path
from . import views

app_name = 'pathfinding'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/calculate-path/', views.calculate_path, name='calculate_path'),
    path('api/validate-path/', views.validate_path, name='validate_path'),
    path('api/replan-path/', views.replan_path, name='replan_path'),
    path('api/optimize-path/', views.optimize_path, name='optimize_path'),
    path('api/export-kml/', views.export_path_kml, name='export_kml'),
    path('api/start-navigation/', views.start_navigation, name='start_navigation'),
    path('api/emergency-return/', views.emergency_return, name='emergency_return'),
    path('api/status/', views.get_navigation_status, name='get_status'),
    path('api/add-waypoint/', views.add_waypoint, name='add_waypoint'),
]
