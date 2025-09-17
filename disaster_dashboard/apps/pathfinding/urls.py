from django.urls import path
from . import views

app_name = 'pathfinding'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/calculate-path/', views.calculate_path, name='calculate_path'),
    path('api/start-navigation/', views.start_navigation, name='start_navigation'),
    path('api/emergency-return/', views.emergency_return, name='emergency_return'),
    path('api/status/', views.get_navigation_status, name='get_status'),
    path('api/add-waypoint/', views.add_waypoint, name='add_waypoint'),
]