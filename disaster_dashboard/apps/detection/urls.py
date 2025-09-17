from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/start-detection/', views.start_detection, name='start_detection'),
    path('api/stop-detection/', views.stop_detection, name='stop_detection'),
    path('api/results/', views.get_detection_results, name='get_results'),
    path('api/settings/', views.update_detection_settings, name='update_settings'),
]