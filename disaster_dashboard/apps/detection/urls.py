from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('', views.index, name='index'),
    # Original simulation endpoints
    path('api/start-detection/', views.start_detection, name='start_detection'),
    path('api/stop-detection/', views.stop_detection, name='stop_detection'),
    path('api/results/', views.get_detection_results, name='get_results'),
    path('api/settings/', views.update_detection_settings, name='update_settings'),
    
    # New YOLO-powered endpoints
    path('api/process-image/', views.process_image, name='process_image'),
    path('api/live-detection/', views.live_detection, name='live_detection'),
    path('api/model-info/', views.model_info, name='model_info'),
    path('api/switch-model/', views.switch_model, name='switch_model'),
    path('api/detect-from-url/', views.detect_from_url, name='detect_from_url'),
]
