from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/emergency-alert/', views.emergency_alert, name='emergency_alert'),
    path('api/refresh-data/', views.refresh_data, name='refresh_data'),
]