from django.urls import path

from api.views import detect_landmark

urlpatterns = [
    path('detect/face/landmark/', detect_landmark),
]