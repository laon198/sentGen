from django.urls import path, include
from .views import generate_sent

urlpatterns = [
    path("api/", generate_sent),
]

