from django.urls import path

from .views import views
from .views import object_detection

urlpatterns = [
    path('object_detection/predict', object_detection.predict, name='predict'),
    path('', views.index, name='index'),
]