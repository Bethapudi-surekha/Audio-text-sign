# sign/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('listen/', views.listen, name='listen'),
]
