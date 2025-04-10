from django.urls import path
from . import views

urlpatterns = [
    #path("", views.base, name="base"),
    path("", views.home, name="home"),
    path("/clustering", views.clustering, name="clustering"),
    path("/regression", views.regression, name="regression"),
    path("/classification", views.classification, name="classification"),
]