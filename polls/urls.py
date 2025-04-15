from django.urls import path

from polls import views

urlpatterns = [
    #path("", views.base, name="base"),
    path("", views.home, name="home"),
    path("/clustering", views.clustering, name="clustering"),
    path("/regression", views.regression, name="regression"),
    path("/classification", views.classification, name="classification"),
    path("/about", views.about, name="about"),
    path("/metricsExplained", views.metricsExplained, name="metricsExplained")
]