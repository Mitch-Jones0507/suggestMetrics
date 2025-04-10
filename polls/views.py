from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

from polls.analysis_module.handle_user_input import analysis_module

def index(request):
    return render(request, "base.html")

def home(request):
    print(request.GET)
    query = request.GET
    task = query.get("task")
    file = query.get("user_data")
    analysis_module(task, file)
    return render(request, "home.html")

def classification(request):
    return render(request, "classification.html")

def regression(request):
    return render(request, "regression.html")

def clustering(request):
    return render(request, "clustering.html")
