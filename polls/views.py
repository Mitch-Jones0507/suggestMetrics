from django.shortcuts import render, redirect
import numpy as np
from polls.analysis_module.handle_user_input import analysis_module


def index(request):
    return render(request, "base.html")

def home(request):
    query = request.POST
    task = query.get("task")
    file = request.FILES.get("user_data")
    features = query.getlist("features")
    target = query.get("target")
    if task:
        (data, result) = analysis_module(task, file, features, target)
        print(data)
        if task == 'clustering' or 'regression':
            if len(data[0]) == 3:
                data_min = [min(sublist[0] for sublist in data for item in sublist),min(sublist[1] for sublist in data for item in sublist),min(sublist[2] for sublist in data for item in sublist)]
                data_max = [max(sublist[0] for sublist in data for item in sublist),max(sublist[1] for sublist in data for item in sublist),max(sublist[2] for sublist in data for item in sublist)]
                data = [data,data_min,data_max]
        return render(request, f"{task}.html", {"result": result, "data": data})
    return render(request, "home.html")

def classification(request):
    return render(request, "classification.html")

def regression(request):
    return render(request, "regression.html")

def clustering(request):
    return render(request, "clustering.html")


def about(request):
    return render(request, "about.html")

def metricsExplained(request):
    return render(request, "metricsExplained.html")
