from django.shortcuts import render

from polls.analysis_module.handle_user_input import analysis_module


def index(request):
    return render(request, "base.html")


def home(request):
    query = request.POST
    task = query.get("task")
    file = request.FILES.get("user_data")
    if task:
        data, result = analysis_module(query,file)
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

def regressionExplained(request):
    return render(request, "regressionExplained.html")

def classificationExplained(request):
    return render(request, "classificationExplained.html")