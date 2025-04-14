from django.shortcuts import render, redirect

from polls.analysis_module.handle_user_input import analysis_module


def index(request):
    return render(request, "base.html")

def home(request):
    print("Hi")
    query = request.POST
    task = query.get("task")
    file = request.FILES.get("user_data")
    if task:
        analysis_module(task, file)
        return redirect(f"suggestMetrics/{task}")
    return render(request, "home.html") #redirect

def classification(request):
    return render(request, "classification.html")

def regression(request):
    return render(request, "regression.html")

def clustering(request):
    return render(request, "clustering.html")
