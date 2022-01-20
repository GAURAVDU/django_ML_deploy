from django.http import HttpResponse
from django.shortcuts import render
import joblib
from sympy import re


def home(request):
    return render(request, "home.html")


def results(request):

    random_forest = joblib.load("random forest.sav")
    decision_tree = joblib.load("decision tree.sav")
    extra_tree = joblib.load("extra tree.sav")
    lis = []

    lis.append(request.GET["RI"])
    lis.append(request.GET["Na"])
    lis.append(request.GET["Mg"])
    lis.append(request.GET["AI"])
    lis.append(request.GET["Si"])
    lis.append(request.GET["K"])
    lis.append(request.GET["Ca"])

    print(lis)

    ans1 = random_forest.predict([lis])
    ans2 = decision_tree.predict([lis])
    ans3 = extra_tree.predict([lis])

    d = {
        "1": "buildingwindowsfloatprocessed",
        "2": "buildingwindowsnonfloatprocessed",
        "3": "vehiclewindowsfloatprocessed",
        "4": "vehiclewindowsnonfloatprocessed (none in this database)",
        "5": "containers",
        "6": "tableware",
        "7": "headlamps",
    }

    return render(
        request, "result.html", {"lis": lis, "ans1": ans1, "ans2": ans2, "ans3": ans3}
    )
