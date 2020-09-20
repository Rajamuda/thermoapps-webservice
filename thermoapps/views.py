# from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed

# Create your views here.
def index(request):
    return HttpResponse(loader.get_template('index.html').render())