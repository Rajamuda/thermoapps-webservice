from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload-image', views.upload_image, name='upload-image'),
    path('process', views.process, name='process')
]