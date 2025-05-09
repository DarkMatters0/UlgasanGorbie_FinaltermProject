# urls.py

from django.urls import path
from . import views

urlpatterns = [
        path('predict/', views.index, name='index'),  # this handles /home/ and /predictGlobalSales/
        path('about/', views.about, name='about'),  # this handles /home/ and /about/
        path('', views.home, name='home'),  # not necessary if already at /home/
]
