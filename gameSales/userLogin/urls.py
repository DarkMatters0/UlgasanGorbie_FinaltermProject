from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),  # URL for logout
    path('signup/', views.signup, name='signup'),  # URL for logout
]