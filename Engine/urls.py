# URL configuration (add this to your urls.py file)
from django.urls import path
from Engine import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
]
