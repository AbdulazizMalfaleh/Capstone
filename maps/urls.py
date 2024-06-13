from django.urls import path
from . import views

urlpatterns = [
    path('members/', views.members, name='members'),
    path('process_form/', views.process_form, name='process_form')
]