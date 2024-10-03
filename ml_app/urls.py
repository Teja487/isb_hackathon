"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, index, predict_page,cuda_full,entry_page,image_index,image_predict

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
   # path('', entry_page, name='entry_page'), 
   path('', entry_page, name='home'),
    path('index/', index, name='index'),
    path('image_index/', image_index, name='image_index'),
    path('image_predict/', image_predict, name='image_predict'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('cuda_full/',cuda_full,name='cuda_full'),
]

from django.urls import path
from . import views

# urlpatterns = [
#     path('', views.entry_page, name='entry_page'),  # Route for entry page
#     path('video_upload/', views.index, name='video_upload'),  # Existing video upload
#     # Add route for image upload if needed
#     # path('image_upload/', views.image_upload, name='image_upload'),
#     path('predict/', views.predict_page, name='predict'),
# ]
