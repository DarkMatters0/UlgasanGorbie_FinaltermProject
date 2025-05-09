from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings
from .dataset import load_or_generate_stats

import pickle
import numpy as np

# Load the trained model
with open('ml_model/game_sales_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders (for categorical variables)
with open('ml_model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load frequency encoders (for Developer and Publisher)
with open('ml_model/freq_encoders.pkl', 'rb') as f:
    freq_encoders = pickle.load(f)

@login_required(login_url='login')
def index(request):
    result = None

    if request.method == 'POST':
        # Retrieve form data
        platform = request.POST.get('Platform')
        genre = request.POST.get('Genre')
        rating = request.POST.get('Rating')
        developer = request.POST.get('Developer')
        publisher = request.POST.get('Publisher')
        year = int(request.POST.get('Year_of_Release'))
        critic_score = float(request.POST.get('Critic_Score'))
        critic_count = int(request.POST.get('Critic_Count'))
        user_score = float(request.POST.get('User_Score'))
        user_count = int(request.POST.get('User_Count'))

        # Label encoding
        def label_encode(col, value):
            le = label_encoders.get(col)
            return le.transform([value])[0] if le and value in le.classes_ else 0

        encoded_platform = label_encode('Platform', platform)
        encoded_genre = label_encode('Genre', genre)
        encoded_rating = label_encode('Rating', rating)
        encoded_year = label_encode('Year_of_Release', year)

        # Frequency encoding
        def freq_encode(enc_map, value):
            return enc_map.get(value, 0.0)

        encoded_developer = freq_encode(freq_encoders.get('Developer', {}), developer)
        encoded_publisher = freq_encode(freq_encoders.get('Publisher', {}), publisher)

        # Prepare input for model
        input_data = np.array([[ 
            encoded_platform,
            encoded_genre,
            encoded_rating,
            encoded_developer,
            encoded_publisher,
            encoded_year,
            critic_score,
            critic_count,
            user_score,
            user_count
        ]])

        # Predict and reverse log1p
        predicted_log_sales = model.predict(input_data)[0]
        predicted_sales = np.expm1(predicted_log_sales)
        result = f"Predicted Global Sales: {predicted_sales:.2f} million units"

    return render(request, 'index.html', {'result': result})

def home(request):
    return render(request, 'home.html')

@login_required(login_url='login')
def about(request):
    csv_path = os.path.join(settings.BASE_DIR, '', 'Video_Games_Sales_as_at_22_Dec_2016.csv')
    stats = load_or_generate_stats(csv_path)
    return render(request, 'about_dataset.html', {'stats': stats})
