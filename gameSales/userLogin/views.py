from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'Login successful!')
            return redirect('index')  # Redirect to the index page or any other page
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('login')
    else:
        return render(request, 'authenticate/login.html')
    
def logout_user(request):
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('home')  # Redirect to the login page or any other page

def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect('signup')

        # Create the user
        user = User.objects.create_user(username=username, password=password1)
        user.save()
        messages.success(request, 'Account created successfully! You can now log in.')
        return redirect('login')

    return render(request, 'authenticate/signup.html')
# Create your views here.
