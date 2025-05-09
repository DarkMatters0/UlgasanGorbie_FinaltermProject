import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load your dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')  # adjust if needed

# Copy for processing
data_copy = data.copy()

# Data Cleaning
required_columns = ['Name', 'Platform', 'Year_of_Release', 'Publisher', 'Rating',
                    'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer']
data_copy.dropna(subset=required_columns, inplace=True)

# Drop columns that would cause leakage
data_copy.drop(columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], inplace=True)

# Drop rows with missing target
data_copy.dropna(subset=['Global_Sales'], inplace=True)

# Convert year to int
data_copy['Year_of_Release'] = data_copy['Year_of_Release'].astype(int)

# Cap outliers
sales_cap = data_copy['Global_Sales'].quantile(0.99)
data_copy = data_copy[data_copy['Global_Sales'] <= sales_cap]

# --- Encoding ---
# Frequency encoding for Developer and Publisher
freq_encoders = {}
for col in ['Developer', 'Publisher']:
    freq_map = data_copy[col].value_counts(normalize=True).to_dict()
    data_copy[col] = data_copy[col].map(freq_map)
    freq_encoders[col] = freq_map

# Label encoding for others
label_encoders = {}
label_columns = ['Platform', 'Genre', 'Rating', 'Year_of_Release']
for col in label_columns:
    le = LabelEncoder()
    data_copy[col] = le.fit_transform(data_copy[col])
    label_encoders[col] = le

# --- Train/Test Split ---
X = data_copy.drop(columns=['Global_Sales', 'Name'])
y = np.log1p(data_copy['Global_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Save Artifacts ---
os.makedirs('ml_model', exist_ok=True)

with open('ml_model/game_sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('ml_model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('ml_model/freq_encoders.pkl', 'wb') as f:
    pickle.dump(freq_encoders, f)

# --- Evaluate ---
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Random Forest Performance:\n - MSE: {mse:.2f}\n - R² Score: {r2:.2f}")
