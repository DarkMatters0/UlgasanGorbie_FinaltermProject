import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import base64
import os
import pickle

def get_dataset_statistics(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    data_copy = df.copy()
    required_columns = ['Name', 'Platform', 'Year_of_Release', 'Publisher', 'Rating',
                        'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer']
    
    data_copy.dropna(subset=required_columns, inplace=True)
    data_copy = data_copy.drop(columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], errors='ignore')
    data_copy = data_copy.dropna(subset=['Global_Sales'])
    data_copy['Year_of_Release'] = data_copy['Year_of_Release'].astype(int)
    sales_cap = data_copy['Global_Sales'].quantile(0.99)
    data_copy = data_copy[data_copy['Global_Sales'] <= sales_cap]


    total_records = len(data_copy)
    total_features = data_copy.shape[1]
    top_platforms = data_copy['Platform'].value_counts().head(5).to_dict()
    top_genres = data_copy['Genre'].value_counts().head(5).to_dict()
    top_publishers = data_copy['Publisher'].value_counts().head(5).to_dict()
    rating_distribution = data_copy['Rating'].value_counts().to_dict()
    global_sales = data_copy['Global_Sales'].tolist()

    model_data = data_copy.copy()

    # Frequency encoding for Developer and Publisher
    for col in ['Developer', 'Publisher']:
        freq = model_data[col].value_counts(normalize=True)
        model_data[col] = model_data[col].map(freq)

    # Label encode other categorical features
    label_encode_cols = ['Platform', 'Genre', 'Rating', 'Year_of_Release']
    for col in label_encode_cols:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col])

    X = model_data.drop(columns=['Global_Sales', 'Name'])
    y = np.log1p(model_data['Global_Sales'])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    mse_list = []
    r2_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse_list.append(mean_squared_error(y_test, predictions))
        r2_list.append(r2_score(y_test, predictions))

    mse = np.mean(mse_list)
    r2 = np.mean(r2_list)
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kf, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, label='Training Score', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Validation Score', marker='o')
    plt.title('Learning Curve (Random Forest)')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    numerical_features = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
    summary_stats = data_copy[numerical_features].describe().to_dict()

    return {
        "total_records": total_records,
        "total_features": total_features,
        "summary_stats": summary_stats,
        "top_platforms": top_platforms,
        "top_genres": top_genres,
        "top_publishers": top_publishers,
        "rating_distribution": rating_distribution,
        "global_sales": global_sales,
        "model_mse": round(mse, 4),
        "model_r2": round(r2, 4),
        "learning_curve": image_base64,
        "cv_results": {
            "folds": [
                {"fold": i + 1, "mse": round(mse_list[i], 4), "r2": round(r2_list[i], 4)}
                for i in range(len(mse_list))
            ],
            "mean_mse": round(mean_mse, 4),
            "std_mse": round(std_mse, 4),
            "mean_r2": round(mean_r2, 4),
            "std_r2": round(std_r2, 4)
        }
    }

def load_or_generate_stats(csv_path, pickle_path='ml_model/dataset_stats.pkl'):
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load pickle. Recomputing. Reason: {e}")

    stats = get_dataset_statistics(csv_path)
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump(stats, f)
    return stats
