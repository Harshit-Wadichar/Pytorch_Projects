import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class StudentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    df = pd.read_csv("student_exam_scores.csv")
    
    # Features
    X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']].values
    
    # Target: Pass if exam_score >= 60
    y = (df['exam_score']).values / 100.0
    
    return X, y

def get_data_loaders(batch_size=16):
    X, y = load_data()
    
    # Normalization (Simple min-max roughly based on domain knowledge for consistency with inference)
    # We will use sklearn's scaler for better practice, but for the API consistency 
    # we need to save the scaler or use fixed normalization. 
    # Given the requirements are simple, let's use fixed scaling based on reasonable max values
    # so we can easily replicate it in the frontend/API without loading a scaler file.
    
    # Hours: / 24
    # Sleep: / 24
    # Attendance: / 100
    # Score: / 100
    
    # NOTE: In a real app, we would fit a MinMaxScaler and save it. 
    # Here we hardcode for simplicity and transparency in the code example.
    
    X_norm = X.copy()
    X_norm[:, 0] = X_norm[:, 0] / 24.0 # Study Hours
    X_norm[:, 1] = X_norm[:, 1] / 24.0 # Sleep Hours
    X_norm[:, 2] = X_norm[:, 2] / 100.0 # Attendance
    X_norm[:, 3] = X_norm[:, 3] / 100.0 # Previous Score
    
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
    
    train_dataset = StudentDataset(X_train, y_train)
    test_dataset = StudentDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
