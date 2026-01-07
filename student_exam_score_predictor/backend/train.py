import torch
import torch.nn as nn
import torch.optim as optim
from model import StudentModel
from preprocessing import get_data_loaders

def train():
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=16)

    model = StudentModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # Lower LR for regression usually better

    print("Training model...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 5 == 0:
            model.eval()
            val_error = 0
            val_count = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    val_out = model(X_val)
                    val_error += torch.sum(torch.abs(val_out - y_val)).item()
                    val_count += y_val.size(0)
            val_mae = val_error / val_count
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val MAE: {val_mae:.4f}')
            model.train()

    # Evaluate
    model.eval()
    total_error = 0
    count = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            total_error += torch.sum(torch.abs(outputs - y_batch)).item()
            count += y_batch.size(0)
            
    print(f"Training complete. Mean Absolute Error: {total_error / count:.2f}")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
