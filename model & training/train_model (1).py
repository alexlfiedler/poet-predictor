import torch
import torch.nn as nn
from transformers import AutoModel
from model import PoetClassifier

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
                    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
    model.to(device)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for poems, attention_mask, labels in train_loader:
            poems, attention_mask, labels = (poems.to(device), attention_mask.to(device), labels.to(device))

            optimizer.zero_grad()
            outputs = model(poems, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0

        
        with torch.no_grad():
            for poems, attention_mask, labels in val_loader:
                poems, attention_mask, labels = poems.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(poems, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
    return model