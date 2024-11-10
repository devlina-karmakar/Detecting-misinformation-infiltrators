import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model(model, criterion, optimizer, X_train, y_train, epochs=150):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    accuracy = accuracy_score(y_test, predicted.numpy())
    precision = precision_score(y_test, predicted.numpy())
    recall = recall_score(y_test, predicted.numpy())
    f1 = f1_score(y_test, predicted.numpy())
    conf_matrix = confusion_matrix(y_test, predicted.numpy())

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    num_detected_misinformers = (predicted.numpy() == 1).sum()
    print(f"Number of detected misinformation spreader nodes: {num_detected_misinformers}")
