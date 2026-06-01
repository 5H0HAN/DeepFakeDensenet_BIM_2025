import torch
import torch.nn as nn
import torch.optim as optim

def train_heacnn(model, train_loader, val_loader, num_epochs=30, lr=1e-4, device='cuda'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = train_correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")
