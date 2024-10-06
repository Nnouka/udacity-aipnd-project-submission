import torch
from torch import nn
from data_loader import load_data
from model_loader import use_model_classifier
import time

#  Save the checkpoint
def save_checkpoint(model, optimizer, class_to_idx, input_size, output_size, hidden_units, epochs, arch, file_path='checkpoint.pth'):
    model.class_to_idx = class_to_idx
    # Create a dictionary for saving the checkpoint
    checkpoint = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'input_size': input_size,
        'output_size': output_size,
        'hidden_layers': [each.out_features for each in model.classifier if isinstance(each, nn.Linear)],
        'learning_rate': optimizer.defaults['lr'],
        'hidden_units': hidden_units
    }

    # Save the checkpoint
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")
    
def train_model(model, data_directory, save_dir, learning_rate, hidden_units, epochs, arch, device):
    model.to(device)
    dataloaders, image_datasets = load_data(data_directory)
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0
        input_size = model.classifier[0].in_features
        output_size = len(image_datasets['train'].classes)
        model, optimizer, criterion = use_model_classifier(model, input_size, output_size, learning_rate, hidden_units)
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        validation_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels.data).item()
        
        epoch_loss = running_loss / len(dataloaders['train'])
        val_loss = validation_loss / len(dataloaders['valid'])
        val_acc = accuracy / len(dataloaders['valid'].dataset)
        
        print(f"Training Loss: {epoch_loss:.4f}.. Validation Loss: {val_loss:.4f}.. Validation Accuracy: {val_acc:.4f}")
    
    # save checkpoint
    save_as = f"{save_dir}/checkpoint_{int(time.time())}.pth"
    
    save_checkpoint(model, optimizer, image_datasets['train'].class_to_idx, input_size, output_size, hidden_units, epochs, arch, save_as)
    
    return model
