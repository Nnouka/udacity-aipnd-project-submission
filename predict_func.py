import torch
from torchvision import transforms, models
from PIL import Image
from model_loader import _use_model_classifier
import numpy as np

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
    # Open the image
    img = Image.open(image_path)
    
    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    # Crop out the center 224x224 portion of the image
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert the image to a Numpy array and scale the pixel values to range between 0 and 1
    np_image = np.array(img) / 255.0
    
    # Normalize the image by subtracting the mean and dividing by the standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder the dimensions so that the color channel comes first
    return torch.from_numpy(np_image.transpose((2, 0, 1))).float()

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, weights_only=False)
    
    arch = checkpoint['arch'] if 'arch' in checkpoint else 'vgg16'
    epochs = checkpoint['epochs']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hl = checkpoint['hidden_layers']
    lr =  checkpoint['learning_rate']
    
    print(arch)
    
    model, optimizer, _ = _use_model_classifier(models.__dict__[arch](weights=f"{arch.upper()}_Weights.DEFAULT"), input_size, output_size, lr, hl)
    
    # Load the model's state_dict (learned weights and biases)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Attach the class_to_idx mapping to the model (useful for inference)
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Rebuild the optimizer and load its state_dict (if resuming training)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the model and optimizer
    return model, optimizer

def predict(image_path, model, top_k, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad(): 
        processed_image = process_image(image_path).unsqueeze_(0)  # Add batch dimension
        
        model.to(device)
        processed_image = processed_image.to(device)
        
        # Forward pass: Get the model output (log probabilities)
        output = model(processed_image)
        
        # Convert log probabilities to actual probabilities
        ps = torch.exp(output)
        
        # Get the top K probabilities and corresponding indices
        probs, indices = ps.topk(top_k)
        
        # Detach them to convert to CPU if needed and to work with them as regular Python objects
        probs = probs.cpu().numpy().squeeze()
        indices = indices.cpu().numpy().squeeze()
        
        # Invert class_to_idx dictionary to map indices back to class labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        
        # Get the corresponding class labels
        classes = [idx_to_class[idx] for idx in indices]
        
        return probs, classes
