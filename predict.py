import argparse
import torch
from torchvision import transforms
from PIL import Image
from predict_func import load_checkpoint, predict
from data_loader import cat_to_name

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Check if GPU is available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load model
    model, _ = load_checkpoint(args.checkpoint)

    # Predict
    probs, classes = predict(args.image_path, model, args.top_k, device)

    cat_names = cat_to_name(args.category_names)
    
    mPI = probs.argmax() # find the max probability index
    mCat = classes[mPI] # get the category with max probability
    prob = probs[mPI] # get  max probability
    name = cat_names[mCat] # get the name predicted
    
    print(f"Probabilities: {probs}")
    print(f"Classes: {classes}")
    
    print(f"Prediction: {name}")
    print(f"Probability: {prob}")

if __name__ == '__main__':
    main()
