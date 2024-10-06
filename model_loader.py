from torch import nn, optim

def use_model_classifier(model, input_size, output_size, learning_rate, hidden_units):
    return _use_model_classifier(model, input_size, output_size, learning_rate, [hidden_units, int(hidden_units/4)])

def _use_model_classifier(model, input_size, output_size, learning_rate, hidden_layers):
    # Freeze parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    # Define a new feed-forward classifier with dynamic input size
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_layers[0]),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Dropout to reduce overfitting
        nn.Linear(hidden_layers[0], hidden_layers[1]),  # Hidden layer
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_layers[1], output_size),
        nn.LogSoftmax(dim=1)  # LogSoftmax for classification
    )

    # Replace the pre-trained classifier with the new one
    model.classifier = classifier
    # Define the loss function and the optimizer
    criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss for classification
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return (model, optimizer, criterion)