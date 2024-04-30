
from tqdm import tqdm
import numpy as np
import torch

def evaluate_model(siamese_model, device, test_loader, criterion):
    """
    Evaluate the Siamese network model.

    Args:
    - siamese_model: Siamese network model to be evaluated
    - device: Device to run the evaluation on (e.g., 'cuda' or 'cpu')
    - test_loader: DataLoader for test data
    - criterion: Loss criterion used for evaluation

    Returns:
    - Tuple containing the average loss and accuracy for the test set
    """

    siamese_model.eval()
    correct_predictions = 0
    total_samples = 0
    all_losses = []

    with torch.no_grad():
        for batch_idx, (input_data0, input_data1, target_label) in enumerate(tqdm(test_loader)):
            input_data0, input_data1, target_label = input_data0.to(device), input_data1.to(device), target_label.to(device)
            model_output = siamese_model(input_data0, input_data1)
            loss = criterion(model_output, target_label)
            all_losses.append(loss.item())

            predicted_labels = torch.argmax(model_output, dim=1)
            correct_predictions += torch.sum(predicted_labels == target_label).item()
            total_samples += len(target_label)

    test_loss = np.mean(all_losses)
    test_accuracy = 100. * correct_predictions / total_samples
    print('\nTest set: Average loss = {:.4f}, Test Accuracy = {:.4f}\n'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy
