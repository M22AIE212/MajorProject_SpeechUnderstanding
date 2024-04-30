
import numpy as np
import torch
from tqdm import tqdm

def train_model(siamese_model, device, train_loader, current_epoch, optimizer, criterion):
    """
    Train the Siamese network model.

    Args:
    - siamese_model: Siamese network model to be trained
    - device: Device to run the training on (e.g., 'cuda' or 'cpu')
    - train_loader: DataLoader for training data
    - current_epoch: Current epoch number
    - optimizer: Optimizer used for training
    - criterion: Loss criterion used for training

    Returns:
    - Tuple containing the average loss and accuracy for the training set
    """

    siamese_model.train()
    all_losses = []
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (input_data0, input_data1, target_label) in enumerate(tqdm(train_loader)):
        input_data0, input_data1, target_label = input_data0.to(device), input_data1.to(device), target_label.to(device)
        optimizer.zero_grad()

        model_output = siamese_model(input_data0, input_data1)
        loss = criterion(model_output, target_label)
        all_losses.append(loss.item())
        loss.backward()

        optimizer.step()

        predicted_labels = torch.argmax(model_output, dim=1)
        correct_predictions += torch.sum(predicted_labels == target_label).item()
        total_samples += len(target_label)

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.6f}'.format(
                current_epoch, (batch_idx+1) * len(input_data0), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item(),
                (100. * correct_predictions / total_samples)))

    average_loss = np.mean(all_losses)
    train_accuracy = 100. * correct_predictions / total_samples
    print('\nTrain set: Average loss = {:.4f}, Train Accuracy = {:.4f}\n'.format(average_loss, train_accuracy))
    return average_loss, train_accuracy
