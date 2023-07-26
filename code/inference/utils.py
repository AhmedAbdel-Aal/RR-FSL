import numpy as np
import torch
from sklearn.metrics import classification_report



def load_npy(file_path):
    """
    Load a .npy file and return its content as a NumPy array.

    Parameters:
        file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray: The content of the .npy file as a NumPy array.
    """
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except:
        print(f"An error occurred while loading '{file_path}'.")
        return None

    

def numpy_to_tensor(numpy_array):
    """
    Convert a NumPy array to a PyTorch tensor.

    Parameters:
        numpy_array (numpy.ndarray): The NumPy array to be converted.

    Returns:
        torch.Tensor: The PyTorch tensor converted from the NumPy array.
    """
    try:
        tensor = torch.from_numpy(numpy_array)
        return tensor
    except:
        print("An error occurred while converting the NumPy array to a PyTorch tensor.")
        return None
    
    

def print_classification_report(y_true, y_pred):
    """
    Compute and print the classification report for two label tensors.

    Parameters:
        y_true (torch.Tensor): The true label tensor of shape (1, 1338).
        y_pred (torch.Tensor): The predicted label tensor of shape (1, 1338).

    Returns:
        None
    """
    # Convert tensors to NumPy arrays
    y_true_np = y_true.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    
    target_names = [
    "Fact",
    "Argument",
    "Precedent",
    "Ratio",
    "RulingL",
    "RulingP",
    "Statute",
    ]

    # Compute classification report
    report = classification_report(y_true_np, y_pred_np, target_names=target_names, zero_division=0)

    # Print the classification report
    print(report)

from sklearn.metrics import f1_score

def get_f1_score(y_true, y_pred):
  predictions = y_pred.squeeze(0).cpu().numpy()
  true_labels = y_true.squeeze(0).cpu().numpy()
  f1 = f1_score(true_labels, predictions, average='weighted')
  return f1  
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(ground_truth, predicted_labels):
    """
    Plot a confusion matrix based on ground truth and predicted labels.

    Args:
        ground_truth (numpy.ndarray): Ground truth labels.
        predicted_labels (numpy.ndarray): Predicted labels.
        classes (list): List of class labels.

    Returns:
        None (displays the plot)
    """
    classes = [
    "Fact",
    "Argument",
    "Precedent",
    "Ratio",
    "RulingL",
    "RulingP",
    "Statute",
    ]
    # Flatten the ground truth and predicted labels
    ground_truth = ground_truth.flatten()
    predicted_labels = predicted_labels.flatten()

    # Compute the confusion matrix
    cm = confusion_matrix(ground_truth, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Fill the confusion matrix cells with values
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.show()
