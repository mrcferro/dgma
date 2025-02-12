import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def process_file(file_path):
    """
    Reads an Excel file, extracts predictions and actual labels,
    computes the confusion matrix, and calculates accuracy.
    """
    df = pd.read_excel(file_path)
    
    # Identify prediction columns (all except 'Score')
    prediction_columns = [col for col in df.columns if col != "Score"]
    
    # Create combined lists of actual and predicted values
    y_true = np.repeat(df["Score"].values, len(prediction_columns))
    y_pred = df[prediction_columns].values.flatten()
    
    # Remove NaN or undefined values
    valid_indices = ~pd.isna(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    
    # Compute confusion matrix
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Compute accuracy
    accuracy = np.mean(y_true == y_pred)
    
    return cm, accuracy, labels

def plot_confusion_matrix(cm, labels, title):
    """Plots the confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# File paths (replace with actual paths if running locally)
file_1 = "dados-1.xlsx"
file_2 = "dados-2.xlsx"

# Process files
cm_1, acc_1, labels_1 = process_file(file_1)
cm_2, acc_2, labels_2 = process_file(file_2)

# Print results
print("Results for 'dados-1.xlsx':")
print("Confusion Matrix:")
print(cm_1)
print(f"Accuracy: {acc_1:.2%}\n")

print("Results for 'dados-2.xlsx':")
print("Confusion Matrix:")
print(cm_2)
print(f"Accuracy: {acc_2:.2%}\n")

# Plot confusion matrices
plot_confusion_matrix(cm_1, labels_1, "Confusion Matrix - 1 ChatGPT")
plot_confusion_matrix(cm_2, labels_2, "Confusion Matrix - 2 ChatGPT")
