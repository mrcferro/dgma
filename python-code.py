file_1 = "data-1.xlsx"
file_2 = "data-2.xlsx"

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
plot_confusion_matrix(cm_1, labels_1, "Confusion Matrix - 1")
plot_confusion_matrix(cm_2, labels_2, "Confusion Matrix - 2")
