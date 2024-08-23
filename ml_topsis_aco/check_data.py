# from skmultilearn.dataset import load_dataset

# # Load the dataset with the variant
# data = load_dataset("bibtex", "train")

# # Print the returned data
# print(data)


from skmultilearn.dataset import load_dataset

# Load the dataset with the variant
X_train, y_train, feature_names, label_names = load_dataset("bibtex", "train")

# Proceed with your code
print(f"Feature matrix shape: {X_train.shape}")
print(f"Label matrix shape: {y_train.shape}")
print(f"Feature names (first 10): {feature_names[:10]}")
print(f"Label names (first 10): {label_names[:10]}")
