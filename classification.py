import os
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# Step 1: Collect and preprocess data
def load_data(dataset_path, extensions=(".mp3", ".wav", ".flac")):
    data = []
    labels = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(extensions):
                continue
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path)
            data.append(y)
            labels.append(folder)
    return data, labels


# Step 2: Extract features
def extract_features(data):
    features = []
    for y in data:
        mfcc = librosa.feature.mfcc(y=y, n_mfcc=20)
        features.append(np.mean(mfcc.T, axis=0))
    return features


# Step 3: Train a machine learning model
def train_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


# Step 4: Evaluate and optimize the model
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate and return the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, y_pred


def plot_confusion_matrix(cm, labels):
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels, fmt=".2f")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


# Main function
def main():
    dataset_path = "musan/music"
    data, labels = load_data(dataset_path)
    features = extract_features(data)

    X = np.array(features)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and evaluate its accuracy
    model = train_model(X_train, y_train)
    accuracy, cm, y_pred = evaluate_model(model, X_test, y_test)

    # Print the accuracy of the model
    print(f"Model accuracy: {accuracy}")

    # Plot confusion matrix
    plot_confusion_matrix(cm, le.classes_)

    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()