import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from pickle import dump, load
from datetime import datetime


class KNNTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.knn = None
        self.scaler = StandardScaler()

    def extract_features(self):
        """Extract features and labels from the CSV data."""
        self.X = self.df[['Hue', 'Saturation', 'Value', 'Red', 'Green', 'Blue']].values  # Features
        self.y = self.df['Label'].values  # Labels (color names)

    def show_csv(self):
        print("Data extracted from CSV:")
        # try:
        #     """Plot the hue against the saturation with each colored dot having the same color as its label using matplotlib."""
        #     plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='hsv', alpha=0.5)
        #     plt.xlabel('Hue')
        #     plt.ylabel('Saturation')
        #     plt.title('Hue vs Saturation')
        #     plt.show()
        # except:
        #     print("Error: Could not plot the data")
        

    def shuffle_split_data(self):
        """Shuffle and split the data into training and test sets."""
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def cross_validate(self):
        """Perform cross-validation to find the best k value."""
        k_values = [i for i in range(1, 50)]
        scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, self.X, self.y, cv=5)
            scores.append(np.mean(score))
            print(f"k={k}, score={np.mean(score)}")
        
        best_index = np.argmax(scores)
        best_k = k_values[best_index]

        # Plot scores against k values as a line graph with dot markers
        plt.plot(k_values, scores, marker='o')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs k')
        plt.show()
        print(f"Best k value: {best_k}")
        return best_k

    def train_knn(self, k):
        """Train the KNN classifier with the best k value."""
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X_train, self.y_train)

    def test_trained_knn(self):
        """Test the trained KNN classifier on the test set."""
        y_pred = self.knn.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        
    def save_model(self):
        """Save the trained KNN model to a file."""
        
        #Define the models folder within token-detection-training
        save_folder = r"Image_Processing_Improvements\token-detection-training\models"
        os.makedirs(save_folder, exist_ok=True)
        
        #Date
        current_date = datetime.now().strftime("%Y-%m-%d")
        #Define the full path for the file
        file_path = os.path.join(save_folder, f"knn_model_{current_date}.pkl")        
         
        # Save model       
        with open(file_path, "wb") as file:
            dump(self.knn, file)
        
        
    
    def run(self):
        """Run the full pipeline."""
        self.extract_features()
        self.show_csv()
        self.shuffle_split_data()
        best_k = self.cross_validate()
        self.train_knn(best_k)
        self.test_trained_knn()
        self.save_model()

# Example usage
if __name__ == "__main__":
    csv_path = r'Image_Processing_Improvements\token-detection-training\labelled_data\CSV_files\Labels_HSV_RGB_20250306_183533.csv'  # Change this to your actual CSV file path
    knn_trainer = KNNTrainer(csv_path)
    knn_trainer.run()
