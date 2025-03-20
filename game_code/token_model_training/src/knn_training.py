import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        
    def plot_features(self):
        """Plot features after scaling transformation to verify correctness."""
        if self.X is None:
            print("Features not yet extracted. Run extract_features() first.")
            return
        
        label_mapping = {label: idx for idx, label in enumerate(self.df['Label'].unique())}
        self.df['Label_Encoded'] = self.df['Label'].map(label_mapping)
        
        # Define custom color mapping
        color_map = {
            'cyan': 'cyan',
            'green': 'green',
            'magenta': 'magenta',
            'yellow': 'yellow',
            'orange' : 'orange',
            'blue' : 'blue'
            # Add more colors here if needed
        }
        
        colors = self.df['Label'].map(color_map).fillna('gray')
        
                # Use the scaled feature set
        hue_scaled = self.X[:, 0]  # Scaled Hue (multiplied by 3)
        saturation_scaled = self.X[:, 1]  # Scaled Saturation
        value_scaled = self.X[:, 2]  # Scaled Value
                
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111,projection='3d')
        
        # Scatter plot
        # Scatter plot with transformed features
        sc = ax.scatter(hue_scaled, saturation_scaled, value_scaled, c=colors, alpha=0.6)

        # Labels and Titles
        ax.set_xlabel("Scaled Hue")
        ax.set_ylabel("Scaled Saturation")
        ax.set_zlabel("Scaled Value")
        ax.set_title("3D Scatter Plot of Scaled Token Colors in HSV Space")

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
        cbar.set_label('Encoded Labels')

        # Show plot
        plt.show()
        
        

    def extract_features(self):
        """Extract features and labels from the CSV data."""
        self.X = self.df[['Hue', 'Saturation', 'Value', 'Red', 'Green', 'Blue']].values  # Features
        #self.X = self.df[['Hue', 'Saturation', 'Value']].values  # Features
        self.y = self.df['Label'].values  # Labels (color names)
        

    
    def set_scaler(self):
        # Fit the scaler once on the entire dataset
        self.scaler.fit(self.X) # Ensure the same scaler is used across all stages
        #Transform  X immediately after fitting
        self.X = self.scaler.transform(self.X)
        # Increase the importance of Hue by multiplying by 3
        self.X[:, 0] *= 3  # Hue is in column index 0
        print("Features normalized and Hue scaled by a factor of 3")
        

    def shuffle_split_data(self):
        """Shuffle and split the data into training and test sets."""
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        

    def cross_validate(self):
        """Perform cross-validation to find the best k value."""
        k_values = [i for i in range(1, 50)]
        scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, self.X, self.y, cv=10)
            scores.append(np.mean(score))
            print(f"k={k}, all scores from its cross validation: {score}")
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
        self.knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
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
        save_folder = r"game_code\token_model_training\models"
        os.makedirs(save_folder, exist_ok=True)
        
        #Date
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        #Define the full path for the file
        model_path = os.path.join(save_folder, f"{current_date}_knn_model.pkl")  
        scaler_path = os.path.join(save_folder, f"{current_date}_scaler.pkl")        
         
        # Save model       
        with open(model_path, "wb") as file:
            dump(self.knn, file)
        print("Model saved to models folder")
        
        # Save scaler       
        with open(scaler_path, "wb") as file:
            dump(self.scaler, file)
        print("Scaler saved to models folder")
        
        
    
    def run(self):
        """Run the full pipeline."""
        self.extract_features()
        self.set_scaler()
        self.plot_features()
        self.shuffle_split_data()
        best_k = self.cross_validate()
        self.train_knn(4)
        self.test_trained_knn()
        self.save_model()

# Example usage
if __name__ == "__main__":
    csv_path = r'C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\game_code\token_model_training\labelled_data\tokens\CSV_files\Labels_HSV_RGB_20250320_111026.csv'  # Change this to your actual CSV file path
    knn_trainer = KNNTrainer(csv_path)
    knn_trainer.run()
