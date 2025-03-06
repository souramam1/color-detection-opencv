import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the CSV data containing labeled colors
# Assuming the CSV file has columns: 'hue', 'saturation', 'value', 'r', 'g', 'b', 'label'
df = pd.read_csv(r'Image_Processing_Improvements\token-detection-training\labelled_data\CSV_files\Labels_HSV_RGB_20250306_155006.csv')

# Extract features and labels
X = df[['Hue', 'Saturation', 'Value', 'Red', 'Green', 'Blue']].values  # Features
y = df['Label'].values  # Labels (color names)

# #plot the hue against the saturation with each coloured dot having the same colour as its label using matplotlib
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='hsv', alpha=0.5)
# plt.xlabel('Hue')
# plt.ylabel('Saturation')
# plt.title('Hue vs Saturation')
# plt.show()


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# # Normalize the features (optional but helps for KNN)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

# Step 2: Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust n_neighbors
knn.fit(X_train, y_train)

#Make predictions using the test dataset
y_pred = knn.predict(X_test)

#Evaluate the test dataset for accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# # Step 3: Load the new image
# image = cv2.imread('new_image.jpg')  # Load your JPG image
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# #Step 4: Extract the average color of the image
# average_color = np.mean(image_rgb, axis=(0, 1))  # Averaging over all pixels

# # If you want to use HSV, you can convert to HSV first
# image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
# average_color_hsv = np.mean(image_hsv, axis=(0, 1))

# # Step 5: Make the prediction using KNN
# # Prepare the feature vector for prediction (HSV or RGB, depending on what you trained with)
# average_color_scaled = scaler.transform([average_color])  # Scaling the feature

# predicted_label = knn.predict(average_color_scaled)
# print(f"The predicted color label is: {predicted_label[0]}")

# # Display the image
# plt.imshow(image_rgb)
# plt.title(f"Predicted Color: {predicted_label[0]}")
# plt.axis('off')
# plt.show()
