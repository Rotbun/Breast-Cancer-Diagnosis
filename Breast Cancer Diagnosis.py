# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:44:43 2023

@author: Rotbun
"""

# Importing the necessary libraries and modules

from sklearn import svm, linear_model, neighbors, neural_network, naive_bayes, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from numpy import unique
from PIL import Image, ImageTk
from sklearn.mixture import GaussianMixture


    # Read data from local disk
data = pd.read_csv('data1.csv')
    
    # Remove rows with NaN values
    
data = data.dropna()
    
    # Split data into features and target variable
    
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
    
    
    
    ################ Scale the data###################
    
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#########Split data into training and testing sets ############

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Initializing the Models

svm_model = svm.SVC()
logistic_model = linear_model.LogisticRegression()
knn_model = neighbors.KNeighborsClassifier()
ann_model = neural_network.MLPClassifier()
naive_bayes_model = naive_bayes.GaussianNB()
random_forest_model = RandomForestClassifier()


######################## Training the Models #################################

svm_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
ann_model = MLPClassifier(max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)
naive_bayes_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

###################### Evaluating the Models ################################

classifiers = [svm_model, logistic_model, knn_model, ann_model, naive_bayes_model, random_forest_model]

for classifier in classifiers:
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='M')
    precision= precision_score(y_test, y_pred, pos_label='M')
     
    print(f"Classifier: {classifier.__class__.__name__}")
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("\n")

##### Feature Importance Based on the Random Forest Model ####

rf_feature_importance = random_forest_model.feature_importances_

# create a table to display the feature importance
table = pd.DataFrame({'Feature': X.columns, 'Importance': rf_feature_importance})
table = table.sort_values('Importance', ascending=False)
print(table)

rf_feature_importance = random_forest_model.feature_importances_
ranks_and_features = zip(rf_feature_importance,X.columns)
ranks_and_features = sorted(ranks_and_features,reverse=True)
for x, y in ranks_and_features:
    print(x, y)
# Plotting feature importance

keys = [k[1] for k in ranks_and_features ] [::-1]
values = [k[0] for k in ranks_and_features ][::-1]
plt.figure(figsize=(10, 6))
plt.barh(keys, values)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest - Feature Importance')
plt.show()

## Top three classifiers based on their performance

top_models = [('SVM', svm_model),('Logistic Regression', logistic_model), ('Naive Bayes', naive_bayes_model),
                   ('Random Forest',random_forest_model)]

print(top_models)

##### Initialize the Ensemble Model #####

ensemble_model = ensemble.VotingClassifier(estimators=top_models,
                                           voting='hard')

# Training the ensemble model with the top models

ensemble_model.fit(X_train, y_train)

# Using the ensemble model for predictions

ensemble_pred = ensemble_model.predict(X_test)

#Evaluating the Ensemble Model

cm_ensemble = confusion_matrix(y_test, ensemble_pred)
acc_ensemble = accuracy_score(y_test, ensemble_pred)
f1_ensemble = f1_score(y_test, ensemble_pred, pos_label='M')
precision_ensemble= precision_score(y_test, ensemble_pred, pos_label='M')

print("Confusion Matrix:")
print(cm_ensemble)
print("Accuracy:", acc_ensemble)
print("F1 Score:", f1_ensemble)
print("Precision:", precision_ensemble)
print("\n")


#### Create a DataFrame to store the new dataset from Ensemble Predictions

# Convert the string in the Ensemble Prediction to float and save as cluster_list
cluster_list = []
for i in range(len(ensemble_pred)):
    if ensemble_pred[i] == 'M':
        cluster_list.append(0)
    elif ensemble_pred[i] =='B':
        cluster_list.append(1)

    print(cluster_list)

##### Create a new Dataset

new_dataset = pd.DataFrame(X_test, columns=X.columns)

# Add the predicted labels as a new column
new_dataset['predicted_diagnosis'] = cluster_list

# Add noise or perturbations to the predictions (optional)
#new_dataset['predicted_diagnosis'] += np.random.normal(0, 0.1, len(cluster_list))

# Save the new dataset to a CSV file
#new_dataset.to_csv('new_dataset3.csv', index=False)

##### Applying unsupervised clustering on the ensemble predictions

cluster_data = []

cluster_data = pd.read_csv("new_dataset3.csv")

cluster_data.head()

features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

data = cluster_data[features].copy()

data.head()

# define the model
n_components = 2
gaussian_model = GaussianMixture(n_components=n_components)

# train the model
gaussian_model.fit(data)

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(data)

#### Evaluate the Gaussian Mixed Model on the Data #####
# Assess component weights and probabilities
weights = gaussian_model.weights_
probabilities = gaussian_model.predict_proba(data)

# Evaluate the model
bic = gaussian_model.bic(data)
aic = gaussian_model.aic(data)

# Plotting the data and estimated Gaussian components
x = np.linspace(-5, 10, 1000)
plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
for i in range(n_components):
    mean = gaussian_model.means_[i][0]
    std = np.sqrt(gaussian_model.covariances_[i][0][0])
    plt.plot(x, weights[i] * np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2),
             label=f'Component {i+1}')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('GMM Fit to the Data')
plt.show()

# Print the results
print("Component Weights:", weights)
print("Probabilities:")
for i in range(n_components):
    print(f"Component {i+1}: {probabilities[:10, i]}")
print("BIC:", bic)
print("AIC:", aic)

# get all of the unique clusters
gaussian_clusters = unique(gaussian_result)

predicted_diagnosis = cluster_data.predicted_diagnosis

# Create an empty list to store cluster dataframes
clusters = []

# Iterate over each cluster
for i in gaussian_clusters:
    # Filter the cluster data based on predicted diagnosis and features
    cluster = cluster_data[predicted_diagnosis == i][["predicted_diagnosis"] + features]
    clusters.append(cluster)

    # Print cluster information
    count = cluster['predicted_diagnosis'].value_counts()
    percentage = cluster['predicted_diagnosis'].value_counts(normalize=True) * 100
    print(f"Cluster {i}")
    print("Count:\n", count)
    print("Percentage:\n", percentage)
    print("\n")

# Calculate average features for each cluster
average_features = []
for i, cluster in enumerate(clusters):
    average = cluster[features].mean()
    average_features.append(average)
    average_df = pd.DataFrame(average_features, columns=features)
    print (average_df)

# Save the new dataset to a CSV file
#average_df.to_csv('average_df.csv', index=False)

# Create a scatter plot for cluster 0
plt.scatter(range(len(average_df.columns)), average_df.iloc[0], marker='o', label="Cluster 0")

# Create a scatter plot for cluster 1
plt.scatter(range(len(average_df.columns)), average_df.iloc[1], marker='o', label="Cluster 1")

# Add legend, labels, and title to the plot
plt.legend()
plt.xlabel("Features")
plt.ylabel("Average Value")
plt.title("Cluster Plot")

plt.show()

# AI implementation
         
# Save the Clustering Model
joblib.dump(gaussian_model, 'gaussian_clustering.joblib')


# Load the trained clustering model
model = joblib.load('gaussian_clustering.joblib')

# Function to perform cancer diagnosis prediction
def predict_diagnosis(data):
    # Perform clustering on the input data
    cluster_label = model.predict(data)
    
    # Map the cluster label to a diagnosis (Malignant or Benign)
    if cluster_label == 0:
        diagnosis = "Malignant"
    else:
        diagnosis = "Benign"
    
    # Return the predicted diagnosis
    return diagnosis



def predict_button_click():
    # Get input values from entry fields
    input_values = [float(entry.get()) for entry in entry_fields.values()]

    # Create a DataFrame from the input values
    input_data = pd.DataFrame([input_values], columns=entry_fields.keys())

    # Perform prediction using the trained clustering model
    result = predict_diagnosis(input_data)

    # Calculate the percentage of the input data in each cluster
    cluster_counts = [len(cluster_data[cluster_data['predicted_diagnosis'] == i]) for i in gaussian_clusters]
    total_count = sum(cluster_counts)
    percentages = [count / total_count * 100 for count in cluster_counts]

    # Show the predicted diagnosis, percentage, and cluster information in a message box
    messagebox.showinfo("Prediction Result", f"The predicted diagnosis is: {result}\nPercentage: {percentages[0]:.2f}% in Cluster 0, {percentages[1]:.2f}% in Cluster 1")

    # Plotting the graph
    plt.scatter(range(len(average_df.columns)), average_df.iloc[0], marker='o', label="Cluster 0")
    plt.scatter(range(len(average_df.columns)), average_df.iloc[1], marker='o', label="Cluster 1")
    plt.xlabel("Features")
    plt.ylabel("Average Value")
    plt.title("Cluster Plot")
    plt.legend()
    plt.show()


# # Function to handle button click event
# def predict_button_click():
#     # Get input values from entry fields
#     input_values = [float(entry.get()) for entry in entry_fields.values()]

#     # Create a DataFrame from the input values
#     input_data = pd.DataFrame([input_values], columns=entry_fields.keys())

#     # Perform prediction using the trained clustering model
#     result = predict_diagnosis(input_data)

#     # Show the prediction result in a message box
#     messagebox.showinfo("Prediction Result", f"The predicted diagnosis is: {result}")

# Create the application window
window = tk.Tk()
window.title("Cancer Diagnosis Prediction")
window.geometry("1200x1200")
window.resizable(False, False)


# Load the background image
bg_image = Image.open("img_cancer.jpg")
bg_photo = ImageTk.PhotoImage(bg_image)


# Create a canvas to place the background image
canvas = tk.Canvas(window, width=1200, height=1200)
canvas.pack()

# Place the background image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)

# Create a frame for the input fields
input_frame = tk.Frame(window, bg='white')
input_frame.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

# Define the default values for each input field
default_values = {
    'radius_mean': 0.0,
    'texture_mean': 0.0,
    'perimeter_mean': 0.0,
    'area_mean': 0.0,
    'smoothness_mean': 0.0,
    # ... Add default values for other fields
}

# Create the input labels and entry fields
labels = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst']
num_labels = len(labels)
num_columns = 3
rows_per_column = num_labels // num_columns + 1

# Create a dictionary to hold the entry fields
entry_fields = {}

for i, label_text in enumerate(labels):
    label = tk.Label(input_frame, text=label_text + ":", font=("Arial", 10))
    label.grid(column=i % num_columns * 2, row=i // num_columns, sticky="e", pady=20)
    
    # Set the default value for the entry field
    default_value = default_values.get(label_text, 10)
    entry = tk.Entry(input_frame, font=("Arial", 10))
    entry.insert(tk.END, default_value) 
    entry.grid(column=i % num_columns * 2 + 1, row=i // num_columns, pady=20)
    
    # Store the entry field in the dictionary
    entry_fields[label_text] = entry

# Create the predict button
predict_button = tk.Button(window, text="Predict", command=predict_button_click,  bg="#4caf50", fg="white", font=("Arial", 12), padx=10, pady=10)
predict_button.place(relx=0.5, rely=0.54, anchor=tk.CENTER)


# Run the application
window.mainloop()