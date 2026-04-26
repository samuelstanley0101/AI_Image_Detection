import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print('===== STARTING TRAINING AND VALIDATING =====')
# LOAD TRAINING AND VALIDATION DATA
trainDF = pd.read_csv('balanced_training_features.csv')
valDF = pd.read_csv('extracted_validation_features.csv')

# CHECK FOR FAILED ROWS AND DROP THEM IF EMPTY
trainDF = trainDF.dropna()
valDF = valDF.dropna()

# SEPERATE FEATURES
featureColumns = [
    'average_brightness', 
    'average_contrast', 
    'average_noise', 
    'noise_deviation',
    'sharpness',
    'edge_density',
    'high_frequency'
]


# SPLIT TRAINING DATA
X_train = trainDF[featureColumns]
y_train = trainDF['labelA']

# SPLIT VALIDATION DATA
X_val = valDF[featureColumns]
y_val = valDF['labelA']

# INITALIZE MODEL
model = LogisticRegression(max_iter=200)

# TRAIN THE MODEL
print("Training the model...")
model.fit(X_train, y_train) 
print("Training complete!")

# VALIDATE THE MODEL
print("\nPredicting on validation data...")
predictions = model.predict(X_val)

# COMPARE MODEL'S GUESSES
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# PRINT REPORT (Precision, Recall, F1-Score)
print("Detailed Classification Report:")
print(classification_report(y_val, predictions, target_names=['Real (0)', 'AI (1)']))

print('===== FINISHED TRAINING AND VALIDATING STARTING TESTING=====')

# LOAD THE TEST DATA SET AND DROP EMPTY ROWS
testDF = pd.read_csv('extracted_test_features.csv')
testDF = testDF.dropna()

# SEPERATE FEATURE COLUMNS
X_test = testDF[featureColumns]
y_test = testDF['labelA']


# PREDICTING DATA
print("Predicting on unseen test data...")
test_predictions = model.predict(X_test)

# PRINT FINAL REPORT
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n")

print("Final Test Classification Report:")
print(classification_report(y_test, test_predictions, target_names=['Real (0)', 'AI (1)']))
print('===== FINISHED TESTING =====')