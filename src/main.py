import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print('===== STARTING TRAINING AND VALIDATING =====')
# LOAD TRAINING AND VALIDATION DATA
trainDF = pd.read_csv('data/balanced_training_features.csv')
valDF = pd.read_csv('data/extracted_validation_features.csv')

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
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)

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
testDF = pd.read_csv('data/extracted_test_features.csv')
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

report_dict = classification_report(y_test, test_predictions, target_names=['Real', 'AI'], output_dict=True)

report_df = pd.DataFrame(report_dict)

plot_df = report_df.iloc[:-1, :2].T  

plot_df.plot(kind='bar', figsize=(8, 5), colormap='viridis', edgecolor='black')

plt.title('Random Forest Performance Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.xticks(rotation=0, fontsize=12)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('outputs/classification_report_chart.png', bbox_inches='tight')
print("Chart saved successfully as 'classification_report_chart.png'!")
ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, display_labels=['Real (0)', 'AI (1)'], cmap='Blues')

plt.title('Test Dataset Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png', bbox_inches='tight')
print("Confusion Matrix saved successfully!")

ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, display_labels=['Real (0)', 'AI (1)'], cmap='Blues')

plt.title('Test Dataset Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png', bbox_inches='tight')
print("Confusion Matrix saved successfully!")