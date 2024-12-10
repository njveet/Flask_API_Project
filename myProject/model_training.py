import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

file_path = "myProject/Student data.csv"  # Replace with your file path
student_data = pd.read_csv(file_path)

# Rename columns
column_names = [
    'First_Term_GPA', 'Second_Term_GPA', 'First_Language', 'Funding', 'School',
    'Fast_Track', 'Coop', 'Residency', 'Gender', 'Previous_Education',
    'Age_Group', 'High_School_Average_Mark', 'Math_Score', 'English_Grade', 'Target_Variable'
]
student_data.columns = column_names[:len(student_data.columns)]

# Drop metadata rows
student_data = student_data.iloc[5:].reset_index(drop=True)

# Convert numeric columns to proper types
numerical_columns = ['First_Term_GPA', 'Second_Term_GPA', 'High_School_Average_Mark', 'Math_Score']
for col in numerical_columns + ['Target_Variable']:
    student_data[col] = pd.to_numeric(student_data[col], errors='coerce')

# Impute missing values for numerical columns
for col in numerical_columns:
    student_data[col].fillna(student_data[col].mean(), inplace=True)

# Impute missing values for categorical columns
categorical_columns = ['First_Language', 'Previous_Education', 'Age_Group', 'English_Grade']
for col in categorical_columns:
    student_data[col].fillna(student_data[col].mode()[0], inplace=True)


# Drop rows with missing critical values
student_data.dropna(subset=['Funding', 'School', 'Fast_Track', 'Target_Variable'], inplace=True)
student_data.reset_index(drop=True, inplace=True)


# Scale numerical features excluding Target_Variable
scaler = StandardScaler()
student_data[numerical_columns] = scaler.fit_transform(student_data[numerical_columns])


# Encode categorical variables
encoder = LabelEncoder()
for col in categorical_columns + ['Funding', 'School', 'Fast_Track', 'Coop', 'Residency', 'Gender']:
    student_data[col] = encoder.fit_transform(student_data[col])

# Separate features and target
X = student_data.drop(columns=['Target_Variable'])
y = student_data['Target_Variable']

# Verify target variable
print("Target variable unique values:", y.unique())
print("Class distribution in the target variable:")
print(y.value_counts())

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure all class labels are represented in class_weights_dict and are consecutive integers
unique_classes = np.unique(np.concatenate([y_train, y_test]))
class_mapping = {label: idx for idx, label in enumerate(unique_classes)}

# Map original labels to consecutive integer indices
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)



# Reset indices after mapping
y_train_mapped = y_train.map(class_mapping).reset_index(drop=True)
y_test_mapped = y_test.map(class_mapping).reset_index(drop=True)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)



# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_mapped), y=y_train_mapped)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights dictionary:", class_weights_dict)

# Build and compile the Neural Network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train_mapped.shape)
print("y_test shape:", y_test_mapped.shape)

history = model.fit(
    X_train, y_train_mapped,
    validation_data=(X_test, y_test_mapped),
    epochs=5,
    batch_size=16,  # Reduced batch size
    class_weight=class_weights_dict,
    verbose=1
)



import time
from tensorflow.keras.callbacks import LambdaCallback

# Callback to measure epoch time
epoch_time_callback = LambdaCallback(
    on_epoch_begin=lambda epoch, logs: print(f"Starting epoch {epoch+1} at {time.time()}"),
    on_epoch_end=lambda epoch, logs: print(f"Finished epoch {epoch+1} at {time.time()}")
)

history = model.fit(
    X_train, y_train_mapped,
    validation_data=(X_test, y_test_mapped),
    epochs=5,
    batch_size=16,
    class_weight=class_weights_dict,
    verbose=1,
    callbacks=[epoch_time_callback]
)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


from sklearn.metrics import classification_report

# Generate predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate classification report
print(classification_report(y_test_mapped, y_pred))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(y_test_mapped, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

model.save("student_success_model.h5")
print("Model saved as student_success_model.h5")
