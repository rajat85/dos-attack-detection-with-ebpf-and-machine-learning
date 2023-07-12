import pickle
import sys
from os.path import isfile
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from sklearn.metrics import confusion_matrix
import ast
import csv

# Check if an argument is present
if len(sys.argv) > 1:
    SECONDS_FOR_FRAME = float(sys.argv[1])
else:
    SECONDS_FOR_FRAME = 1.1

print(f"SECONDS_FOR_FRAME = {SECONDS_FOR_FRAME}")

my_services_dict = None
# Deserialize the dictionary to a file
with open('serialized_dict.pkl', 'rb') as file:
    my_services_dict = pickle.load(file)

# generate an encoding dictionary
ips = list(my_services_dict.keys())
n_ips = len(ips)
# Define the file path
file_path = "data/output/combined.csv"
#
# # Step 1: Load the processed dataset from CSV file
# Deserialization
deserialized_tensors = []
deserialized_labels = []

with open(file_path, "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        deserialized_tensor = ast.literal_eval(row[0])
        deserialized_label = ast.literal_eval(row[1])
        deserialized_tensors.append(np.array(deserialized_tensor))
        deserialized_labels.append(np.array(deserialized_label))

# Print the shape of the loaded data
print(f"Total length of the input data: {len(deserialized_tensors)}")
print(f"Target variable shape: {deserialized_labels[0].shape}")
print(f"Input features shape: {deserialized_tensors[0].shape}")
print(f"Target variable shape: {deserialized_labels[0].shape}")

# Calculate the split index based on the 75:25 ratio
split_index = int(len(deserialized_tensors) * 0.75)

# Divide the list into two parts
train_data = np.array(deserialized_tensors[:split_index])
train_labels = np.array(deserialized_labels[:split_index])
test_data = np.array(deserialized_tensors[split_index:])
test_labels = np.array(deserialized_labels[split_index:])

print(f"train_data: {train_data.shape}")
print(f"train_labels: {train_labels.shape}")
print(f"test_data: {test_data.shape}")
print(f"test_labels: {test_labels.shape}")
#
# # # Convert labels to categorical one-hot encoding


num_classes = 2
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
# # # Step 2: Define the CNN model architecture
model = Sequential()
model.add(Conv2D(n_ips, (3, 3), activation='relu', input_shape=(n_ips, n_ips, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# # # Step 3: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # # Step 4: Train the model
epochs = 31
batch_size = 32
history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()
# Plot training and validation accuracy
train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
plt.figure()
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')
plt.show()
# # # Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
# print('Training Accuracy:', train_accuracy)
# print('Validation Accuracy:', validation_accuracy)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Step 6: Generate the confusion matrix
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(test_labels, axis=1)
cm = confusion_matrix(y_true, y_pred)
# Compute row sums to normalize the confusion matrix
row_sums = np.sum(cm, axis=1, keepdims=True)

# Divide each element of the confusion matrix by the sum of the corresponding row
cm_percentage = cm / row_sums
class_names = ['Normal', 'Malicious']
# Step 7: Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Attack Prediction (Percentage)')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Attacks')
plt.ylabel('True Attacks')

# FP: Predicted as positive (1), but actually negative (0)
fp = cm_percentage[0, 1]
# Get the count of false negatives (FN)
fn = cm_percentage[1, 0]

training_accuracy_with_frame_size_dict = pickle.load(open('training_accuracy_with_frame_size_dict.pkl', 'rb')) if isfile('training_accuracy_with_frame_size_dict.pkl') else {}
training_accuracy_with_frame_size_dict[SECONDS_FOR_FRAME] = {
    "TestingAccuracy": test_accuracy,
    "FP": fp,
    "FN": fn
}
# Serialize the dictionary to a file
with open('training_accuracy_with_frame_size_dict.pkl', 'wb') as file:
    pickle.dump(training_accuracy_with_frame_size_dict, file)

# Add labels to each cell
thresh = cm_percentage.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{cm_percentage[i, j]:.2%}", ha='center', va='center', color='white' if cm_percentage[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
