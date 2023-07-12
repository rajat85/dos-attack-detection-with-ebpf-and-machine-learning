import pickle
from os.path import isfile

training_accuracy_with_frame_size_dict = pickle.load(open('training_accuracy_with_frame_size_dict.pkl', 'rb')) if isfile('training_accuracy_with_frame_size_dict.pkl') else None
print(training_accuracy_with_frame_size_dict)

for value in range(5, 16, 1):
    current_value = value / 10
    current_value = "{:0.1f}".format(current_value)

    run_1 = training_accuracy_with_frame_size_dict[f'{current_value}_1']
    run_2 = training_accuracy_with_frame_size_dict[f'{current_value}_2']
    run_3 = training_accuracy_with_frame_size_dict[f'{current_value}_3']

    training_accuracy_with_frame_size_dict[f'{current_value}'] = {
        "TestingAccuracy": (run_1['TestingAccuracy'] + run_2['TestingAccuracy'] + run_3['TestingAccuracy']) / 3.0,
        "FP": (run_1['FP'] + run_2['FP'] + run_3['FP']) / 3.0,
        "FN": (run_1['FN'] + run_2['FN'] + run_3['FN']) / 3.0
    }

    del training_accuracy_with_frame_size_dict[f'{current_value}_1']
    del training_accuracy_with_frame_size_dict[f'{current_value}_2']
    del training_accuracy_with_frame_size_dict[f'{current_value}_3']


print(training_accuracy_with_frame_size_dict)


# training_accuracy_with_frame_size_dict = pickle.load(open('training_accuracy_with_frame_size_dict.pkl', 'rb')) if isfile('training_accuracy_with_frame_size_dict.pkl') else None
# del training_accuracy_with_frame_size_dict[f'{1.1}_1']
# del training_accuracy_with_frame_size_dict[f'{1.1}_2']
# del training_accuracy_with_frame_size_dict[f'{1.1}_3']
# # Serialize the dictionary to a file
# with open('training_accuracy_with_frame_size_dict.pkl', 'wb') as file:
#     pickle.dump(training_accuracy_with_frame_size_dict, file)

