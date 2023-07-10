import subprocess
import pickle
from os.path import isfile

# Execute Python file with argument
file1 = "step_1_process_input_data.py"
file2 = "step_2_process_input_data.py"
file3 = "step_3_combine_two_csvs.py"
file4 = "step_4_model_with_cnn.py"

value = str(1.1)

subprocess.run(["python", file1, value])
subprocess.run(["python", file2, value])
subprocess.run(["python", file3])
subprocess.run(["python", file4, value])


training_accuracy_with_frame_size_dict = pickle.load(open('training_accuracy_with_frame_size_dict.pkl', 'rb')) if isfile('training_accuracy_with_frame_size_dict.pkl') else None
print(training_accuracy_with_frame_size_dict)

