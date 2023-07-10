import pickle
import sys
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import itertools
import csv

pd.options.mode.chained_assignment = None  # default='warn'

FILE_NAME = 'valid_few_requests_through_lua_for_tcp_states_07_07.csv'
LABEL = '0'
# Check if an argument is present
if len(sys.argv) > 1:
    SECONDS_FOR_FRAME = float(sys.argv[1])
else:
    SECONDS_FOR_FRAME = 1.1

print(f"SECONDS_FOR_FRAME = {SECONDS_FOR_FRAME}")

tcp_traces = pd.read_csv(f"data/input/{FILE_NAME}",
                         encoding='latin-1', sep=',', keep_default_na=False)
pattern = r'^172\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
tcp_traces = tcp_traces[tcp_traces['LADDR'].str.match(pattern)]
tcp_traces = tcp_traces[tcp_traces['RADDR'].str.match(pattern)]

tcp_traces['TIME'] = pd.to_datetime(tcp_traces['TIME'], format='%H:%M:%S')
tcp_traces.set_index('TIME', inplace=True)

services = tcp_traces['C-COMM'].unique()
n_services = len(services)

source_ips = tcp_traces['LADDR'].unique()
n_source_ips = len(source_ips)
print(f'Number of source unique IP addresses are: {n_source_ips}')

destination_ips = tcp_traces['RADDR'].unique()
n_destination_ips = len(destination_ips)
print(f'Number of destination unique IP addresses are: {n_destination_ips}')

tcp_states = set(tcp_traces['OLDSTATE']) | set(tcp_traces['NEWSTATE'])
n_tcp_states = len(tcp_states)
print(f'Number of unique TCP states are: {n_tcp_states}')

tcp_states_dict = dict(zip(tcp_states, range(1, n_tcp_states + 1)))

"""
START of This code is required only for calculating the corrct size of the matrix
"""
OTHER_FILE_NAME = 'invalid_few_requests_through_lua_for_tcp_states_07_07.csv'

other_tcp_traces = pd.read_csv(f"data/input/{OTHER_FILE_NAME}",
                               encoding='latin-1', sep=',', keep_default_na=False)
other_tcp_traces = other_tcp_traces[other_tcp_traces['LADDR'].str.match(pattern)]
other_tcp_traces = other_tcp_traces[other_tcp_traces['RADDR'].str.match(pattern)]
other_source_ips = other_tcp_traces['LADDR'].unique()
other_n_source_ips = len(other_source_ips)
other_destination_ips = other_tcp_traces['RADDR'].unique()
other_n_destination_ips = len(other_destination_ips)
"""
END of This code is required only for calculating the corrct size of the matrix
"""

# generate an encoding dictionary
ips = sorted(set(source_ips) | set(destination_ips) | set(other_source_ips) | set(other_destination_ips))
r_services = range(1, len(ips) + 1)
my_services_dict = dict(zip(ips, r_services))

# Serialize the dictionary to a file
with open('serialized_dict.pkl', 'wb') as file:
    pickle.dump(my_services_dict, file)

print(my_services_dict)


# Function to recursively convert lists and arrays to tuples
def convert_to_tuples(obj):
    if isinstance(obj, list):
        return tuple(convert_to_tuples(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    return obj


tcp_traces['LADDR_NUM'] = tcp_traces['LADDR'].map(my_services_dict)
tcp_traces['RADDR_NUM'] = tcp_traces['RADDR'].map(my_services_dict)
tcp_traces['OLDSTATE_NUM'] = tcp_traces['OLDSTATE'].map(tcp_states_dict)
tcp_traces['NEWSTATE_NUM'] = tcp_traces['NEWSTATE'].map(tcp_states_dict)

my_tcp_traces_df = tcp_traces[
    ['C-PID', 'SKADDR', 'LADDR', 'LADDR_NUM', 'LPORT', 'RADDR', 'RADDR_NUM', 'RPORT', 'OLDSTATE', 'NEWSTATE',
     'OLDSTATE_NUM', 'NEWSTATE_NUM']]

my_tcp_traces_df.to_csv("filtered_data.csv", sep=',', encoding='utf-8', index=False)

# Create a dictionary to store the mappings
combination_dict = {}
# Assign integer values to combinations
assigned_value = 1


def get_a_unique_number(obj):
    if isinstance(obj, list):
        global assigned_value
        _tuple = convert_to_tuples(obj)
        if _tuple in combination_dict:
            return combination_dict[_tuple]
        else:
            combination_dict[_tuple] = assigned_value
            assigned_value += 1
            return combination_dict[_tuple]
    else:
        return 0


def create_adjacency_matrix_with_meta_information(_my_tcp_traces_df):
    df = _my_tcp_traces_df
    services_list = list(r_services)
    adjacency_matrix = pd.DataFrame(0, index=services_list, columns=services_list)
    edge_matrix = pd.DataFrame('', index=services_list, columns=services_list)

    for _, row in df.iterrows():
        src = row['LADDR_NUM']
        dest = row['RADDR_NUM']
        old_state_info = row['OLDSTATE_NUM']
        new_state_info = row['NEWSTATE_NUM']
        adjacency_matrix.loc[src, dest] = 1

        if isinstance(edge_matrix.loc[src, dest], list):
            edge_matrix.loc[src, dest].extend([old_state_info, new_state_info])
        else:
            edge_matrix.loc[src, dest] = [old_state_info, new_state_info]

    return adjacency_matrix, edge_matrix


# break the data into multiple frames

interval = timedelta(seconds=SECONDS_FOR_FRAME)

# Initialize the start time as the minimum value in the time column
start_time = my_tcp_traces_df.index.min()

# List to store the frames
frames = []

# Iterate over the time range, breaking the data into frames
while start_time <= my_tcp_traces_df.index.max():
    end_time = start_time + interval
    frame = my_tcp_traces_df.loc[(my_tcp_traces_df.index >= start_time) & (my_tcp_traces_df.index < end_time)]
    frames.append(frame)
    start_time = end_time

tensors = []
# Iterate over the individual frames
for index, frame in enumerate(frames):
    adjacency_matrix, edge_info_matrix = create_adjacency_matrix_with_meta_information(frame)
    # if index < 11:
    #     print(edge_info_matrix)
    edge_info_matrix = edge_info_matrix.applymap(get_a_unique_number)
    tensor_2 = edge_info_matrix.values
    tensors.append(tensor_2)

# Generate all combinations of specified LABEL with the specified number of repetitions
combinations = itertools.product(LABEL, repeat=len(frames))
combination_array = np.array(list(combinations))
labels = combination_array.flatten()
labeled_tensors = [(tensor, label) for tensor, label in zip(tensors, labels)]

# Define the file path
file_path = f"data/output/{FILE_NAME}"

with open(file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Tensor", "Label"])  # Write header row
    for tensor, label in zip(tensors, labels):
        serialized_tensor = np.array2string(tensor, separator=',', prefix='[', suffix=']')
        serialized_label = np.array2string(np.array([label]), separator=',', prefix='[', suffix=']')
        writer.writerow([serialized_tensor, serialized_label])


# Serialize the dictionary to a file
with open('serialized_uniq_number_dict.pkl', 'wb') as file:
    pickle.dump(combination_dict, file)


# Use 1306 for normal connection
# min and max for all

# ../wrk2/wrk -D exp -t 5 -c 10 -d 120 -L -s ./wrk2/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 10 & ../wrk2/wrk -D exp -t 5 -c 10 -d 120 -L -s ./wrk2/scripts/social-network/read-home-timeline.lua http://localhost:8080/wrk2-api/home-timeline/read -R 10 & ../wrk2/wrk -D exp -t 5 -c 10 -d 120 -L -s ./wrk2/scripts/social-network/read-user-timeline.lua http://localhost:8080/wrk2-api/user-timeline/read -R 10


# ./wrk -D exp -t 500 -c 500 -d 139 -L -s ./script/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 5000
#
# As the normal activity, three types of social-network activities including ‘read-home-timeline’, ‘compose-post’, and ‘read-user-timeline’ are executed in


# {'0.0.0.0': 1, '10.0.0.252': 2, '127.0.0.1': 3, '172.18.0.1': 4, '172.18.0.10': 5, '172.18.0.17': 6, '172.18.0.18': 7, '172.18.0.19': 8, '172.18.0.2': 9, '172.18.0.20': 10, '172.18.0.21': 11, '172.18.0.23': 12, '172.18.0.24': 13, '172.18.0.25': 14, '172.18.0.26': 15, '172.18.0.27': 16, '172.18.0.28': 17, '172.18.0.29': 18, '172.18.0.3': 19, '172.18.0.30': 20, '172.18.0.4': 21, '172.18.0.5': 22, '172.18.0.6': 23, '44.233.226.27': 24}

# {'0.0.0.0': 1, '10.0.0.252': 2, '127.0.0.1': 3, '172.18.0.1': 4, '172.18.0.10': 5, '172.18.0.17': 6, '172.18.0.18': 7, '172.18.0.19': 8, '172.18.0.2': 9, '172.18.0.20': 10, '172.18.0.21': 11, '172.18.0.23': 12, '172.18.0.24': 13, '172.18.0.25': 14, '172.18.0.26': 15, '172.18.0.27': 16, '172.18.0.28': 17, '172.18.0.29': 18, '172.18.0.3': 19, '172.18.0.30': 20, '172.18.0.5': 21, '172.18.0.6': 22, '44.233.10.108': 23}
