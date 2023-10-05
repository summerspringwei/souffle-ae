import sys
from typing import List

import numpy as np


def read_ncu_csv(file_path, fetch_list: List):
    header = []
    units = []
    all_data = []
    with open(file_path, newline='') as csvfile:
        lines = csvfile.readlines()
        for i, line in zip(range(len(lines)), lines):
            # Header
            if i == 0:
                header = line.strip().split('","')
                header[0] = header[0].replace('"', '')
                header[-1] = header[-1].replace('"', '')
            # print(header)
            # print("len: ", len(header))
            elif i == 1:
              units = line.strip().split('","')  
            else:
                com = line.strip().split('","')
                com[0] = com[0].replace('"', '')
                com[-1] = com[-1].replace('"', '')
                # print(com)
                # print("len: ", len(com))
                all_data.append(com)
    # Col to idx
    name_to_idx = {}
    for i, name in zip(range(len(header)), header):
        name_to_idx[name] = i
    fetch_list_idx = [name_to_idx[name] for name in fetch_list]
    
    extract_units = [units[i] for i in fetch_list_idx]
    output = []
    for record in all_data:
        extract_record = [record[i] for i in fetch_list_idx]
        output.append(extract_record)
    return output, extract_units


def filter_invalid_ncu_records(values: List):
    output = []
    for value in values:
        if isinstance(value, list):
            output.append(filter_invalid_ncu_records(value))
        elif isinstance(value, str) and (value == "N/A" or value.find("nan") >= 0):
            output.append(0)
        else:
            output.append(value)
    return output

def get_ncu_sum_of_latency(file_path):
    output, units = read_ncu_csv(file_path, ["Kernel Name", "gpu__time_duration.sum"])
    output = filter_invalid_ncu_records(output)
    for record in output:
        print(record)
    # Always assume latency is the last column
    output_np = np.sum(np.array(output)[:, -1].astype(np.float32).reshape((-1,)))
    print(output_np)


def main():
  get_ncu_sum_of_latency(sys.argv[1])

if __name__ == "__main__":
    main()
