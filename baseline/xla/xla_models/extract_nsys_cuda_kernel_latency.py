import os
import sys
import re

def sum_latency(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        sum = 0
        pattern = r',(\d+)$'
        for line in lines:
            # Use re.search to find the integer at the end of the line
            match = re.search(pattern, line)
            # Check if a match was found
            if match:
                # Extract and return the integer
                integer_at_end = int(match.group(1))
                sum += integer_at_end
        return float(sum)


if __name__=="__main__":
    # print(f"latency: {sum_latency(sys.argv[1])/1e6} ms")
    print(sum_latency(sys.argv[1])/1e6)