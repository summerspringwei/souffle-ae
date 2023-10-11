#!/bin/bash

# Define the name of the environment variable to check
# ENV_VARIABLE_NAME="YOUR_VARIABLE_NAME"

# Check if the environment variable exists
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
    # Check if the value of the environment variable is "REAL"
    echo "Running the real application"
fi
a="12.1"
b="34"
c=123
echo ${a}, ${b},\
    ${c} | tee tmp.csv
