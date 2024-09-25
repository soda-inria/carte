# Download data. See README for information on the variables.

ENV_NAME="myenv"  # Change the environment name accordingly
OPTIONS="basic_examples" 
INCLUDE_RAW="False"
INCLUDE_KEN="False"

conda run -n $ENV_NAME python -W ignore scripts/download_data.py -op $OPTIONS -ir $INCLUDE_RAW -ik $INCLUDE_KEN

