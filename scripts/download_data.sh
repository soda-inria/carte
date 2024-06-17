# Download data

OPTIONS=("basic_examples")
INCLUDE_RAW=("False")
INCLUDE_KEN=("False")

conda run -n carte_test python -W ignore scripts/download_data.py -op ${OPTIONS[@]} -ir ${INCLUDE_RAW[@]} -ik ${INCLUDE_KEN[@]}

