#!/bin/bash


path_python_executable=$(pipenv --py)
echo "Using Python executable: $path_python_executable"
export PYO3_PYTHON=$path_python_executable
export PYTHON_SYS_EXECUTABLE=$path_python_executable
cargo build -v
