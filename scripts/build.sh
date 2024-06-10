#!/bin/bash


path_python_executable=$(poetry env info --path)
echo "Using Python executable: $path_python_executable"
export PYO3_PYTHON=$path_python_executable
export PYTHON_SYS_EXECUTABLE=$path_python_executable

flags=$1
# export RUSTFLAGS="-Z macro-backtrace"
cargo build -v $flags
