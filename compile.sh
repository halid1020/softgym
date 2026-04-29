. activate softgym-py3.8
cd PyFlex/bindings
rm -rf build
mkdir build
cd build
# Seuss 
if [[ $(hostname) = *"compute-0"* ]] || [[ $(hostname) = *"autobot-"* ]] || [[ $(hostname) = *"yertle"* ]]; then
    export CUDA_BIN_PATH=/usr/local/cuda-9.1
fi
    cmake .. -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
make -j$(nproc)
cd ../../../
