## SoftGym
export SOFTGYM_PATH=${PWD}
export PYFLEXROOT=${SOFTGYM_PATH}/PyFlex
export PYTHONPATH=${SOFTGYM_PATH}:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH