# Cross toolchain for building the Mupen64Plus plugin for aarch64 Linux from an x86_64 host
# (cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64-linux-gnu.cmake ...).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
