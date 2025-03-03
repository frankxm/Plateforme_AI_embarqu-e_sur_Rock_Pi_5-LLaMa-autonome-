#!/bin/bash
# Debug / Release / RelWithDebInfo
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi


C_COMPILER=/d/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu.tar/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc.exe
CXX_COMPILER=/d/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu.tar/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++.exe
STRIP_COMPILER=/d/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu.tar/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-strip.exe




TARGET_ARCH=aarch64
TARGET_PLATFORM=linux
if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi
# 获取脚本所在目录
ROOT_PWD=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
#ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
  if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create build directory: ${BUILD_DIR}"
        exit 1
    fi
fi

#cd ${BUILD_DIR}
cd "${BUILD_DIR}" || exit 1
cmake ../.. \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON\
    -DCMAKE_CXX_FLAGS="-I/d/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu.tar/gcc-arm-10.2-2020.11-mingw-w64-i686-aarch64-none-linux-gnu/aarch64-none-linux-gnu/include/c++/10.2.1"\
    -G "Unix Makefiles"  # 强制使用 Unix Makefiles 而不是 MSVC 编译器




make -j4

