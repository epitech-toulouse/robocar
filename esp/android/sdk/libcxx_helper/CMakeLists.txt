cmake_minimum_required(VERSION 3.6)

# Add this line to fix the NDK r27 / Clang 18 compatibility issue
cmake_policy(SET CMP0057 NEW)

project(opencv_jni_shared)

# dummy target to bring libc++_shared.so into packages
add_library(opencv_jni_shared STATIC dummy.cpp)
