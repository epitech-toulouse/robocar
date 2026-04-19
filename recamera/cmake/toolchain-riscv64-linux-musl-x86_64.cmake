include(CMakeForceCompiler)

# The Generic system name is used for embedded targets (targets without OS) in
# CMake
set( CMAKE_SYSTEM_NAME          Linux )
set( CMAKE_SYSTEM_PROCESSOR     riscv )
set( ARCH riscv )
set( CROSS_COMPILE riscv64-unknown-linux-musl- )

get_filename_component(TOOLCHAIN_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
set(TOOLCHAIN_BIN_DIR ${TOOLCHAIN_ROOT_DIR}/host-tools/gcc/riscv64-linux-musl-x86_64/bin)

if(EXISTS "${TOOLCHAIN_BIN_DIR}")
    set(CMAKE_C_COMPILER ${TOOLCHAIN_BIN_DIR}/${CROSS_COMPILE}gcc)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_BIN_DIR}/${CROSS_COMPILE}g++)
    set(CMAKE_OBJCOPY ${TOOLCHAIN_BIN_DIR}/${CROSS_COMPILE}objcopy
            CACHE FILEPATH "The toolchain objcopy command " FORCE )
else()
    set(CMAKE_C_COMPILER ${CROSS_COMPILE}gcc)
    set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++)
    set(CMAKE_OBJCOPY ${CROSS_COMPILE}objcopy
            CACHE FILEPATH "The toolchain objcopy command " FORCE )
endif()

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" )
set( CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=c906fdv" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=c906fdv" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" ) 


if(DEFINED ENV{SG200X_SDK_PATH})
	set(SG200X_SDK_PATH $ENV{SG200X_SDK_PATH})
endif()
