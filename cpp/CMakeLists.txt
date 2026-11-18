cmake_minimum_required(VERSION 3.20)
project(risk_engine_cpp LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

set(RISK_ENGINE_SOURCES src/risk_engine.cpp)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" RISK_ENGINE_PROCESSOR)

if(RISK_ENGINE_PROCESSOR MATCHES "^(x86_64|amd64|x64|i[3-6]86)$")
    list(APPEND RISK_ENGINE_SOURCES src/risk_engine_avx2.cpp)
    set(RISK_ENGINE_HAS_AVX2_KERNEL ON)
endif()

add_library(risk_engine_cpp STATIC ${RISK_ENGINE_SOURCES})
target_include_directories(risk_engine_cpp PUBLIC include)
set_target_properties(risk_engine_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RISK_ENGINE_HAS_AVX2_KERNEL)
    target_compile_definitions(risk_engine_cpp PRIVATE RISK_ENGINE_HAS_AVX2_KERNEL=1)
    if(MSVC)
        set_source_files_properties(src/risk_engine_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/risk_engine_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    endif()
endif()

if(MSVC)
    target_compile_options(risk_engine_cpp PRIVATE /W4 /fp:precise)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(risk_engine_cpp PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()

install(TARGETS risk_engine_cpp ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
