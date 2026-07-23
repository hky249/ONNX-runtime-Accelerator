cmake_minimum_required(VERSION 3.10)
project(OnnxRuntimeInference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

execute_process(
    COMMAND brew --prefix onnxruntime
    OUTPUT_VARIABLE ONNXRUNTIME_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
    if(NOT EXISTS ${ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")
    endif()
endif()

if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNX Runtime not found! Please set ONNXRUNTIME_ROOT")
endif()

message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_ROOT}")

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(onnx_inference
    main.cpp
    image_utils.cpp
)

# Ensure the target can find OpenCV and ONNX Runtime headers
target_include_directories(onnx_inference PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(onnx_inference 
    ${OpenCV_LIBS}
    onnxruntime
)

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")