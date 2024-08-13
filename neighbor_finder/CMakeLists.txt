cmake_minimum_required(VERSION 3.28)
project(neighbor_finder)

set(CMAKE_CXX_STANDARD 17)

# Get Python include directory
execute_process(
        COMMAND python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")

# Get Python library directory
execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        OUTPUT_VARIABLE PYTHON_LIBRARY
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python library directory: ${PYTHON_LIBRARY}")

# Manually specify NumPy include directory
execute_process(
        COMMAND python3 -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "NumPy include directory: ${PYTHON_NUMPY_INCLUDE_DIRS}")

execute_process(
        COMMAND python3 -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyBind11 include directory: ${pybind11_DIR}")

include_directories(${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIRS})

# Grab Python, 3.8 or newer
find_package(pybind11 REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(neighbor_finder SHARED neighbor_finder.cpp thread_pool.h)

# Link the libraries
target_link_libraries(neighbor_finder PRIVATE pybind11::module ${Python3_LIBRARIES} ${NumPy_LIBRARIES})

# Create an executable for testing (optional)
add_executable(test_neighbor_finder neighbor_finder.cpp thread_pool.h test_neighbor_finder.cpp)

# Link the executable with the neighbor_finder library
target_link_libraries(test_neighbor_finder PRIVATE neighbor_finder)

# Set the output directory for the executable
set_target_properties(test_neighbor_finder PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
