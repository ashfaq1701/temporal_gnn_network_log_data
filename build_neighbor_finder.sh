rm -rf build
mkdir build
cd build || exit

cmake ../cpp_libs/neighbor_finder/
cmake --build . --config Release --target neighbor_finder
