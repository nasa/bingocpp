mkdir -p build
cd build 
cmake -D CMAKE_BUILD_TYPE=Release ..
make VERBOSE=1 -j
#./run_tests
cd ..
