mkdir -p build
cd build 
cmake -DCMAKE_BUILD_TYPE=Release ..
make VERBOSE=1 -j
make bingocpp
#make gtest
cd ..
