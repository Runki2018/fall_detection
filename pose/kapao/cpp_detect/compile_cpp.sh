#-std=gnu++11
g++ detect.cpp -o detect.exe $(pkg-config --cflags --libs opencv) -std=c++11 &
echo "Compile Done."  &
./detect.exe