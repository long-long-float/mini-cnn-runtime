COMPILER=clang++
OPTIONS=-std=c++14 -I/usr/local/include -L/usr/local/lib -lprotobuf -lOpenCL

cnnr: main.cpp onnx.proto3.pb.o
	$(COMPILER) $(OPTIONS) -o $@ $^

onnx.proto3.pb.o: onnx.proto3.pb.cc
	$(COMPILER) $(OPTIONS) -c -o $@ $^

test: test.cpp
	$(COMPILER) -std=c++14 $^ -o $@ -L/usr/local/lib -lgtest -lgtest_main -pthread

