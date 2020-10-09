#include <iostream>
#include <fstream>

#include "onnx.proto3.pb.h"

using namespace std;

int main(int argc, char const** argv) {
  onnx::ModelProto model;

  fstream input("./mnist.onnx", ios::in | ios::binary);
  model.ParseFromIstream(&input);

  cout << model.DebugString() << endl;

  return 0;
}

