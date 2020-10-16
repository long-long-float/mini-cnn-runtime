#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>

#include <CL/cl.h>

#include "onnx.proto3.pb.h"

using namespace std;
using namespace onnx;

string toDataTypeString(int type) {
  switch (type) {
    case TensorProto::FLOAT: return "FLOAT";
    case TensorProto::INT64: return "INT64";
    default: return "unknown";
  }
}

void printValueInfo(const ValueInfoProto& info) {
  auto& it = info.type();
  cout << info.name() << ", type = ";
  switch (it.value_case()) {
    case TypeProto::kTensorType:
      {
        auto& tensorType = it.tensor_type();
        cout << "(" << toDataTypeString(tensorType.elem_type()) << ", [";
        for (auto& dim : tensorType.shape().dim()) {
          switch (dim.value_case()) {
            case TensorShapeProto::Dimension::kDimValue:
              cout << dim.dim_value();
              break;
            case TensorShapeProto::Dimension::kDimParam:
              cout << dim.dim_param();
              break;
            default:
              break;
          }
          cout << ",";
        }
        cout << "])" << endl;
        break;
      }
    default:
      cout << "unsupported type" << endl;
      break;
  }
}

void printTensor(const TensorProto& tensor) {
  cout << tensor.name() << ": ";
  cout << "(";
  for (auto& dim : tensor.dims()) {
    cout << dim << ",";
  }
  cout << ")";

  switch (tensor.data_type()) {
    case TensorProto::INT64:
      {
        cout << "[";
        auto raw = tensor.raw_data();
        for (int i = 0; i < raw.size(); i += 8) {
          long int v = 0;
          for (int j = 0; j < 8; j++) {
            v |= (unsigned char)raw[i + j] << (j * 8);
          }
          cout << v << " ";
        }
        cout << "]";

        break;
      }
    case TensorProto::FLOAT:
      {
        cout << "[";
        auto raw = tensor.raw_data();
        for (int i = 0; i < raw.size(); i += 4) {
          int v = 0;
          for (int j = 0; j < 4; j++) {
            v |= (unsigned char)raw[i + j] << (j * 4);
          }
          cout << *reinterpret_cast<float*>(&v) << " ";

          if (i > 40) {
            cout << "...";
            break;
          }
        }
        cout << "]";

        break;
      }

    default:
      cout << "(" << tensor.data_type() << ")";
  }
  cout << endl;
}

string toString(const AttributeProto& attr) {
  stringstream ss;
  ss << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto::INT:
      ss << attr.i();
      break;
    case AttributeProto::INTS:
      ss << "[";
      for (auto& i : attr.ints()) {
        ss << i << ",";
      }
      ss << "]";
      break;
    default:
      ss << "(" << attr.type() << ")";
      break;
  }
  return ss.str();
}

class CLError {
public:
  cl_int code;
  CLError(cl_int code) : code(code) {}
};

cl_int clErrorCheck(cl_int result) {
  if (result != CL_SUCCESS) {
    throw CLError(result);
  }
  return result;
}

int main(int argc, char const** argv) {
  ModelProto model;

  fstream input("./mnist.onnx", ios::in | ios::binary);
  model.ParseFromIstream(&input);

  auto& graph = model.graph();

  cout << "Initializer: " << endl;
  for (auto& init : graph.initializer()) {
    printTensor(init);
  }
  cout << endl;

  cout << "Inputs: " << endl;
  for (auto& input : graph.input()) {
    printValueInfo(input);
  }
  cout << endl;

  cout << "Outputs: " << endl;
  for (auto& output : graph.output()) {
    printValueInfo(output);
  }
  cout << endl;

  cout << "ValueInfo: " << endl;
  for (auto& info : graph.value_info()) {
    printValueInfo(info);
  }
  cout << endl;

  cout << "Nodes:" << endl;
  for (auto& node : graph.node()) {
    cout << node.op_type() << ": ";
    for (auto& attr : node.attribute()) {
      cout << toString(attr) << ", ";
    }
    cout << endl;

    cout << "  ";
    for (auto &input : node.input()) {
      cout << input << ", ";
    }
    cout << " --> " << endl;

    cout << "  ";
    for (auto &output : node.output()) {
      cout << output << ", ";
    }
    cout << endl;
  }
  cout << endl;

  cl_uint numPlatforms = 0;
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_uint numDevices = 0;

  clErrorCheck(clGetPlatformIDs(1, &platformId, &numPlatforms));
  clErrorCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices));

  cl_int ret;
  cl_context context = clCreateContext(nullptr, numDevices, &deviceId, nullptr, nullptr, &ret);
  clErrorCheck(ret);

  cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, &ret);
  clErrorCheck(ret);

  const int bufferSize = 128;
  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &ret);
  clErrorCheck(ret);

  ifstream kernelIfs("./kernel.cl");
  string kernelStr = string(istreambuf_iterator<char>(kernelIfs), istreambuf_iterator<char>());
  char *strptr = const_cast<char*>(kernelStr.c_str()); // strptr must not be rewrote.
  const size_t kernelSize = kernelStr.size();

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&strptr, &kernelSize, &ret);
  clErrorCheck(ret);

  clErrorCheck(clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr));

  cl_kernel kernel = clCreateKernel(program, "hello", &ret);
  clErrorCheck(ret);

  clErrorCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf));

  clErrorCheck(clEnqueueTask(queue, kernel, 0, nullptr, nullptr));

  char str[128] = {0};
  clErrorCheck(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufferSize, str, 0, nullptr, nullptr));

  cout << str << endl;

  // TODO: Release there objects when ever an error is occured.
  clErrorCheck(clFlush(queue));
  clErrorCheck(clFinish(queue));
  clErrorCheck(clReleaseKernel(kernel));
  clErrorCheck(clReleaseProgram(program));
  clErrorCheck(clReleaseMemObject(buf));
  clErrorCheck(clReleaseCommandQueue(queue));
  clErrorCheck(clReleaseContext(context));

  return 0;
}

