#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <memory>
#include <stdexcept>

#include <CL/cl.h>

#include "onnx.proto3.pb.h"

using namespace std;
using namespace onnx;

struct Shape {
  int x, y, z, w;
};

enum class ElementType {
  Float,
  Int64,
  Unknown,
};

ElementType toElementType(int type) {
  switch (type) {
    case TensorProto::FLOAT: return ElementType::Float;
    case TensorProto::INT64: return ElementType::Int64;
    default: return ElementType::Unknown;
  }
}

std::string toDataTypeString(int type) {
  switch (type) {
    case TensorProto::FLOAT: return "FLOAT";
    case TensorProto::INT64: return "INT64";
    default: return "unknown";
  }
}

class Tensor {
  public:
    std::string name;

    ElementType elemType;
    const std::string& raw;

    Tensor(const std::string& name, ElementType elemType, const std::string& raw) : name(name), elemType(elemType), raw(raw) {}
    Tensor(const TensorProto& tensor) : Tensor(tensor.name(), toElementType(tensor.data_type()), tensor.raw_data()) {}
};

// template <ElementType T>
// class TensorImpl : private Tensor {
//
// };

// Host-side
class Variable {
  public:
    string name;

    union {
      int raw[4];
      Shape s;
    } shape;

    ElementType elemType;

    vector<TensorShapeProto::Dimension> unresolvedShape;

    shared_ptr<Tensor> tensor;

    Variable(const string &name) : name(name) {
      shape.s.x = shape.s.y = shape.s.z = shape.s.w = 0;
    }

    Variable(const ValueInfoProto& info, std::shared_ptr<Tensor> tensor) : Variable(info.name()) {
      auto& it = info.type();
      switch (it.value_case()) {
        case TypeProto::kTensorType:
          {
            auto& tensorType = it.tensor_type();

            elemType = toElementType(tensorType.elem_type());
            auto& dim = tensorType.shape().dim();
            unresolvedShape.assign(dim.begin(), dim.end());

            break;
          }
        default:
          throw runtime_error("unsupported type: " + to_string(it.value_case()));
          break;
      }

      this->tensor = tensor;
    }

    string toString() const {
      stringstream ss;
      ss << name << ": [";
      if (shape.s.x == 0 && shape.s.y == 0 && shape.s.z == 0 && shape.s.w == 0) {
        for (auto& dim : unresolvedShape) {
          switch (dim.value_case()) {
            case TensorShapeProto::Dimension::kDimValue:
              ss << dim.dim_value();
              break;
            case TensorShapeProto::Dimension::kDimParam:
              ss << dim.dim_param();
              break;
            default:
              break;
          }
          ss << ",";
        }
      }
      else {
        for (int i = 0; i < 4; i++) {
          ss << shape.raw[i] << ",";
        }
      }
      ss << "]";
      return ss.str();
    }
};

class Layer {
  public:
    string name;

    using Variables = std::vector<std::shared_ptr<Variable>>;
    Variables inputs;
    Variables outputs;

    using Attributes = std::vector<AttributeProto>;
    Attributes attributes;

    shared_ptr<Layer> next;

    Layer(string name, const Variables &&inputs, const Variables &&outputs, const Attributes &&attributes)
      : name(name), inputs(inputs), outputs(outputs), attributes(attributes), next(nullptr) {}
    Layer(string name, const Variables &inputs, const Variables &outputs, const Attributes &attributes)
      : name(name), inputs(inputs), outputs(outputs), attributes(attributes), next(nullptr) {}

    string toString() const {
      stringstream ss;
      ss << name << " ";
      for (auto& i : inputs) {
        ss << i->toString() << ", ";
      }
      ss << " => ";
      for (auto& o : outputs) {
        ss << o->toString() << ", ";
      }
      return ss.str();
    }
};

void printValueInfo(const ValueInfoProto& info) {
  cout << info.name() << ", type = ";
  auto& it = info.type();
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

  std::map<std::string, std::shared_ptr<Tensor>> initMap;
  cout << "Initializer: " << endl;
  for (auto& init : graph.initializer()) {
    initMap.emplace(init.name(), std::make_shared<Tensor>(init));
    printTensor(init);
  }
  cout << endl;

  std::map<std::string, const ValueInfoProto&> inputMap;
  cout << "Inputs: " << endl;
  for (auto& input : graph.input()) {
    // This makes invalid entries that second values are broken.
    // inputMap.insert(std::make_pair(input.name(), input));
    inputMap.emplace(input.name(), input);
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

  // test input
  std::vector<float> inputData;
  for (int i = 0; i < 28 * 28; i++) {
    inputData.push_back(i);
  }
  std::string inputRaw;
  for (auto &v : inputData) {
    size_t elemSize = sizeof(decltype(inputData)::value_type);
    int iv = 0;
    std::memcpy(&iv, &v, elemSize);
    for (int i = 0; i < elemSize; i++) {
      inputRaw.push_back(iv & (0xffu << (i * 8)));
    }
  }

  // TODO: support multiple outputs
  shared_ptr<Layer> rootLayer = make_shared<Layer>("terminal", Layer::Variables{std::make_shared<Variable>(graph.output(0), nullptr)}, Layer::Variables{}, Layer::Attributes{});
  while (true) {
    auto& nodes = graph.node();
    auto prevNode = find_if(nodes.begin(), nodes.end(), [&](auto &node) {
        auto& outputs = node.output();
        // TODO: care multiple outputs
        return find_if(outputs.begin(), outputs.end(), [&](auto &out) {
            return out == rootLayer->inputs[0]->name;
            }) != outputs.end();
        });

    if (prevNode != nodes.end()) {
      Layer::Variables inputs, outputs;
      auto &i = prevNode->input();
      auto &o = prevNode->output();
      std::transform(i.begin(), i.end(), std::back_inserter(inputs), [&](auto &n) {
          auto it = inputMap.find(n);
          if (it != inputMap.end()) {
            auto itt = initMap.find(n);
            if (itt != initMap.end()) {
              return std::make_shared<Variable>(it->second, itt->second);
            }
            else {
              // tensor is the input of the graph
              return std::make_shared<Variable>(it->second, std::make_shared<Tensor>(n, ElementType::Float, inputRaw));
            }
          }
          else {
            return std::make_shared<Variable>(n);
          }
          });
      std::transform(o.begin(), o.end(), std::back_inserter(outputs), [](auto &n) { return std::make_shared<Variable>(n); });

      auto &a = prevNode->attribute();
      Layer::Attributes attributes(a.begin(), a.end());

      auto prevLayer = make_shared<Layer>(prevNode->op_type(), inputs, outputs, attributes);
      prevLayer->next = rootLayer;
      rootLayer = prevLayer;
    }
    else {
      break;
    }
  }

  for (auto &cur = rootLayer; cur != nullptr; cur = cur->next) {
    cout << cur->toString() << endl;
  }
  cout << endl;

  // resolve shapes of variables
  for (auto &cur = rootLayer; cur != nullptr; cur = cur->next) {

  }

  cl_uint numPlatforms = 0;
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_uint numDevices = 0;

  clErrorCheck(clGetPlatformIDs(1, &platformId, &numPlatforms));
  clErrorCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices));

  cl_int ret;
  cl_context context = clCreateContext(nullptr, numDevices, &deviceId, nullptr, nullptr, &ret);
  clErrorCheck(ret);

  cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceId, 0, &ret);
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

  const size_t globalSize[] = {1};
  const size_t localSize[] = {1};
  clErrorCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize, localSize, 0, nullptr, nullptr));

  char str[128] = {0};
  clErrorCheck(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufferSize, str, 0, nullptr, nullptr));

  cout << str << endl;

  // TODO: Release there objects even when an error is occured.
  clErrorCheck(clFlush(queue));
  clErrorCheck(clFinish(queue));
  clErrorCheck(clReleaseKernel(kernel));
  clErrorCheck(clReleaseProgram(program));
  clErrorCheck(clReleaseMemObject(buf));
  clErrorCheck(clReleaseCommandQueue(queue));
  clErrorCheck(clReleaseContext(context));

  return 0;
}

