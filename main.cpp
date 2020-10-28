#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx.proto3.pb.h"

using namespace std;
using namespace onnx;

template <typename... T>
struct TypeDisplayer;

struct Shape {
  int x, y, z, w;

  // Shape() : x(0), y(0), z(0), w(0) {}
};

enum class ElementType {
  Float,
  Int64,
  Unknown,
};

template <ElementType T>
struct ElementType2Cpp {
  using t = void;
};
template <>
struct ElementType2Cpp<ElementType::Float> {
  using t = float;
};
template <>
struct ElementType2Cpp<ElementType::Int64> {
  using t = long long;
};

size_t getSize(ElementType type) {
  switch (type) {
    case ElementType::Float:
      return sizeof(ElementType2Cpp<ElementType::Float>::t);
    case ElementType::Int64:
      return sizeof(ElementType2Cpp<ElementType::Int64>::t);
    default:
      return 0;
  }
}

ElementType toElementType(int type) {
  switch (type) {
    case TensorProto::FLOAT:
      return ElementType::Float;
    case TensorProto::INT64:
      return ElementType::Int64;
    default:
      return ElementType::Unknown;
  }
}

std::string toDataTypeString(int type) {
  switch (type) {
    case TensorProto::FLOAT:
      return "FLOAT";
    case TensorProto::INT64:
      return "INT64";
    default:
      return "unknown";
  }
}

class Tensor {
 public:
  std::string name;

  ElementType elemType;
  const std::string& raw;

  const size_t size;

  Tensor(const std::string& name, ElementType elemType, const std::string& raw)
      : name(name),
        elemType(elemType),
        raw(raw),
        size(raw.size() / getSize(elemType)) {}
  Tensor(const TensorProto& tensor)
      : Tensor(tensor.name(), toElementType(tensor.data_type()),
               tensor.raw_data()) {}

  template <ElementType T>
  std::vector<typename ElementType2Cpp<T>::t> getDataAs() const {
    using Elem = typename ElementType2Cpp<T>::t;

    assert(elemType == T);

    if (elemType == T) {
      std::vector<Elem> vec(raw.size() / sizeof(Elem));
      for (int i = 0; i < raw.size(); i += sizeof(Elem)) {
        Elem v = Elem();
        for (int j = 0; j < sizeof(Elem); j++) {
          v |= (Elem)raw[i + j] << (j * 8);
        }
        vec[i / sizeof(Elem)] = v;
      }
      return vec;
    } else {
      return std::vector<Elem>();
    }
  }
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

  // vector<TensorShapeProto::Dimension> unresolvedShape;

  shared_ptr<Tensor> tensor;

  Variable(const string& name) : name(name) {
    shape.s.x = shape.s.y = shape.s.z = shape.s.w = 0;
  }

  Variable(const ValueInfoProto& info, std::shared_ptr<Tensor> tensor)
      : Variable(info.name()) {
    auto& it = info.type();
    switch (it.value_case()) {
      case TypeProto::kTensorType: {
        auto& tensorType = it.tensor_type();

        elemType = toElementType(tensorType.elem_type());

        // infer shape from tensor
        int varIndex = -1;
        int knownSize = 1;
        int i = 0;
        for (auto& dim : tensorType.shape().dim()) {
          switch (dim.value_case()) {
            case TensorShapeProto::Dimension::kDimValue: {
              int v = dim.dim_value();
              shape.raw[i] = v;
              knownSize *= v;
              break;
            }
            case TensorShapeProto::Dimension::kDimParam:
              // dim.dim_param();
              varIndex = i;
              break;
            default:
              break;
          }
          i++;
        }
        if (varIndex >= 0) {
          shape.raw[varIndex] = tensor->size / knownSize;
        }

        break;
      }
      default:
        throw runtime_error("unsupported type: " + to_string(it.value_case()));
        break;
    }

    this->tensor = tensor;
  }

  // void assignShape(const Shape& shape) { this->shape.s = shape; }

  size_t elementCount() const {
    size_t s = 1;
    for (int i = 0; i < 4; i++) s *= shape.raw[i];
    return s;
  }

  size_t size() const { return getSize(elemType) * elementCount(); }

  string toString() const {
    stringstream ss;
    ss << name << ": [";
    // if (shape.s.x == 0 && shape.s.y == 0 && shape.s.z == 0 && shape.s.w == 0)
    // {
    //   ss << "? ";
    //   for (auto& dim : unresolvedShape) {
    //     switch (dim.value_case()) {
    //       case TensorShapeProto::Dimension::kDimValue:
    //         ss << dim.dim_value();
    //         break;
    //       case TensorShapeProto::Dimension::kDimParam:
    //         ss << dim.dim_param();
    //         break;
    //       default:
    //         break;
    //     }
    //     ss << ",";
    //   }
    // } else {
    for (int i = 0; i < 4; i++) {
      ss << shape.raw[i] << ",";
    }
    // }
    ss << "]";
    return ss.str();
  }
};

class DeviceVariable {
 public:
  const std::shared_ptr<Variable> var;

  const size_t size;

  DeviceVariable(std::shared_ptr<Variable> var, cl_context context)
      : var(var), size(var->size()), context(context) {
    cl_int ret = 0;
    buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE, var->size(), nullptr, &ret);
    clErrorCheck(ret);
  }

  ~DeviceVariable() {
    if (buffer != nullptr) {
      clReleaseMemObject(buffer);
    }
  }

  cl_mem buffer() const { return buffer; }

 private:
  cl_context context;
  cl_mem buffer;
};

class Layer {
 public:
  string name;

  using Variables = std::vector<std::shared_ptr<Variable>>;
  Variables inputs;
  Variables outputs;

  using Attributes = std::vector<AttributeProto>;
  Attributes attributes;

  std::shared_ptr<Layer> child;
  std::vector<std::shared_ptr<Layer>> parents;

  Layer(string name, const Variables&& inputs, const Variables&& outputs,
        const Attributes&& attributes)
      : name(name),
        inputs(inputs),
        outputs(outputs),
        attributes(attributes),
        child(nullptr) {}
  Layer(string name, const Variables& inputs, const Variables& outputs,
        const Attributes& attributes)
      : name(name),
        inputs(inputs),
        outputs(outputs),
        attributes(attributes),
        child(nullptr) {}

  string toString() const {
    stringstream ss;
    ss << name << "\n";
    for (auto& i : inputs) {
      ss << "  " << i->toString() << ", ";
    }
    ss << " => \n";
    for (auto& o : outputs) {
      ss << "  " << o->toString() << ", ";
    }
    return ss.str();
  }
};

void printValueInfo(const ValueInfoProto& info) {
  cout << info.name() << ", type = ";
  auto& it = info.type();
  switch (it.value_case()) {
    case TypeProto::kTensorType: {
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
    case TensorProto::INT64: {
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
    case TensorProto::FLOAT: {
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
  if (argc != 2) {
    cerr << argv[0] << " [onnx file]" << endl;
    return 1;
  }

  std::string onnxFileName(argv[1]);

  ModelProto model;

  fstream input(onnxFileName, ios::in | ios::binary);
  if (!input.is_open()) {
    cerr << "onnx file cannot be opened: " << onnxFileName << endl;
    return 1;
  }
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
    for (auto& input : node.input()) {
      cout << input << ", ";
    }
    cout << " --> " << endl;

    cout << "  ";
    for (auto& output : node.output()) {
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
  for (auto& v : inputData) {
    size_t elemSize = sizeof(decltype(inputData)::value_type);
    int iv = 0;
    std::memcpy(&iv, &v, elemSize);
    for (int i = 0; i < elemSize; i++) {
      inputRaw.push_back(iv & (0xffu << (i * 8)));
    }
  }

  // create layer graph from nodes

  // TODO: support multiple outputs
  std::map<std::string, std::shared_ptr<Variable>> variableMap;
  // the input of the model
  std::shared_ptr<Variable> inputVariable;

  auto& nodes = graph.node();
  std::queue<std::shared_ptr<Layer>> layerQue;

  std::set<const NodeProto*> visitedNodePtr;

  std::shared_ptr<Layer> terminalLayer = std::make_shared<Layer>(
      "terminal",
      Layer::Variables{std::make_shared<Variable>(graph.output(0).name())},
      Layer::Variables{}, Layer::Attributes{});
  std::shared_ptr<Layer> rootLayer;

  layerQue.push(terminalLayer);
  while (!layerQue.empty()) {
    auto currentLayer = layerQue.front();
    layerQue.pop();

    int inputIndex = 0;
    for (auto& input : currentLayer->inputs) {
      auto prevNode = find_if(nodes.begin(), nodes.end(), [&](auto& node) {
        auto& outputs = node.output();
        return find_if(outputs.begin(), outputs.end(), [&](auto& out) {
                 return out == input->name;
               }) != outputs.end();
      });

      if (prevNode != nodes.end() &&
          visitedNodePtr.find(&*prevNode) == visitedNodePtr.end()) {
        visitedNodePtr.insert(&*prevNode);

        // create the previous node
        Layer::Variables inputs, outputs;
        auto& i = prevNode->input();
        auto& o = prevNode->output();

        auto registerVariable = [&](const std::shared_ptr<Variable> v) {
          auto it = variableMap.find(v->name);
          if (it == variableMap.end()) {
            variableMap.emplace(v->name, v);
            return v;
          } else {
            return it->second;
          }
        };

        bool isRootLayer = false;

        std::transform(
            i.begin(), i.end(), std::back_inserter(inputs), [&](auto& n) {
              auto it = inputMap.find(n);
              if (it != inputMap.end()) {
                std::shared_ptr<Tensor> tensor;
                std::shared_ptr<Variable> var;

                auto itt = initMap.find(n);
                if (itt != initMap.end()) {
                  var = std::make_shared<Variable>(it->second, itt->second);
                } else {
                  // this tensor is the input of the graph
                  var = std::make_shared<Variable>(
                      it->second, std::make_shared<Tensor>(
                                      n, ElementType::Float, inputRaw));
                  inputVariable = var;
                  isRootLayer = true;
                }

                return registerVariable(var);
              } else {
                return registerVariable(std::make_shared<Variable>(n));
              }
            });
        std::transform(o.begin(), o.end(), std::back_inserter(outputs),
                       [&](auto& n) {
                         return registerVariable(std::make_shared<Variable>(n));
                       });

        auto& a = prevNode->attribute();
        Layer::Attributes attributes(a.begin(), a.end());

        auto prevLayer = make_shared<Layer>(prevNode->op_type(), inputs,
                                            outputs, attributes);
        prevLayer->child = currentLayer;
        currentLayer->parents.push_back(prevLayer);

        if (isRootLayer) {
          rootLayer = prevLayer;
        }

        layerQue.push(prevLayer);
      }
    }
    inputIndex++;
  }

  cout << "finished creating layer graph" << endl;

  // output layers as dot format
  {
    std::ofstream dotFile("model.dot");
    dotFile << "digraph model {\n";

    auto q = [](std::string s) { return '"' + s + '"'; };

    std::queue<std::shared_ptr<Layer>> que;
    que.push(terminalLayer);
    while (!que.empty()) {
      auto cur = que.front();
      que.pop();

      for (auto layer : cur->parents) {
        que.push(layer);

        auto inputs = layer->inputs;
        if (inputs.size() == 0) {
          inputs.push_back(std::make_shared<Variable>("(null)"));
        }
        for (auto& i : inputs) {
          for (auto& o : layer->outputs) {
            dotFile << q(i->name) << " -> " << q(o->name)
                    << " [label = " << q(layer->name) << "];\n";
          }
        }
      }
    }

    dotFile << "}";

    dotFile.close();
  }

  // resolve shapes of variables
  // Shapes of variables cannot be resolved statically, because arguments of
  // Reshape may be dynamically.
  //
  // for (auto cur = rootLayer; cur != nullptr; cur
  // = cur->child) {
  //   if (cur->name == "Reshape") {
  //     auto& data = cur->inputs[0];
  //     auto& shape = cur->inputs[1];
  //     auto& reshaped = cur->outputs[0];
  //
  //     auto shapeValue = shape->tensor->getDataAs<ElementType::Int64>();
  //     shapeValue.resize(4, 0);
  //
  //     int varIndex = -1;
  //     int knownSize = 1;
  //     for (int i = 0; i < shapeValue.size(); i++) {
  //       if (shapeValue[i] == 0) {
  //         shapeValue[i] = data->shape.raw[i];
  //       }
  //
  //       if (shapeValue[i] == -1) {
  //         varIndex = i;
  //       } else {
  //         knownSize *= shapeValue[i];
  //       }
  //     }
  //     if (varIndex >= 0) {
  //       shapeValue[varIndex] = data->size() / knownSize;
  //     }
  //
  //     Shape newShape;
  //     newShape.x = shapeValue[0];
  //     newShape.y = shapeValue[1];
  //     newShape.z = shapeValue[2];
  //     newShape.w = shapeValue[3];
  //     reshaped->assignShape(newShape);
  //   } else if (cur->name == "terminal") {
  //     // do nothing
  //   } else {
  //     cur->outputs[0]->assignShape(cur->inputs[0]->shape.s);
  //   }
  // }

  for (auto cur = rootLayer; cur != nullptr; cur = cur->child) {
    cout << cur->toString() << endl;
  }
  cout << endl;

  // Initialization of OpenCL

  cl_uint numPlatforms = 0;
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_uint numDevices = 0;

  clErrorCheck(clGetPlatformIDs(1, &platformId, &numPlatforms));
  clErrorCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId,
                              &numDevices));

  cl_int ret;
  cl_context context =
      clCreateContext(nullptr, numDevices, &deviceId, nullptr, nullptr, &ret);
  clErrorCheck(ret);

  ifstream kernelIfs("./kernel.cl");
  string kernelStr =
      string(istreambuf_iterator<char>(kernelIfs), istreambuf_iterator<char>());
  char* strptr =
      const_cast<char*>(kernelStr.c_str());  // strptr must not be rewrote.
  const size_t kernelSize = kernelStr.size();

  cl_program program = clCreateProgramWithSource(
      context, 1, (const char**)&strptr, &kernelSize, &ret);
  clErrorCheck(ret);

  clErrorCheck(
      clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr));

  // Run kernels

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, deviceId, 0, &ret);
  clErrorCheck(ret);

  std::map<std::string, std::shared_ptr<DeviceVariable>> deviceVariableMap;

  auto allocateDeviceVar = [&](std::shared_ptr<Variable> var) {
    auto it = deviceVariableMap.find(var->name);
    if (it != deviceVariableMap.end()) {
      return *it;
    } else {
      auto dv = std::make_shared<DeviceVariable>(var, context);
      deviceVariableMap.emplace(var->name, dv);
      return dv;
    }
  };

  auto writeBuffer = [&](cl_command_queue queue,
                         std::shared_ptr<DeviceVariable> dest,
                         const std::string& src) {
    assert(dest->size == src.size());
    clErrorCheck(clEnqueueWriteBuffer(queue, dest->buffer(), CL_TRUE, 0,
                                      dest->size, src.data(), 0, nullptr,
                                      nullptr));
  };

  auto readBuffer = [&](cl_command_queue queue, std::string& dest,
                        std::shared_ptr<DeviceVariable> src, ) {
    assert(dest->size == src.size());
    clErrorCheck(clEnqueueWriteBuffer(queue, src.data(), CL_TRUE, 0, dest->size,
                                      dest->buffer(), 0, nullptr, nullptr));
  };

  auto inputDVariable = allocateDeviceVar(inputVariable);
  writeBuffer(queue, inputDVariable, inputRaw);

  cl_kernel kernel = clCreateKernel(program, "twice", &ret);
  clErrorCheck(ret);

  using DeviceVariables = std::vector<std::shared_ptr<DeviceVariable>>;
  {
    auto fstLayer = rootLayer;
    cout << "Run " << fstLayer->name << endl;

    DeviceVariables dinputs;
    for (auto& input : fstLayer->inputs) {
      auto dv = allocateDeviceVar(input);
      dinputs.push_back(dv);
      if (dv->var->tensor) {
        writeBuffer(queue, dv, dv->var->tensor->raw);
      } else {
        // It should be done to writeBuffer to dv.
      }

      clErrorCheck(
          clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dv->buffer()));
    }
    DeviceVariables doutputs;
    for (auto& output : fstLayer->outputs) {
      auto dv = allocateDeviceVar(output);
      doutputs.push_back(dv);

      clErrorCheck(
          clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dv->buffer()));
    }

    const size_t globalSize[] = {dinputs[0]->var->elementCount()};
    const size_t localSize[] = {1};
    clErrorCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize,
                                        localSize, 0, nullptr, nullptr));

    string str(28 * 28, '\0');
    readBuffer(queue, str, doutputs[0]);

    cout << "[";
    for (int i = 0; i < str.size(); i += 4) {
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
    cout << "]" << endl;
  }

  cl_kernel kernel = clCreateKernel(program, "hello", &ret);
  clErrorCheck(ret);

  const int bufferSize = 128;
  cl_mem buf =
      clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &ret);
  clErrorCheck(ret);

  clErrorCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf));

  const size_t globalSize[] = {1};
  const size_t localSize[] = {1};
  clErrorCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize,
                                      localSize, 0, nullptr, nullptr));

  char str[128] = {0};
  clErrorCheck(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufferSize, str, 0,
                                   nullptr, nullptr));

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

