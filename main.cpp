#include <CL/cl.h>

#include <fstream>
#include <iomanip>
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

struct Shape {
  int x, y, z, w;

  // Shape() : x(0), y(0), z(0), w(0) {}

  void set(int _x, int _y, int _z, int _w) {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }
};

using ElementType = TensorProto::DataType;

template <TensorProto::DataType T>
struct ElementType2Cpp {
  using t = void;
};
template <>
struct ElementType2Cpp<TensorProto::FLOAT> {
  using t = float;
};
template <>
struct ElementType2Cpp<TensorProto::INT32> {
  using t = long;
};
template <>
struct ElementType2Cpp<TensorProto::INT64> {
  using t = long long;
};
template <>
struct ElementType2Cpp<TensorProto::BOOL> {
  using t = long long;
};

size_t getSize(ElementType type) {
#define DECL_CASE(ty)   \
  case TensorProto::ty: \
    return sizeof(ElementType2Cpp<TensorProto::ty>::t);

  switch (type) {
    DECL_CASE(FLOAT)
    DECL_CASE(INT32)
    DECL_CASE(INT64)
    DECL_CASE(BOOL)
    default:
      throw runtime_error("unsupported type: " + to_string(type));
      return 0;
  }

#undef DECL_CASE
}

ElementType toElementType(int type) { return static_cast<ElementType>(type); }

template <class R>
void printTensor(const R& raw, ElementType elemType, Shape shape);

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

  size_t dim() const {
    size_t d = 0;
    for (; d < 4 && shape.raw[d] != 0; d++)
      ;
    return d;
  }

  size_t elementCount() const {
    size_t s = 1;
    for (int i = 0; i < dim(); i++) s *= shape.raw[i];
    return s;
  }

  size_t size() const {
    // TODO: Ensure elemType is valid
    return getSize(elemType) * elementCount();
  }

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
    _buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE, var->size(), nullptr, &ret);
    clErrorCheck(ret);
  }

  ~DeviceVariable() {
    if (_buffer != nullptr) {
      clReleaseMemObject(_buffer);
    }
  }

  cl_mem buffer() const { return _buffer; }
  // For clSetKernelArg
  void* bufferPtr() const { return (void*)&_buffer; }

 private:
  cl_context context;
  cl_mem _buffer;
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

class OpenCLRuntime {
 private:
  cl_uint numPlatforms = 0;
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_uint numDevices = 0;
  cl_context context;
  cl_program program;
  cl_command_queue queue;

  std::map<std::string, cl_kernel> kernels;

  std::map<std::string, std::shared_ptr<DeviceVariable>> deviceVariableMap;

 public:
  OpenCLRuntime(const std::string& kernelPath) {
    clErrorCheck(clGetPlatformIDs(1, &platformId, &numPlatforms));
    clErrorCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1,
                                &deviceId, &numDevices));

    cl_int ret;
    context =
        clCreateContext(nullptr, numDevices, &deviceId, nullptr, nullptr, &ret);
    clErrorCheck(ret);

    ifstream kernelIfs(kernelPath);
    string kernelStr = string(istreambuf_iterator<char>(kernelIfs),
                              istreambuf_iterator<char>());
    char* strptr =
        const_cast<char*>(kernelStr.c_str());  // strptr must not be rewrote.
    const size_t kernelSize = kernelStr.size();

    program = clCreateProgramWithSource(context, 1, (const char**)&strptr,
                                        &kernelSize, &ret);
    clErrorCheck(ret);

    clErrorCheck(
        clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr));

    // It's enabled at higher than OpenCL 2.0
    // cl_queue_properties prop[] = {
    //   CL_QUEUE_PROPERTIES | CL_QUEUE_PROFILING_ENABLE,
    //   0
    // };
    // cl_command_queue queue =
    //     clCreateCommandQueueWithProperties(context, deviceId, prop, &ret);
    // clErrorCheck(ret);
    queue = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE,
                                 &ret);
    clErrorCheck(ret);
  }

  ~OpenCLRuntime() {
    clErrorCheck(clFlush(queue));
    clErrorCheck(clFinish(queue));
    clErrorCheck(clReleaseCommandQueue(queue));

    for (auto p : kernels) {
      clErrorCheck(clReleaseKernel(p.second));
    }

    clErrorCheck(clReleaseProgram(program));
    clErrorCheck(clReleaseContext(context));
  }

  void writeBuffer(std::shared_ptr<DeviceVariable> dest,
                   const std::string& src) {
    assert(dest->size == src.size());
    clErrorCheck(clEnqueueWriteBuffer(queue, dest->buffer(), CL_TRUE, 0,
                                      dest->size, src.data(), 0, nullptr,
                                      nullptr));
  }

  void readBuffer(std::vector<unsigned char>& dest,
                  std::shared_ptr<DeviceVariable> src) {
    assert(dest.size() == src->size);
    clErrorCheck(clEnqueueReadBuffer(queue, src->buffer(), CL_TRUE, 0,
                                     dest.size(), dest.data(), 0, nullptr,
                                     nullptr));
  }

  void createKernel(const std::string& name) {
    cl_int ret;
    kernels.emplace(name, clCreateKernel(program, name.c_str(), &ret));
    clErrorCheck(ret);
  }

  std::shared_ptr<DeviceVariable> allocateDeviceVar(
      std::shared_ptr<Variable> var) {
    auto it = deviceVariableMap.find(var->name);
    if (it != deviceVariableMap.end()) {
      return it->second;
    } else {
      auto dv = std::make_shared<DeviceVariable>(var, context);
      deviceVariableMap.emplace(var->name, dv);
      return dv;
    }
  }

  std::pair<std::shared_ptr<DeviceVariable>, std::vector<unsigned char>>
  runLayers(std::shared_ptr<Layer> rootLayer) {
    using DeviceVariables = std::vector<std::shared_ptr<DeviceVariable>>;
    int ii = 0;

    std::shared_ptr<DeviceVariable> result;
    std::vector<unsigned char> rawResult;

    for (auto currentLayer = rootLayer; currentLayer != nullptr;
         currentLayer = currentLayer->child) {
      if (ii >= 1) break;

      const std::string layerName = currentLayer->name;

      cout << "Run " << layerName << endl;

      if (layerName == "Reshape") {
        auto& data = currentLayer->inputs[0];
        auto& shape = currentLayer->inputs[1];
        auto& reshaped = currentLayer->outputs[0];

        // TODO: Read tensor from device memory
        auto shapeValue = shape->tensor->getDataAs<TensorProto::INT64>();
        shapeValue.resize(4, 0);

        int varIndex = -1;
        int knownSize = 1;
        for (int i = 0; i < shapeValue.size(); i++) {
          if (shapeValue[i] == 0) {
            shapeValue[i] = data->shape.raw[i];
          }

          if (shapeValue[i] == -1) {
            varIndex = i;
          } else {
            knownSize *= shapeValue[i];
          }
        }
        if (varIndex >= 0) {
          shapeValue[varIndex] = data->elementCount() / knownSize;
        }

        Shape newShape;
        newShape.x = shapeValue[0];
        newShape.y = shapeValue[1];
        newShape.z = shapeValue[2];
        newShape.w = shapeValue[3];

        reshaped->shape.s = newShape;
        reshaped->elemType = data->elemType;
      } else if (layerName == "Conv") {
        auto& X = currentLayer->inputs[0];
        auto& W = currentLayer->inputs[1];
        auto& B = currentLayer->inputs[2];
        auto& Y = currentLayer->outputs[0];

      } else if (layerName == "MatMul") {
        auto& A = currentLayer->inputs[0];
        auto& B = currentLayer->inputs[1];
        auto& Y = currentLayer->outputs[0];

        Shape newShape;
        newShape.x = B->shape.s.x;
        newShape.y = A->shape.s.y;
        newShape.z = A->shape.s.z;
        newShape.w = A->shape.s.w;

        Y->shape.s = newShape;
        Y->elemType = A->elemType;
      }

      auto kernel = kernels[layerName];

      int argCount = 0;

      DeviceVariables dinputs;
      for (auto& input : currentLayer->inputs) {
        cout << input->toString() << endl;
        auto dv = allocateDeviceVar(input);
        dinputs.push_back(dv);
        if (dv->var->tensor) {
          writeBuffer(dv, dv->var->tensor->raw);
        } else {
          // It should be done to writeBuffer to dv.
        }

        clErrorCheck(clSetKernelArg(kernel, argCount++, sizeof(cl_mem),
                                    dv->bufferPtr()));
        clErrorCheck(
            clSetKernelArg(kernel, argCount++, sizeof(Shape), &dv->var->shape));
      }
      DeviceVariables doutputs;
      for (auto& output : currentLayer->outputs) {
        cout << output->toString() << endl;
        auto dv = allocateDeviceVar(output);
        doutputs.push_back(dv);

        clErrorCheck(clSetKernelArg(kernel, argCount++, sizeof(cl_mem),
                                    dv->bufferPtr()));
        clErrorCheck(
            clSetKernelArg(kernel, argCount++, sizeof(Shape), &dv->var->shape));
      }

      // const size_t globalSize[] = {dinputs[0]->var->elementCount()};
      const size_t globalSize[] = {1};
      const size_t localSize[] = {1};
      cl_event handler;
      clErrorCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize,
                                          localSize, 0, nullptr, &handler));

      clErrorCheck(clWaitForEvents(1, &handler));
      {
        cl_ulong start, end;
        clErrorCheck(clGetEventProfilingInfo(handler,
                                             CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &start, NULL));
        clErrorCheck(clGetEventProfilingInfo(handler, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &end, NULL));
        std::cout << "execution time: " << (end - start) / (1000 * 1000)
                  << "(ms)" << endl;
      }
      clErrorCheck(clReleaseEvent(handler));

      result = doutputs[0];
      rawResult.resize(result->size, 0);
      readBuffer(rawResult, result);

      std::ofstream ofs("output.bin", std::ios::binary);
      for (auto v : rawResult) {
        ofs << v;
      }

      printTensor(rawResult, result->var->elemType, result->var->shape.s);

      ii++;
    }

    return std::make_pair(result, rawResult);
  }

  std::string getBuildLog() {
    size_t logSize = 0;

    clErrorCheck(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG,
                                       0, nullptr, &logSize));

    std::vector<char> log(logSize + 1);
    clErrorCheck(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG,
                                       logSize, log.data(), nullptr));
    log[logSize] = '\0';

    return std::string(log.data(), log.size() - 1);
  }
};

void printValueInfo(const ValueInfoProto& info) {
  cout << info.name() << ", type = ";
  auto& it = info.type();
  switch (it.value_case()) {
    case TypeProto::kTensorType: {
      auto& tensorType = it.tensor_type();
      cout << "(" << TensorProto::DataType_Name(tensorType.elem_type())
           << ", [";
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
          v |= (unsigned char)raw[i + j] << (j * 8);
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

template <class T>
std::string vecToStr(const std::vector<T> vec) {
  std::string ret;
  size_t elemSize = sizeof(T);
  for (auto& v : vec) {
    std::vector<unsigned char> d(elemSize);
    std::memcpy(d.data(), &v, elemSize);
    for (auto& dd : d) {
      ret.push_back(dd);
    }
  }
  return ret;
}

template <class R>
void printTensor(const R& raw, ElementType elemType, Shape shape) {
  cout << "[" << endl;
  int elemSize = getSize(elemType);
  for (int y = 0; y < shape.y; y++) {
    cout << " [";
    for (int x = 0; x < shape.x; x++) {
      int idx = (y * shape.x + x) * elemSize;
      int v = 0;
      for (int i = 0; i < elemSize; i++) {
        unsigned char c = raw[idx + i];
        v |= c << (i * 8);
      }
      cout << std::setw(3) << *reinterpret_cast<float*>(&v) << " ";

      if (x > 10) {
        cout << "...";
        break;
      }
    }
    cout << "]" << endl;

    if (y > 10) {
      cout << "..." << endl;
      break;
    }
  }
  cout << "]" << endl;
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
    size_t elemSize = sizeof(float);
    int iv = *reinterpret_cast<int*>(&v);
    for (int i = 0; i < elemSize; i++) {
      inputRaw.push_back(iv >> (i * 8) & 0xffu);
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
                                      n, TensorProto::FLOAT, inputRaw));
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

  std::unique_ptr<OpenCLRuntime> runtime;
  try {
    runtime.reset(new OpenCLRuntime("./kernel.cl"));

    auto inputDVariable = runtime->allocateDeviceVar(inputVariable);
    runtime->writeBuffer(inputDVariable, inputRaw);

    runtime->createKernel("Reshape");
    runtime->createKernel("Conv");
    runtime->createKernel("MatMul");

    bool debug = false;

    // Create a MatMul layer for testing
    Layer::Variables mmInput, mmOutput;
    Shape shapeA, shapeB;
    if (debug) {
      shapeA.set(32, 16, 0, 0);
      shapeB.set(16, 32, 0, 0);
    } else {
      shapeA.set(512, 512, 0, 0);
      shapeB.set(512, 512, 0, 0);
    }
    std::string matA, matB;
    {
      std::vector<float> inputDataA;
      std::vector<float> inputDataB;
      for (int i = 0; i < shapeA.x * shapeA.y; i++) {
        inputDataA.push_back(i);
      }
      for (int i = 0; i < shapeB.x * shapeB.y; i++) {
        inputDataB.push_back(i);
      }

      matA = vecToStr(inputDataA);
      matB = vecToStr(inputDataB);
    }
    auto tensorA = std::make_shared<Tensor>("A", TensorProto::FLOAT, matA);
    auto tensorB = std::make_shared<Tensor>("B", TensorProto::FLOAT, matB);
    auto varA = std::make_shared<Variable>("A");
    auto varB = std::make_shared<Variable>("B");
    varA->elemType = TensorProto::FLOAT;
    varA->tensor = tensorA;
    varA->shape.s = shapeA;
    varB->elemType = TensorProto::FLOAT;
    varB->tensor = tensorB;
    varB->shape.s = shapeB;
    mmInput.push_back(varA);
    mmInput.push_back(varB);
    mmOutput.push_back(std::make_shared<Variable>("Y"));
    auto mmLayer = std::make_shared<Layer>("MatMul", mmInput, mmOutput,
                                           Layer::Attributes{});
    rootLayer = mmLayer;

    // printTensor(matA, varA->elemType, varA->shape.s);
    // printTensor(matB, varB->elemType, varB->shape.s);

    runtime->runLayers(rootLayer);

  } catch (const CLError& ex) {
    if (ex.code == CL_BUILD_PROGRAM_FAILURE) {
      std::cerr << "Build error!" << std::endl;
      std::cerr << runtime->getBuildLog() << std::endl;

      return -1;
    } else {
      std::cerr << "OpenCL Error: " << ex.code << std::endl;
    }
  }

  return 0;
}
