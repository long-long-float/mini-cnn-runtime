#include "OpenCLRuntime.h"

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

#include "Utility.h"
using namespace std;
using namespace onnx;
using namespace minicnn;

template <class R>
void printTensor(const R& raw, ElementType elemType, Shape shape);

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
