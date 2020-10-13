#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

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

  return 0;
}

