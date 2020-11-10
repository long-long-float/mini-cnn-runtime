#pragma once

#include <iomanip>
#include <iostream>

#include "onnx.proto3.pb.h"
namespace minicnn {

using ElementType = onnx::TensorProto::DataType;

template <onnx::TensorProto::DataType T>
struct ElementType2Cpp {
  using t = void;
};
template <>
struct ElementType2Cpp<onnx::TensorProto::FLOAT> {
  using t = float;
};
template <>
struct ElementType2Cpp<onnx::TensorProto::INT32> {
  using t = long;
};
template <>
struct ElementType2Cpp<onnx::TensorProto::INT64> {
  using t = long long;
};
template <>
struct ElementType2Cpp<onnx::TensorProto::BOOL> {
  using t = long long;
};

size_t getSize(ElementType type);
ElementType toElementType(int type);

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
  Tensor(const onnx::TensorProto& tensor)
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

// raw is expected as container of bytes (e.g. vector<unsigned char>, string)
template <class R>
std::string tensorToStr(const R& raw, ElementType elemType, Shape shape,
                        bool omit) {
  std::stringstream ss;
  ss << "[\n";
  int elemSize = getSize(elemType);
  for (int y = 0; y < shape.y; y++) {
    ss << " [";
    for (int x = 0; x < shape.x; x++) {
      int idx = (y * shape.x + x) * elemSize;
      int v = 0;
      for (int i = 0; i < elemSize; i++) {
        unsigned char c = raw[idx + i];
        v |= c << (i * 8);
      }
      ss << std::setw(3) << *reinterpret_cast<float*>(&v) << " ";

      if (omit && x > 10) {
        ss << "...";
        break;
      }
    }
    ss << "]\n";

    if (omit && y > 10) {
      ss << "...\n";
      break;
    }
  }
  ss << "]\n";

  return ss.str();
}

template <>
std::string tensorToStr<std::vector<float>>(const std::vector<float>& raw,
                                            ElementType elemType, Shape shape,
                                            bool omit);

template <class R>
void printTensor(const R& raw, ElementType elemType, Shape shape, bool omit) {
  std::cout << tensorToStr(raw, elemType, shape, omit) << std::endl;
}

// Host-side
class Variable {
 public:
  std::string name;

  union {
    int raw[4];
    Shape s;
  } shape;

  ElementType elemType;
  std::shared_ptr<Tensor> tensor;

  Variable(const std::string& name) : name(name) {
    shape.s.x = shape.s.y = shape.s.z = shape.s.w = 0;
  }

  Variable(const onnx::ValueInfoProto& info, std::shared_ptr<Tensor> tensor);

  size_t dim() const;
  size_t elementCount() const;
  size_t size() const;

  std::string toString() const;
};

}  // namespace minicnn

