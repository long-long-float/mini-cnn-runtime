#include "Variable.h"

using namespace minicnn;
using namespace onnx;

size_t minicnn::getSize(ElementType type) {
#define DECL_CASE(ty)   \
  case TensorProto::ty: \
    return sizeof(ElementType2Cpp<TensorProto::ty>::t);

  switch (type) {
    DECL_CASE(FLOAT)
    DECL_CASE(INT32)
    DECL_CASE(INT64)
    DECL_CASE(BOOL)
    default:
      throw std::runtime_error("unsupported type: " + std::to_string(type));
      return 0;
  }

#undef DECL_CASE
}

ElementType minicnn::toElementType(int type) {
  return static_cast<ElementType>(type);
}

template <>
std::string minicnn::tensorToStr<std::vector<float>>(const std::vector<float>& raw,
                                            ElementType elemType, Shape shape,
                                            bool omit) {
  std::stringstream ss;
  ss << "[\n";
  assert(elemType == onnx::TensorProto::FLOAT);
  for (int y = 0; y < shape.y; y++) {
    ss << " [";
    for (int x = 0; x < shape.x; x++) {
      int idx = y * shape.x + x;
      ss << std::setw(3) << raw[idx] << " ";

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

Variable::Variable(const onnx::ValueInfoProto& info,
                   std::shared_ptr<Tensor> tensor)
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
      throw std::runtime_error("unsupported type: " + std::to_string(it.value_case()));
      break;
  }

  this->tensor = tensor;
}

size_t Variable::dim() const {
  size_t d = 0;
  for (; d < 4 && shape.raw[d] != 0; d++)
    ;
  return d;
}

size_t Variable::elementCount() const {
  size_t s = 1;
  for (int i = 0; i < dim(); i++) s *= shape.raw[i];
  return s;
}

size_t Variable::size() const {
  // TODO: Ensure elemType is valid
  return getSize(elemType) * elementCount();
}

std::string Variable::toString() const {
  std::stringstream ss;
  ss << name << ": [";
  for (int i = 0; i < 4; i++) {
    ss << shape.raw[i] << ",";
  }
  ss << "]";
  return ss.str();
}

