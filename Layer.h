#pragma once

#include "Variable.h"

namespace minicnn {

class Layer {
 public:
  std::string name;

  using Variables = std::vector<std::shared_ptr<Variable>>;
  Variables inputs;
  Variables outputs;

  using Attributes = std::vector<onnx::AttributeProto>;
  Attributes attributes;

  std::shared_ptr<Layer> child;
  std::vector<std::shared_ptr<Layer>> parents;

  Layer(std::string name, const Variables&& inputs, const Variables&& outputs,
        const Attributes&& attributes)
      : name(name),
        inputs(inputs),
        outputs(outputs),
        attributes(attributes),
        child(nullptr) {}
  Layer(std::string name, const Variables& inputs, const Variables& outputs,
        const Attributes& attributes)
      : name(name),
        inputs(inputs),
        outputs(outputs),
        attributes(attributes),
        child(nullptr) {}

  std::string toString() const;
};

}  // namespace minicnn
