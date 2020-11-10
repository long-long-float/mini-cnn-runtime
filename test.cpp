#include <gtest/gtest.h>

#include <vector>

#include "OpenCLRuntime.h"
#include "Utility.h"

using namespace minicnn;
using namespace onnx;

TEST(Simple, Add) { EXPECT_TRUE(1 + 1 == 2); }

int idx(int x, int y, int w) { return y * w + x; }

void matmul(const std::vector<float>& a, Shape shapeA,
            const std::vector<float>& b, Shape shapeB, std::vector<float>& yy,
            Shape shapeY) {
  for (int y = 0; y < shapeY.y; y++) {
    for (int x = 0; x < shapeY.x; x++) {
      float sum = 0.0f;
      for (int i = 0; i < shapeA.x; i++) {
        sum += a[idx(i, y, shapeA.x)] * a[idx(x, i, shapeB.x)];
      }
      yy[idx(x, y, shapeY.x)] = sum;
    }
  }
}

TEST(Kernel, MatMul) {
  Layer::Variables mmInput, mmOutput;
  Shape shapeA, shapeB;
  shapeA.set(32, 16, 0, 0);
  shapeB.set(16, 32, 0, 0);
  // shapeA.set(10, 10, 0, 0);
  // shapeB.set(10, 10, 0, 0);
  std::string matA, matB;

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

  auto mmLayer =
      std::make_shared<Layer>("MatMul", mmInput, mmOutput, Layer::Attributes{});

  Shape shapeRef;
  shapeRef.x = shapeB.x;
  shapeRef.y = shapeA.y;
  shapeRef.z = shapeA.z;
  shapeRef.w = shapeA.w;
  std::vector<float> ref(shapeRef.x * shapeRef.y);

  matmul(inputDataA, shapeA, inputDataB, shapeB, ref, shapeRef);

  std::string refStr(tensorToStr(ref, TensorProto::FLOAT, shapeRef, false));

  OpenCLRuntime runtime("./kernel.cl");
  runtime.createKernel("MatMul");
  auto result = runtime.runLayers(mmLayer);

  std::string rawResult(result.second.begin(), result.second.end());
  auto &rv = result.first->var;
  std::string resultStr(tensorToStr(rawResult, rv->elemType, rv->shape.s, false));

  EXPECT_EQ(refStr, resultStr);
}

