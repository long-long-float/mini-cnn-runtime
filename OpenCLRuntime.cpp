#include "OpenCLRuntime.h"
#include <fstream>

using namespace minicnn;
using namespace std;

cl_int minicnn::clErrorCheck(cl_int result) {
  if (result != CL_SUCCESS) {
    throw CLError(result);
  }
  return result;
}

OpenCLRuntime::OpenCLRuntime(const string& kernelPath) {
  clErrorCheck(clGetPlatformIDs(1, &platformId, &numPlatforms));
  clErrorCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId,
                              &numDevices));

  cl_int ret;
  context =
      clCreateContext(nullptr, numDevices, &deviceId, nullptr, nullptr, &ret);
  clErrorCheck(ret);

  ifstream kernelIfs(kernelPath);
  string kernelStr =
      string(istreambuf_iterator<char>(kernelIfs), istreambuf_iterator<char>());
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
  queue =
      clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &ret);
  clErrorCheck(ret);
}

OpenCLRuntime::~OpenCLRuntime() {
  clErrorCheck(clFlush(queue));
  clErrorCheck(clFinish(queue));
  clErrorCheck(clReleaseCommandQueue(queue));

  for (auto p : kernels) {
    clErrorCheck(clReleaseKernel(p.second));
  }

  clErrorCheck(clReleaseProgram(program));
  clErrorCheck(clReleaseContext(context));
}

void OpenCLRuntime::writeBuffer(std::shared_ptr<DeviceVariable> dest,
                                const std::string& src) {
  assert(dest->size == src.size());
  clErrorCheck(clEnqueueWriteBuffer(queue, dest->buffer(), CL_TRUE, 0,
                                    dest->size, src.data(), 0, nullptr,
                                    nullptr));
}

void OpenCLRuntime::readBuffer(std::vector<unsigned char>& dest,
                               std::shared_ptr<DeviceVariable> src) {
  assert(dest.size() == src->size);
  clErrorCheck(clEnqueueReadBuffer(queue, src->buffer(), CL_TRUE, 0,
                                   dest.size(), dest.data(), 0, nullptr,
                                   nullptr));
}

void OpenCLRuntime::createKernel(const std::string& name) {
  cl_int ret;
  kernels.emplace(name, clCreateKernel(program, name.c_str(), &ret));
  clErrorCheck(ret);
}

std::shared_ptr<DeviceVariable> OpenCLRuntime::allocateDeviceVar(
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
OpenCLRuntime::runLayers(std::shared_ptr<Layer> rootLayer) {
  using DeviceVariables = std::vector<std::shared_ptr<DeviceVariable>>;
  int ii = 0;

  std::shared_ptr<DeviceVariable> result;
  std::vector<unsigned char> rawResult;

  for (auto currentLayer = rootLayer; currentLayer != nullptr;
       currentLayer = currentLayer->child) {
    if (ii >= 1) break;

    const std::string layerName = currentLayer->name;

    // cout << "Run " << layerName << endl;

    if (layerName == "Reshape") {
      auto& data = currentLayer->inputs[0];
      auto& shape = currentLayer->inputs[1];
      auto& reshaped = currentLayer->outputs[0];

      // TODO: Read tensor from device memory
      auto shapeValue = shape->tensor->getDataAs<onnx::TensorProto::INT64>();
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
      // cout << input->toString() << endl;
      auto dv = allocateDeviceVar(input);
      dinputs.push_back(dv);
      if (dv->var->tensor) {
        writeBuffer(dv, dv->var->tensor->raw);
      } else {
        // It should be done to writeBuffer to dv.
      }

      clErrorCheck(
          clSetKernelArg(kernel, argCount++, sizeof(cl_mem), dv->bufferPtr()));
      clErrorCheck(
          clSetKernelArg(kernel, argCount++, sizeof(Shape), &dv->var->shape));
    }
    DeviceVariables doutputs;
    for (auto& output : currentLayer->outputs) {
      // cout << output->toString() << endl;
      auto dv = allocateDeviceVar(output);
      doutputs.push_back(dv);

      clErrorCheck(
          clSetKernelArg(kernel, argCount++, sizeof(cl_mem), dv->bufferPtr()));
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
      clErrorCheck(clGetEventProfilingInfo(handler, CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong), &start, NULL));
      clErrorCheck(clGetEventProfilingInfo(handler, CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong), &end, NULL));
      std::cout << "execution time: " << (end - start) / (1000 * 1000) << "(ms)"
                << endl;
    }
    clErrorCheck(clReleaseEvent(handler));

    result = doutputs[0];
    rawResult.resize(result->size, 0);
    readBuffer(rawResult, result);

    std::ofstream ofs("output.bin", std::ios::binary);
    for (auto v : rawResult) {
      ofs << v;
    }

    // printTensor(rawResult, result->var->elemType, result->var->shape.s);

    ii++;
  }

  return std::make_pair(result, rawResult);
}

std::string OpenCLRuntime::getBuildLog() {
  size_t logSize = 0;

  clErrorCheck(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0,
                                     nullptr, &logSize));

  std::vector<char> log(logSize + 1);
  clErrorCheck(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG,
                                     logSize, log.data(), nullptr));
  log[logSize] = '\0';

  return std::string(log.data(), log.size() - 1);
}

