#pragma once

#include <CL/cl.h>

#include <memory>

#include "Layer.h"
#include "Variable.h"

namespace minicnn {

class CLError {
 public:
  cl_int code;
  CLError(cl_int code) : code(code) {}
};

cl_int clErrorCheck(cl_int result);

// Device(GPU)-side
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
  OpenCLRuntime(const std::string& kernelPath);

  ~OpenCLRuntime();

  void writeBuffer(std::shared_ptr<DeviceVariable> dest,
                   const std::string& src);
  void readBuffer(std::vector<unsigned char>& dest,
                  std::shared_ptr<DeviceVariable> src);

  void createKernel(const std::string& name);

  std::shared_ptr<DeviceVariable> allocateDeviceVar(
      std::shared_ptr<Variable> var);

  std::pair<std::shared_ptr<DeviceVariable>, std::vector<unsigned char>>
  runLayers(std::shared_ptr<Layer> rootLayer);

  std::string getBuildLog();
};

}  // namespace minicnn
