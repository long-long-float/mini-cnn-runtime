#pragma once

namespace minicnn {

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

}  // namespace minicnn
