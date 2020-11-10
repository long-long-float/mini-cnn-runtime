#include "Layer.h"

using namespace minicnn;

std::string Layer::toString() const {
  std::stringstream ss;
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
