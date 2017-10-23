#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "BingoCpp/acyclic_graph.hh"

double add(double i, double j) {
  return i + j;
}

PYBIND11_MODULE(bingocpp, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring
  m.def("evauluate", &Evaluate, "evaluate");
}

