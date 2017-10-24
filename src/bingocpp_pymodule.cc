/*!
 * \file bingocpp_pymodule.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the python bindings of the BingoCpp library.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "BingoCpp/acyclic_graph.h"

double add(double i, double j) {
  return i + j;
}

PYBIND11_MODULE(bingocpp, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring
  m.def("evauluate", &Evaluate, "evaluate");
  m.def("simplify_and_evauluate", &SimplifyAndEvaluate,
        "evaluate after simplification");
  m.def("evauluate_with_derivative", &EvaluateWithDerivative,
        "evaluate with derivative");
  m.def("simplify_and_evauluate_with_derivative",
        &SimplifyAndEvaluateWithDerivative,
        "evaluate with derivative after simplification");
}



