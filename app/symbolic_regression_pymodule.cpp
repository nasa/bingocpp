/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense> 

#include "bingocpp/explicit_regression.h"
#include "bingocpp/implicit_regression.h"


namespace py = pybind11;
using namespace bingo;

void add_regressor_classes(py::module &parent) {
  py::class_<ImplicitTrainingData>(parent, "ImplicitTrainingData")
    .def(py::init<Eigen::ArrayXXd &>())
    .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd &>())
    .def_readwrite("x", &ImplicitTrainingData::x)
    .def_readwrite("dx_dt", &ImplicitTrainingData::dx_dt)
    .def("__getitem__", 
         (ImplicitTrainingData *(ImplicitTrainingData::*)(int))
         &ImplicitTrainingData::GetItem)
    .def("__getitem__",
         (ImplicitTrainingData *(ImplicitTrainingData::*)(const std::vector<int>&))
         &ImplicitTrainingData::GetItem)
    .def("__len__", &ImplicitTrainingData::Size)
    .def("__getstate__", &ImplicitTrainingData::DumpState)
    .def("__setstate__", [](ImplicitTrainingData &td, const ImplicitTrainingDataState &state) {
            new (&td) ImplicitTrainingData(state); });
  
  py::class_<ExplicitTrainingData>(parent, "ExplicitTrainingData")
    .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd&>())
    .def_readwrite("x", &ExplicitTrainingData::x)
    .def_readwrite("y", &ExplicitTrainingData::y)
    .def("__getitem__", 
         (ExplicitTrainingData *(ExplicitTrainingData::*)(int))
         &ExplicitTrainingData::GetItem)
    .def("__getitem__",
         (ExplicitTrainingData *(ExplicitTrainingData::*)(const std::vector<int>&))
         &ExplicitTrainingData::GetItem)
    .def("__len__", &ExplicitTrainingData::Size)
    .def("__getstate__", &ExplicitTrainingData::DumpState)
    .def("__setstate__", [](ExplicitTrainingData &td, const ExplicitTrainingDataState &state) {
            new (&td) ExplicitTrainingData(state); });
  
  py::class_<ExplicitRegression>(parent, "ExplicitRegression")
    .def(py::init<ExplicitTrainingData *, std::string &, bool &>(),
        py::arg("training_data"),
        py::arg("metric")="mae",
        py::arg("relative")=false)
    .def_property("eval_count",
                  &ExplicitRegression::GetEvalCount,
                  &ExplicitRegression::SetEvalCount)
    .def("__call__", &ExplicitRegression::EvaluateIndividualFitness)
    .def("evaluate_fitness_vector", &ExplicitRegression::EvaluateFitnessVector)
    .def("__getstate__", &ExplicitRegression::DumpState)
    .def("__setstate__", [](ExplicitRegression &r, const ExplicitRegressionState &state) {
            new (&r) ExplicitRegression(state); });
  
  py::class_<ImplicitRegression>(parent, "ImplicitRegression")
    .def(py::init<ImplicitTrainingData *, int &, std::string &>(),
         py::arg("training_data"),
         py::arg("required_params") = -1,
         py::arg("metric") = "mae")
    .def_property("eval_count",
                  &ImplicitRegression::GetEvalCount,
                  &ImplicitRegression::SetEvalCount)
    .def("__call__", &ImplicitRegression::EvaluateIndividualFitness)
    .def("evaluate_fitness_vector", &ImplicitRegression::EvaluateFitnessVector)
    .def("__getstate__", &ImplicitRegression::DumpState)
    .def("__setstate__", [](ImplicitRegression &r, const ImplicitRegressionState &state) {
            new (&r) ImplicitRegression(state); });
}