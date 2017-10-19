#include <pybind11/pybind11.h>

double add(double i, double j) {
    return i + j;
}

PYBIND11_MODULE(bingocpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
