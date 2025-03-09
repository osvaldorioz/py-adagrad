#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` adagrad.cpp -o adagrad`python3.12-config --extension-suffix`


namespace py = pybind11;

class Adagrad {
private:
    std::vector<double> weights;
    std::vector<double> grad_accum;
    double learning_rate;
    double epsilon;

public:
    Adagrad(size_t num_features, double lr = 0.01, double eps = 1e-8)
        : weights(num_features, 0.0), grad_accum(num_features, 0.0), learning_rate(lr), epsilon(eps) {}

    std::vector<double> get_weights() {
        return weights;
    }

    void update(py::array_t<double> gradients) {
        py::buffer_info buf = gradients.request();
        double* grad_ptr = static_cast<double*>(buf.ptr);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            grad_accum[i] += grad_ptr[i] * grad_ptr[i];
            weights[i] -= (learning_rate / (std::sqrt(grad_accum[i]) + epsilon)) * grad_ptr[i];
        }
    }
};

PYBIND11_MODULE(adagrad, m) {
    py::class_<Adagrad>(m, "Adagrad")
        .def(py::init<size_t, double, double>())
        .def("update", &Adagrad::update)
        .def("get_weights", &Adagrad::get_weights);
}
