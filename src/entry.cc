#include "sim.h"

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(coder_model_sim, m) {
    nb::class_<SimManager>(m, "SimManager")
        .def("__init__", [](SimManager *self, 
                            uint32_t num_worlds) {
            new (self) SimManager(num_worlds);
        }, nb::arg("num_worlds"))
        .def("reset", &SimManager::reset)
        .def("step", &SimManager::step, nb::arg("action_tensor"))
        .def("get_prog_observations", &SimManager::getProgObservations)
        .def("get_io_pair_observations", &SimManager::getIOPairObservations)
        .def("get_rewards", &SimManager::getRewards);

    m.attr("prog_observation_size") = kProgObservationSize;
    m.attr("io_pair_observation_size") = kIOPairObservationSize;
    m.attr("action_size") = kActionSize;
}
