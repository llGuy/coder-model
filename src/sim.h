#pragma once

#include <memory>
#include <nanobind/ndarray.h>

/* Define size of the observation and action spaces. */
inline constexpr uint32_t kObservationSize = 6016;
inline constexpr uint32_t kActionSize = 31;

/* Define tensors and their shapes. */
using ObservationTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any, kObservationSize>
>;

using ActionTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any, kActionSize>
>;

using RewardTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any>
>;

/* Handles the simulation */
struct SimManager {
    struct Impl;
    std::unique_ptr<Impl> impl;

    SimManager(const char *path_to_data);
    ~SimManager();

    void step(ActionTensor action);
    ObservationTensor getObservations();
};
