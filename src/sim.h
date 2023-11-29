#pragma once

#include <memory>
#include <nanobind/ndarray.h>

/* Define size of the observation and action spaces. */
inline constexpr uint32_t kProgObservationSize = 16;
inline constexpr uint32_t kIOPairObservationSize = 6000;
inline constexpr uint32_t kActionSize = 31;

/* Define tensors and their shapes. */
using ProgObservationTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any, kProgObservationSize>
>;

using IOPairObservationTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any, kIOPairObservationSize>
>;

using ActionTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any, kActionSize>
>;

using RewardTensor = nanobind::ndarray<
    float, nanobind::shape<nanobind::any>
>;

/* Handles the simulation. Just for the sake of RL terminology,
 * each "world" is going to be a simulation where an agent tries to
 * generate the correct program */
struct SimManager {
    struct Impl;
    std::unique_ptr<Impl> impl;

    /* This is the path to the folder in which we will find a bunch of
     * IO pairs with file names io-pair-### */
    SimManager();
    ~SimManager();

    SimManager(uint32_t num_worlds);

    void reset();
    void step(ActionTensor action);

    /* These change throughout the rollout */
    ProgObservationTensor getProgObservations();

    /* These stay constant throughout a rollout */
    IOPairObservationTensor getIOPairObservations();

    RewardTensor getRewards();
};
