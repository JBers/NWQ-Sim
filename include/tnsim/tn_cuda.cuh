#pragma once

#include "../state.hpp"
#include "../nwq_util.hpp"
#include "../gate.hpp"
#include "../circuit.hpp"
#include "../config.hpp"
#include "private/exp_gate_declarations_host.hpp"
#include "../circuit_pass/fusion.hpp"
#include "../private/macros.hpp"
#include "../private/sim_gate.hpp"

#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace NWQSim
{
    class TN_CUDA : public QuantumState
    {
    public:
        TN_CUDA(IdxType _n_qubits)
        : QuantumState(SimType::TN),
          n_qubits(_n_qubits),
          results(nullptr),
          rng(),
        {
            rng.seed(Config::RANDOM_SEED);
        }

        // Virtual destructor inherits from QuantumState
        ~TN_CUDA() override = default;

        void reset_state() override
        {
            throw std::runtime_error("TN_CUDA::set_initial not implemented");
        }

        void set_seed(IdxType seed) override
        {
            rng.seed(seed);
        }

        void set_initial(std::string /*fpath*/, std::string /*format*/) override
        {
            throw std::runtime_error("TN_CUDA::set_initial not implemented");
        }

        void dump_res_state(std::string /*outpath*/) override
        {
            throw std::runtime_error("TN_CUDA::dump_res_state not implemented");
        }

        void sim(std::shared_ptr<NWQSim::Circuit> circuit) override
        {

        }

        IdxType* get_results() override
        {
            throw std::runtime_error("TN_CUDA::get_results not implemented");
        }

        IdxType measure(IdxType /*qubit*/) override
        {
            throw std::runtime_error("TN_CUDA::measure not implemented");
        }

        IdxType* measure_all(IdxType repetition) override
        {
            throw std::runtime_error("TN_CUDA::get_real not implemented");
        }

        // Override pure-virtual stubs from QuantumState
        ValType* get_real() const override
        {
            throw std::runtime_error("TN_CUDA::get_real not implemented");
        }

        ValType* get_imag() const override
        {
            throw std::runtime_error("TN_CUDA::get_imag not implemented");
        }

        ValType get_exp_z() override
        {
            throw std::runtime_error("TN_CUDA::get_exp_z() not implemented");
        }

        ValType get_exp_z(const std::vector<size_t>& /*in_bits*/) override
        {
            throw std::runtime_error("TN_CUDA::get_exp_z(bits) not implemented");
        }

        void print_res_state() override
        {
            throw std::runtime_error("TN_CUDA::print_res_state not implemented");
        }

    protected:
        IdxType n_qubits;
        IdxType n_cpu;
        IdxType* results;
        std::mt19937                            rng;
        std::uniform_real_distribution<ValType> uni_dist;
    };

} // namespace NWQSim
