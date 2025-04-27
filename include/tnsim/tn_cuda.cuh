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

// Error checking macros
#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if (err != cudaSuccess) \
  { printf("CUDA error %s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); fflush(stdout); std::abort(); } \
}

#define HANDLE_CUTN_ERROR(x) \
{ const auto err = x; \
  if (err != CUTENSORNET_STATUS_SUCCESS) \
  { printf("cuTensorNet error %s in %s:%d\n", cutensornetGetErrorString(err), __FILE__, __LINE__); fflush(stdout); std::abort(); } \
}

namespace NWQSim
{
    class TN_CUDA : public QuantumState
    {
    public:
        TN_CUDA(IdxType _n_qubits)
        : QuantumState(SimType::TN),
          n_qubits(_n_qubits),
        {
            HANDLE_CUDA_ERROR(cudaSetDevice(0));
            HANDLE_CUDA_ERROR(cutensornetCreate(&cutnHandle_));

            extents_.resize(n_qubits);
            extentsPtr_.resize(n_qubits);
            for (int i = 0; i < n_qubits; i++)
            {
                if (i == 0 || i = n_qubits - 1)
                    extents_[i] = {2, 2};
                else
                    extents_[i] = {2, 2, 2};
            }

            // scratch buffer
            size_t freeBytes, totalBytes;
            HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeBytes, &totalBytes));
            scratchSize_ = (freeBytes - (freeBytes % 4096)) / 2;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch_, scratchSize_));

            // workspace descriptor
            HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle_, &workDesc_));

            // create the initial quantum state

            std::vector<int64_t> qubtiDims(n_qubits, 2);
            HANDLE_CUTN_ERROR(cutensornetCreateState(
                cutnHandle_,
                CUTENSOR_STATE_PURITY_PURE,
                n_qubits,
                qubitDims.data(),
                CUDA_C_4F,
                &quantumState_));

            rng.seed(Config::RANDOM_SEED);
        }

        // Virtual destructor inherits from QuantumState
        ~TN_CUDA() override 
        {
            if (sampler_)
                HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler_));
            if (workDesc_)
                HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc_));
            if (quantumState_)
                HANDLE_CUTN_ERROR(cutensornetDestroyState(cutnHandle_, quantumState_));
                HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle_));

        HANDLE_CUDA_ERROR(cudaFree(d_scratch_));
        for (auto p : d_mpsTensor_)
            HANDLE_CUDA_ERROR(cudaFree(p));

        SAFE_FREE_HOST(results);
        }

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
        IdxType* results = NULL;
        std::mt19937                            rng;
        std::uniform_real_distribution<ValType> uni_dist;
    };

} // namespace NWQSim
