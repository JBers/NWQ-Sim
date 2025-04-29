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
#include <cstdio>

#include <cutensornet.h>
#include <cuda_runtime.h>

// Error checking macros
#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if (err != cudaSuccess) \
  { printf("CUDA error %s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); fflush(stdout); std::abort(); } \
}

#define HANDLE_CUTN_ERROR(x) \
{ const auto err = x; \
  if (err != CUTENSORNET_STATUS_SUCCESS) { \
    printf("cuTensorNet error %s in %s:%d\n", \
           cutensornetGetErrorString(err), __FILE__, __LINE__); \
    fflush(stdout); std::abort(); \
  } \
}

// API Calls for cutensornet can be found here:
// https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/functions.html

namespace NWQSim
{
    class TN_CUDA : public QuantumState
    {
    public:
        TN_CUDA(IdxType _n_qubits)
        : QuantumState(SimType::TN),
          n_qubits(_n_qubits)
        {
            HANDLE_CUDA_ERROR(cudaSetDevice(0));
            HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle_));

            extents_.resize(n_qubits);
            extentsPtr_.resize(n_qubits);
            for (int i = 0; i < n_qubits; i++)
            {
                if (i == 0 || i == n_qubits - 1)
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

            std::vector<int64_t> qubitDims(n_qubits, 2);
            HANDLE_CUTN_ERROR(cutensornetCreateState(
                cutnHandle_,
                CUTENSORNET_STATE_PURITY_PURE,
                n_qubits,
                qubitDims.data(),
                CUDA_C_64F,
                &quantumState_));
        }

        // Virtual destructor inherits from QuantumState
        ~TN_CUDA() override 
        {
            if (sampler_)
                HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler_));
            if (workDesc_)
                HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc_));
            if (quantumState_)
            {
                HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState_));
                HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle_));
            }

        HANDLE_CUDA_ERROR(cudaFree(d_scratch_));
        for (auto p : d_mpsTensor_)
            HANDLE_CUDA_ERROR(cudaFree(p));

        SAFE_FREE_HOST(results);
        }

        void reset_state() override
        {
            extents_.resize(n_qubits);
            extentsPtr_.resize(n_qubits);
            for (int i = 0; i < n_qubits; i++)
            {
                if (i == 0 || i == n_qubits - 1)
                    extents_[i] = {2, 2};
                else
                    extents_[i] = {2, 2, 2};
            }

            std::vector<int64_t> qubitDims(n_qubits, 2);
            HANDLE_CUTN_ERROR(cutensornetCreateState(
                cutnHandle_,
                CUTENSORNET_STATE_PURITY_PURE,
                n_qubits,
                qubitDims.data(),
                CUDA_C_64F,
                &quantumState_));

        }

        void set_seed(IdxType seed) override
        {
            throw std::runtime_error("TN_CUDA does not use RNG seed, not accessible form cutensornet API");
        }

        void set_initial(std::string fpath, std::string format) override
        {
            std::cout << "set function was called" << std::endl;
        }

        void dump_res_state(std::string outpath) override
        {
            std::cout << "dump function was called" << std::endl;
        }

        void sim(std::shared_ptr<NWQSim::Circuit> circuit) override
        {
            assert(circuit->num_qubits() == n_qubits);
        // one static device buffer for all 2-qubit gates
        static void* d_gate_mat = nullptr;
            auto gates = fuse_circuit_sv(circuit);
            for (auto const& g : gates)
            {
                if (g.op_name == OP::C1)
                {
                    //pull the real and imaginary components of the sv-gate
                    const ValType* gm_real = g.gm_real;
                    const ValType* gm_imag = g.gm_imag;

                    // create tensor gate data
                    std::vector<std::complex<ValType>> gate_matrix(4);

        		    static void* d_gate_mat = nullptr;
        
        		    if (!d_gate_mat) {
            			cudaMalloc(&d_gate_mat, 4 * sizeof(std::complex<ValType>));
        		    }

                    for (int i = 0; i < 4; ++i)
                    {
                        gate_matrix[i] = std::complex<ValType>(gm_real[i], gm_imag[i]);
                    }

        		    cudaMemcpy(d_gate_mat,
                        gate_matrix.data(),
                        4 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice);

                    int32_t state_modes[1] = {static_cast<int32_t>(g.qubit)};

                    int64_t tensor_mode_strides[2] = {1, 2};

                    printf("Got to right before tensor code in 1 qubit gate");
                    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
                        cutnHandle_, quantumState_,
                        1, state_modes,
                        d_gate_mat, tensor_mode_strides,
                        1, 0, 1, nullptr));
                }
                else if (g.op_name == OP::C2)
                {
                    //pull the real and imaginary components of the sv-gate
                    const ValType* gm_real = g.gm_real;
                    const ValType* gm_imag = g.gm_imag;

                    // create tensor gate data
                    std::vector<std::complex<ValType>> gate_matrix(16);

        		    static void* d_gate_mat = nullptr;
        
        		    if (!d_gate_mat) {
            			cudaMalloc(&d_gate_mat, 16 * sizeof(std::complex<ValType>));
        		    }

                    for (int i = 0; i < 16; ++i)
                    {
                        gate_matrix[i] = std::complex<ValType>(gm_real[i], gm_imag[i]);
                    }

        		    cudaMemcpy(d_gate_mat,
                        gate_matrix.data(),
                        16 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice);

                    int32_t state_modes[2] = {static_cast<int32_t>(g.ctrl), static_cast<int32_t>(g.qubit)};

                    int64_t tensor_mode_strides[4] = {1, 2, 4, 8};

                    // debug‐print exactly what we’ll hand to cuTensorNet
                    fprintf(stderr,
                            "DEBUG applyTensorOperator: gate_ptr=%p, modes=[%d,%d], strides=[%lld,%lld,%lld,%lld]\n",
                            d_gate_mat,
                            state_modes[0], state_modes[1],
                            tensor_mode_strides[0], tensor_mode_strides[1],
                            tensor_mode_strides[2], tensor_mode_strides[3]);

                    // inline call + error‐check
                    {
                      auto _st = cutensornetStateApplyTensorOperator(
                          cutnHandle_, quantumState_,
                          2, state_modes,
                          d_gate_mat,
                          tensor_mode_strides,
                          1, 0, 1, nullptr
                      );
                      if (_st != CUTENSORNET_STATUS_SUCCESS) {
                        fprintf(stderr,
                                "ERROR applyTensorOperator: %s\n",
                                cutensornetGetErrorString(_st));
                        std::abort();
                      }
                    }
                }
            }

            // finalize MPS
            HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
                cutnHandle_, quantumState_,
                CUTENSORNET_BOUNDARY_CONDITION_OPEN,
                extentsPtr_.data(), /*strides=*/nullptr));

            // setup SVD
            cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
            HANDLE_CUTN_ERROR(cutensornetStateConfigure(
                cutnHandle_, quantumState_,
                CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
                &algo, sizeof(algo)));

            // prepare factorizatoin
            HANDLE_CUTN_ERROR(cutensornetStatePrepare(
                cutnHandle_, quantumState_,
                scratchSize_, workDesc_, 0x0));

            // workspace memory
            int64_t reqSize = 0;
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
                cutnHandle_, workDesc_,
                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                &reqSize));
            HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
                cutnHandle_, workDesc_,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                d_scratch_, reqSize));

            // set MPS tensor buffers
            d_mpsTensor_.resize(n_qubits);
            for (int i = 0; i < n_qubits; ++i)
            {
                int64_t elems = 1;
                for (auto e : extents_[i])
                    elems *= e;
                
                HANDLE_CUDA_ERROR(cudaMalloc(
                    &d_mpsTensor_[i],
                    elems * sizeof(std::complex<double>)));
            }

            // compute MPS
            HANDLE_CUTN_ERROR(cutensornetStateCompute(
                cutnHandle_, quantumState_,
                workDesc_,
                extentsPtr_.data(), nullptr,
                d_mpsTensor_.data(), 0));
        }

        IdxType* get_results() override
        {
            throw std::runtime_error("TN_CUDA::get_results not implemented");
        }

        IdxType measure(IdxType qubit) override
        {
            throw std::runtime_error("TN_CUDA::measure not implemented");
        }

        IdxType* measure_all(IdxType repetition) override
        {
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);

            // create and configure the sampler
            HANDLE_CUTN_ERROR(cutensornetCreateSampler(
                cutnHandle_, quantumState_,
                n_qubits, nullptr,
                &sampler_));

            int32_t numHyper = 8;
            HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
                cutnHandle_, sampler_,
                CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES,
                &numHyper, sizeof(numHyper)));

            // prepare and sample
            HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(
                cutnHandle_, sampler_,
                scratchSize_, workDesc_, 0x0));
            HANDLE_CUTN_ERROR(cutensornetSamplerSample(
                cutnHandle_, sampler_,
                repetition,
                workDesc_,
                reinterpret_cast<int64_t*>(results),
                0));

            return results;
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

        ValType get_exp_z(const std::vector<size_t>& in_bits) override
        {
            throw std::runtime_error("TN_CUDA::get_exp_z(bits) not implemented");
        }

        void print_res_state() override
        {
            throw std::runtime_error("TN_CUDA::print_res_state not implemented");
        }

    protected:
        IdxType n_qubits;
        IdxType* results = NULL;

        cutensornetHandle_t cutnHandle_{};
        cutensornetState_t quantumState_{};
        cutensornetWorkspaceDescriptor_t workDesc_{};
        cutensornetStateSampler_t sampler_{};

        std::vector<std::vector<int64_t>> extents_;
        std::vector<int64_t*> extentsPtr_;
        std::vector<void*> d_mpsTensor_;

        void* d_scratch_{nullptr};
        size_t scratchSize_{0};
    };

} // namespace NWQSim
