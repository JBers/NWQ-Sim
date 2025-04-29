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
            int64_t bond_dim = 10;
            
            for (int i = 0; i < n_qubits; ++i)
            {
                if (i == 0)
                    extents_[i] = {2, bond_dim};
                else if (i == n_qubits - 1)
                    extents_[i] = {bond_dim, 2};
                else
                    extents_[i] = {bond_dim, 2, bond_dim};
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
            int64_t bond_dim = 10;
            
            for (int i = 0; i < n_qubits; ++i)
            {
                if (i == 0)
                    extents_[i] = {2, bond_dim};
                else if (i == n_qubits - 1)
                    extents_[i] = {bond_dim, 2};
                else
                    extents_[i] = {bond_dim, 2, bond_dim};
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
        
            // allocate (or reuse) device buffer for any gate (16 complex entries)
            static void* d_gate_mat = nullptr;
            if (!d_gate_mat) {
                HANDLE_CUDA_ERROR(cudaMalloc(
                    &d_gate_mat,
                    16 * sizeof(std::complex<ValType>)
                ));
            }
        
            // fuse the circuit into a list of tensor-apply gates
            auto gates = fuse_circuit_sv(circuit);
        
            // prepare persistent mode-index arrays
            int32_t oneQubitMode[1];
            int32_t twoQubitModes[2];
        
            // apply each gate
            IdxType repetitions = 0;
        
            // prepare explicit column-major strides for 1- and 2-mode operators
            int64_t strides1[2] = { 2, 1 };
            int64_t strides2[4] = { 8, 4, 2, 1 };
        
            for (size_t idx = 0; idx < gates.size(); ++idx) {
                auto const &g = gates[idx];
        
                if (g.op_name == OP::C1) {
                    // reconstruct the 2x2 SVGate in row-major
                    std::complex<ValType> sv_mat[4];
                    for (int k = 0; k < 4; ++k) {
                        sv_mat[k] = { g.gm_real[k], g.gm_imag[k] };
                    }
                    // reorder to column-major [out, in]
                    std::complex<ValType> gate_matrix[4];
                    for (size_t i0 = 0; i0 < 2; ++i0) {
                        for (size_t j0 = 0; j0 < 2; ++j0) {
                            size_t orig = i0 * 2 + j0;
                            size_t off  = j0 * 2 + i0;
                            gate_matrix[off] = sv_mat[orig];
                        }
                    }
                    // upload and apply
                    HANDLE_CUDA_ERROR(cudaMemcpy(
                        d_gate_mat,
                        gate_matrix,
                        4 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice
                    ));
                    oneQubitMode[0] = static_cast<int32_t>(g.qubit);
                    int64_t tensorId1 = -1;
                    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
                        cutnHandle_, quantumState_,
                        1,                  // number of modes
                        oneQubitMode,       // modes array
                        d_gate_mat,         // matrix
                        nullptr,            // explicit column-major strides
                        1,                  // immutable
                        0,                  // adjoint
                        1,                  // unitary
                        &tensorId1          // output tensor identifier
                    ));
                }
                else if (g.op_name == OP::C2) {
                    // reconstruct the 4x4 SVGate in row-major
                    std::complex<ValType> sv_mat[16];
                    for (int k = 0; k < 16; ++k) {
                        sv_mat[k] = { g.gm_real[k], g.gm_imag[k] };
                    }
                    // reorder to column-major [ctrl_out, tgt_out, ctrl_in, tgt_in]
                    std::complex<ValType> gate_matrix[16];
                    for (int in_ctrl = 0; in_ctrl < 2; ++in_ctrl) {
                        for (int in_tgt = 0; in_tgt < 2; ++in_tgt) {
                            for (int out_ctrl = 0; out_ctrl < 2; ++out_ctrl) {
                                for (int out_tgt = 0; out_tgt < 2; ++out_tgt) {
                                    size_t row  = in_ctrl * 2 + in_tgt;
                                    size_t col  = out_ctrl * 2 + out_tgt;
                                    size_t orig = row * 4 + col;
                                    size_t off  = out_ctrl
                                               + out_tgt * 2
                                               + in_ctrl * 4
                                               + in_tgt * 8;
                                    gate_matrix[off] = sv_mat[orig];
                                }
                            }
                        }
                    }
                    // upload and apply
                    HANDLE_CUDA_ERROR(cudaMemcpy(
                        d_gate_mat,
                        gate_matrix,
                        16 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice
                    ));
                    twoQubitModes[0] = static_cast<int32_t>(g.ctrl);
                    twoQubitModes[1] = static_cast<int32_t>(g.qubit);
                    int64_t tensorId2 = -1;
                    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
                        cutnHandle_, quantumState_,
                        2,                  // number of modes
                        twoQubitModes,      // modes array
                        d_gate_mat,         // matrix
                        nullptr,            // explicit column-major strides
                        1,                  // immutable
                        0,                  // adjoint
                        1,                  // unitary
                        &tensorId2          // output tensor identifier
                    ));
                }
                else if (g.op_name == OP::MA) {
                    repetitions = g.qubit;
                }
                else {
                    continue;
                }
        
                // synchronize device
                auto syncErr = cudaDeviceSynchronize();
                if (syncErr != cudaSuccess) {
                    fprintf(
                        stderr,
                        "CUDA error after gate %zu: %s\n",
                        idx,
                        cudaGetErrorString(syncErr)
                    );
                    std::abort();
                }
            }
        
            // finalize MPS factorization
            for (int i = 0; i < n_qubits; ++i) {
                extentsPtr_[i] = extents_[i].data();
            }
            HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
                cutnHandle_, quantumState_,
                CUTENSORNET_BOUNDARY_CONDITION_OPEN,
                extentsPtr_.data(),
                nullptr
            ));
        
            // configure the SVD algorithm
            cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
            HANDLE_CUTN_ERROR(cutensornetStateConfigure(
                cutnHandle_, quantumState_,
                CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
                &algo, sizeof(algo)
            ));
        
            // prepare for MPS computation
            HANDLE_CUTN_ERROR(cutensornetStatePrepare(
                cutnHandle_, quantumState_,
                scratchSize_, workDesc_, 0x0
            ));
        
            // allocate and set workspace
            int64_t reqSize = 0;
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
                cutnHandle_, workDesc_,
                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                &reqSize
            ));
            HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
                cutnHandle_, workDesc_,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                d_scratch_, reqSize
            ));
        
            // allocate MPS tensor buffers
            d_mpsTensor_.resize(n_qubits);
            for (int i = 0; i < n_qubits; ++i) {
                int64_t elems = 1;
                for (auto e : extents_[i]) {
                    elems *= e;
                }
                HANDLE_CUDA_ERROR(cudaMalloc(
                    &d_mpsTensor_[i],
                    elems * sizeof(std::complex<double>)
                ));
            }
        
            // execute MPS computation
            HANDLE_CUTN_ERROR(cutensornetStateCompute(
                cutnHandle_, quantumState_,
                workDesc_,
                extentsPtr_.data(), nullptr,
                d_mpsTensor_.data(), 0
            ));
        
            measure_all(repetitions);
        }


        IdxType* get_results() override
        {
            return results;
        }

        IdxType measure(IdxType qubit) override
        {
            throw std::runtime_error("TN_CUDA::measure not implemented");
        }

        IdxType* measure_all(IdxType repetition) override
        {
            // allocate raw bit buffer of size repetition * n_qubits
            IdxType* bitbuf = nullptr;
            SAFE_ALOC_HOST(bitbuf, sizeof(IdxType) * repetition * n_qubits);
        
            // create or reuse sampler
            HANDLE_CUTN_ERROR(cutensornetCreateSampler(
                cutnHandle_, quantumState_,
                n_qubits, nullptr,
                &sampler_));
        
            // configure number of hyper samples
            int32_t numHyper = 1024;
            HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
                cutnHandle_, sampler_,
                CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES,
                &numHyper, sizeof(numHyper)));
        
            // prepare sampler
            HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(
                cutnHandle_, sampler_,
                scratchSize_, workDesc_, 0));
        
            // attach sampler workspace
            int64_t samplerWorkSize = 0;
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
                cutnHandle_, workDesc_,
                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                &samplerWorkSize));
            HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
                cutnHandle_, workDesc_,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                d_scratch_, samplerWorkSize));
        
            // sample raw bits
            HANDLE_CUTN_ERROR(cutensornetSamplerSample(
                cutnHandle_, sampler_,
                repetition,
                workDesc_,
                reinterpret_cast<int64_t*>(bitbuf),
                0));
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        
            // allocate results array
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);
        
            // pack bits into single integers
            for (IdxType s = 0; s < repetition; ++s) {
                IdxType idx = 0;
                for (IdxType q = 0; q < n_qubits; ++q) {
                    auto b = bitbuf[s * n_qubits + q];
                    idx |= (b << (n_qubits - 1 - q));
                }
                results[s] = idx;
            }
        
            // free bit buffer
            SAFE_FREE_HOST(bitbuf);
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
