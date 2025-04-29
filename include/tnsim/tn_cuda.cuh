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
//            for (IdxType i = 0; i < n_qubits; ++i) {
//                extentsPtr_[i] = extents_[i].data();
//            }
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

//            for (IdxType i = 0; i < n_qubits; ++i) {
//                extentsPtr_[i] = extents_[i].data();
//            }

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
            printf("[DEBUG] TN_CUDA::sim start, n_qubits=%lld\n", (long long)n_qubits);
        
            // 1) Allocate (or reuse) device buffer for any gate (16 complex entries)
            static void* d_gate_mat = nullptr;
            if (!d_gate_mat) {
                printf("[DEBUG] Allocating d_gate_mat (16 complex entries)\n");
                HANDLE_CUDA_ERROR(cudaMalloc(&d_gate_mat,
                                             16 * sizeof(std::complex<ValType>)));
                printf("[DEBUG] d_gate_mat = %p\n", d_gate_mat);
            } else {
                printf("[DEBUG] Reusing existing d_gate_mat = %p\n", d_gate_mat);
            }
        
            // 2) Print scratch buffer info
            printf("[DEBUG] Scratch buffer at %p, size = %zu\n",
                   d_scratch_, scratchSize_);
        
            // 3) Fuse the circuit into a list of tensor‐apply gates
            auto gates = fuse_circuit_sv(circuit);
            printf("[DEBUG] fuse_circuit_sv returned %zu gates\n", gates.size());
        
            // 4) Prepare persistent mode‐index arrays
            int32_t oneQubitMode[1];
            int32_t twoQubitModes[2];
        
            // 5) Apply each gate with detailed debug

            IdxType repetitions = 0;


            // helper to print a 2×2 matrix in row-major
            auto print2x2 = [&](const std::complex<ValType>* M, const char* label) {
                printf("%s (2×2 row-major):\n", label);
                for(int r = 0; r < 2; ++r) {
                    for(int c = 0; c < 2; ++c) {
                        auto v = M[r*2 + c];
                        printf(" (% .4f%+.4fi)", v.real(), v.imag());
                    }
                    printf("\n");
                }
            };
        
            // helper to print a 4×4 matrix in row-major
            auto print4x4 = [&](const std::complex<ValType>* M, const char* label) {
                printf("%s (4×4 row-major):\n", label);
                for(int r = 0; r < 4; ++r) {
                    for(int c = 0; c < 4; ++c) {
                        auto v = M[r*4 + c];
                        printf(" (% .4f%+.4fi)", v.real(), v.imag());
                    }
                    printf("\n");
                }
            };

            
            // Prepare explicit column‐major strides for 1- and 2-mode operators
            int64_t strides1[2] = { 2, 1 };        // for a 2×2 matrix: out_stride=1, in_stride=2
            int64_t strides2[4] = { 8, 4, 2, 1 };  // for a 2×2×2×2 tensor: {o0,o1,i0,i1}-strides
            
            for (size_t idx = 0; idx < gates.size(); ++idx) {
                auto const &g = gates[idx];
                printf("[DEBUG] Gate %zu: op_name=%d, ctrl=%d, tgt=%d\n",
                       idx, (int)g.op_name, (int)g.ctrl, (int)g.qubit);
            
                if (g.op_name == OP::C1) {
                    // 1) Reconstruct the 2×2 SVGate in row-major
                    std::complex<ValType> sv_mat[4];
                    for (int k = 0; k < 4; ++k) {
                        sv_mat[k] = { g.gm_real[k], g.gm_imag[k] };
                    }
                    print2x2(sv_mat, "[DEBUG] SVGate matrix");
            
                    // 2) Reorder to column-major [out, in]
                    std::complex<ValType> gate_matrix[4];
                    for (size_t i0 = 0; i0 < 2; ++i0) {
                        for (size_t j0 = 0; j0 < 2; ++j0) {
                            size_t orig = i0*2 + j0;    // row-major index
                            size_t off  = j0*2 + i0;    // column-major index
                            gate_matrix[off] = sv_mat[orig];
                        }
                    }
                    print2x2(gate_matrix, "[DEBUG] Reordered tensor gate");
            
                    // 3) Upload & apply
                    HANDLE_CUDA_ERROR(cudaMemcpy(
                        d_gate_mat,
                        gate_matrix,
                        4 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice
                    ));
                    oneQubitMode[0] = static_cast<int32_t>(g.qubit);
                    printf("[DEBUG] Applying C1 on mode [%d]\n", oneQubitMode[0]);

                    int64_t tensorId1 = -1;

                    auto status1 = cutensornetStateApplyTensorOperator(
                        cutnHandle_, quantumState_,
                        1,                       // nModes
                        oneQubitMode,            // modes
                        d_gate_mat,              // matrix
                        nullptr,                // explicit column-major strides
                        /*immutable=*/1,
                        /*adjoint=*/0,
                        /*unitary=*/1,
                        /*tensorId=*/&tensorId1
                    );
                    printf("[DEBUG] C1 apply returned %d\n", (int)status1);
                    HANDLE_CUTN_ERROR(status1);
                }
                else if (g.op_name == OP::C2) {
                    // 1) Reconstruct the 4×4 SVGate in row-major
                    std::complex<ValType> sv_mat[16];
                    for (int k = 0; k < 16; ++k) {
                        sv_mat[k] = { g.gm_real[k], g.gm_imag[k] };
                    }
                    print4x4(sv_mat, "[DEBUG] SVGate matrix");
            
                    // 2) Reorder to column-major [ctrl_out, tgt_out, ctrl_in, tgt_in]
                    std::complex<ValType> gate_matrix[16];
                    for (int in_ctrl = 0; in_ctrl < 2; ++in_ctrl) {
                        for (int in_tgt = 0; in_tgt < 2; ++in_tgt) {
                            for (int out_ctrl = 0; out_ctrl < 2; ++out_ctrl) {
                                for (int out_tgt = 0; out_tgt < 2; ++out_tgt) {
                                    size_t row  = in_ctrl*2 + in_tgt;
                                    size_t col  = out_ctrl*2 + out_tgt;
                                    size_t orig = row*4 + col;                 // row-major
                                    size_t off  = out_ctrl                    // column-major:
                                               + out_tgt*2
                                               + in_ctrl*4
                                               + in_tgt*8;
                                    gate_matrix[off] = sv_mat[orig];
                                }
                            }
                        }
                    }
                    print4x4(gate_matrix, "[DEBUG] Reordered tensor gate");
            
                    // 3) Upload & apply
                    HANDLE_CUDA_ERROR(cudaMemcpy(
                        d_gate_mat,
                        gate_matrix,
                        16 * sizeof(std::complex<ValType>),
                        cudaMemcpyHostToDevice
                    ));
                    twoQubitModes[0] = static_cast<int32_t>(g.ctrl);
                    twoQubitModes[1] = static_cast<int32_t>(g.qubit);
                    printf("[DEBUG] Applying C2 on modes [%d,%d]\n",
                           twoQubitModes[0], twoQubitModes[1]);
                    int64_t tensorId2 = -1;
                    auto status2 = cutensornetStateApplyTensorOperator(
                        cutnHandle_, quantumState_,
                        2,                       // nModes
                        twoQubitModes,           // modes
                        d_gate_mat,              // matrix
                        nullptr,                // explicit column-major strides
                        /*immutable=*/1,
                        /*adjoint=*/0,
                        /*unitary=*/1,
                        &tensorId2
                    );
                    printf("[DEBUG] C2 apply returned %d (tensorId=%lld)\n",
                           (int)status2, (long long)tensorId2);
                    HANDLE_CUTN_ERROR(status2);
                }
                else if (g.op_name == OP::MA) {
                    repetitions = g.qubit;
                }
                else {
                    printf("[DEBUG] Skipping unknown op_name=%d\n", (int)g.op_name);
                    continue;
                }
            
                // 4) Synchronize
                printf("[DEBUG] cudaDeviceSynchronize after gate %zu\n", idx);
                auto syncErr = cudaDeviceSynchronize();
                if (syncErr != cudaSuccess) {
                    fprintf(stderr,
                            "CUDA error after gate %zu: %s\n",
                            idx,
                            cudaGetErrorString(syncErr));
                    std::abort();
                }
            }
        
            // 7) Finalize MPS factorization
            printf("[DEBUG] Setting up extentsPtr_ for MPS finalize\n");
            for (int i = 0; i < n_qubits; ++i) {
                extentsPtr_[i] = extents_[i].data();
                printf("[DEBUG]  extents[%d] = {", i);
                for (size_t d = 0; d < extents_[i].size(); ++d) {
                    printf("%lld%s", (long long)extents_[i][d],
                           d+1 < extents_[i].size() ? ", " : "");
                }
                printf("}\n");
            }
            printf("[DEBUG] Calling cutensornetStateFinalizeMPS\n");
            HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
                cutnHandle_, quantumState_,
                CUTENSORNET_BOUNDARY_CONDITION_OPEN,
                extentsPtr_.data(), /*strides=*/nullptr
            ));
            printf("[DEBUG] cutensornetStateFinalizeMPS completed\n");
        
            // 8) Configure the SVD algorithm
            printf("[DEBUG] Configuring MPS SVD algorithm\n");
            cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
            HANDLE_CUTN_ERROR(cutensornetStateConfigure(
                cutnHandle_, quantumState_,
                CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
                &algo, sizeof(algo)
            ));
        
            // 9) Prepare for MPS computation
            printf("[DEBUG] Preparing MPS computation\n");
            HANDLE_CUTN_ERROR(cutensornetStatePrepare(
                cutnHandle_, quantumState_,
                scratchSize_, workDesc_, /*flags=*/0x0
            ));
        
            // 10) Allocate and set workspace
            printf("[DEBUG] Querying workspace size\n");
            int64_t reqSize = 0;
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
                cutnHandle_, workDesc_,
                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                &reqSize
            ));
            printf("[DEBUG] Required workspace size = %lld bytes\n", (long long)reqSize);
            printf("[DEBUG] Attaching workspace buffer at %p\n", d_scratch_);
            HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
                cutnHandle_, workDesc_,
                CUTENSORNET_MEMSPACE_DEVICE,
                CUTENSORNET_WORKSPACE_SCRATCH,
                d_scratch_, reqSize
            ));
            printf("[DEBUG] Workspace buffer set\n");
        
            // 11) Allocate MPS tensor buffers
            printf("[DEBUG] Allocating MPS tensor buffers\n");
            d_mpsTensor_.resize(n_qubits);
            for (int i = 0; i < n_qubits; ++i) {
                int64_t elems = 1;
                for (auto e : extents_[i]) elems *= e;
                HANDLE_CUDA_ERROR(cudaMalloc(
                    &d_mpsTensor_[i],
                    elems * sizeof(std::complex<double>)
                ));
                printf("[DEBUG]  d_mpsTensor_[%d] = %p, elements = %lld\n",
                       i, d_mpsTensor_[i], (long long)elems);
            }
        
            // 12) Execute MPS computation
            printf("[DEBUG] Executing cutensornetStateCompute\n");
            HANDLE_CUTN_ERROR(cutensornetStateCompute(
                cutnHandle_, quantumState_,
                workDesc_,
                extentsPtr_.data(), /*strides=*/nullptr,
                d_mpsTensor_.data(), /*flags=*/0
            ));
            printf("[DEBUG] cutensornetStateCompute completed\n");

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
            // 1) Allocate raw‐bit buffer of size (shots × n_qubits)
            IdxType* bitbuf = nullptr;
            SAFE_ALOC_HOST(bitbuf, sizeof(IdxType) * repetition * n_qubits);
        
            // 2) Create (or reuse) the sampler
            printf("[DEBUG] cutensornetCreateSampler(cutnHandle_=%p, quantumState_=%p, n_qubits=%lld)\n",
                   (void*)cutnHandle_, (void*)quantumState_, (long long)n_qubits);
            HANDLE_CUTN_ERROR(
                cutensornetCreateSampler(
                    cutnHandle_, quantumState_,
                    n_qubits, nullptr,
                    &sampler_));
            // 3) Configure hyper‐samples (and optionally determinism)
            int32_t numHyper = 1024;
            HANDLE_CUTN_ERROR(
                cutensornetSamplerConfigure(
                    cutnHandle_, sampler_,
                    CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES,
                    &numHyper, sizeof(numHyper)));
            // 4) Prepare the sampler
            HANDLE_CUTN_ERROR(
                cutensornetSamplerPrepare(
                    cutnHandle_, sampler_,
                    scratchSize_, workDesc_, /*flags=*/0));
            // 5) Attach sampler workspace
            int64_t samplerWorkSize = 0;
            HANDLE_CUTN_ERROR(
                cutensornetWorkspaceGetMemorySize(
                    cutnHandle_, workDesc_,
                    CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                    CUTENSORNET_MEMSPACE_DEVICE,
                    CUTENSORNET_WORKSPACE_SCRATCH,
                    &samplerWorkSize));
            HANDLE_CUTN_ERROR(
                cutensornetWorkspaceSetMemory(
                    cutnHandle_, workDesc_,
                    CUTENSORNET_MEMSPACE_DEVICE,
                    CUTENSORNET_WORKSPACE_SCRATCH,
                    d_scratch_, samplerWorkSize));
        
            // 6) Sample raw bits
            printf("[DEBUG] Sampling (%lld samples)\n", (long long)repetition);
            HANDLE_CUTN_ERROR(
                cutensornetSamplerSample(
                    cutnHandle_, sampler_,
                    repetition,
                    workDesc_,
                    reinterpret_cast<int64_t*>(bitbuf),
                    /*flags=*/0));
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        
            // 7) Allocate packed‐results array (one integer per shot)
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);
        
            // 8) Pack each group of n_qubits bits into a single integer
            for (IdxType s = 0; s < repetition; ++s) {
                IdxType idx = 0;
                // Little-endian: qubit 0 → LSB. For big-endian, shift by (n_qubits - 1 - q).
                for (IdxType q = 0; q < n_qubits; ++q) {
                    auto b = bitbuf[s * n_qubits + q];
                    idx |= (b << (n_qubits - 1 - q));
                }
                results[s] = idx;
            }
        
            // 9) Debug: print first few packed results
            IdxType inspect = std::min<IdxType>(repetition, (IdxType)4);
            printf("[DEBUG] First %lld packed results:\n", (long long)inspect);
            for (IdxType s = 0; s < inspect; ++s) {
                printf("  shot[%2lld] = %lld\n",
                       (long long)s, (long long)results[s]);
            }
        
            // 10) Clean up and return
            SAFE_FREE_HOST(bitbuf);
            printf("[DEBUG] measure_all end, returning results=%p\n", (void*)results);
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
