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

#include "itensor/all.h"

namespace NWQSim
{
    class TN_CUDA : public QuantumState
    {
    public:
        TN_CUDA(IdxType _n_qubits)
        : QuantumState(SimType::TN),
          n_qubits(_n_qubits),
          n_cpu(1),
          results(nullptr),
          rng(),
          uni_dist(0.0, 1.0),
          sites_(itensor::SpinHalf(_n_qubits)),
          psi_full_()
        {
            rng.seed(Config::RANDOM_SEED);

            // Build full-state IndexSet from SpinHalf sites
            std::vector<itensor::Index> idxs;
            idxs.reserve(n_qubits);
            for(int i = 1; i <= n_qubits; ++i)
                idxs.push_back(sites_(i));
            itensor::IndexSet iset(idxs);
            psi_full_ = itensor::ITensor(iset);
        }

        // Virtual destructor inherits from QuantumState
        ~TN_CUDA() override = default;

        void reset_state() override
        {
            psi_full_.fill(0.0);
            // Prepare |0…0⟩: first level of each spin
            std::vector<itensor::IndexVal> iv(n_qubits);
            for(int i = 0; i < n_qubits; ++i)
                iv[i] = sites_(i+1)(1);
            psi_full_.set(iv, 1.0);
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
            reset_state();
            assert(circuit->num_qubits() == n_qubits);
            auto gates = fuse_circuit_sv(circuit);
            for(auto const& g : gates)
            {
                if(g.op_name == OP::C1)
                    one_qubit_gate(g);
                else if(g.op_name == OP::C2)
                    two_qubit_gate(g);
            }
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
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);

            const IdxType nstates = IdxType(1) << n_qubits;
            std::vector<ValType> probs(nstates);
            std::vector<itensor::IndexVal> iv(n_qubits);

            for(IdxType s = 0; s < nstates; ++s)
            {
                IdxType tmp = s;
                for(int q = 0; q < n_qubits; ++q)
                {
                    auto bit = tmp & 1;
                    tmp >>= 1;
                    iv[q] = sites_(q+1)(bit+1);
                }
                auto amp = itensor::elt(psi_full_, iv);
                probs[s] = std::norm(amp);
            }

            std::discrete_distribution<IdxType> dist(probs.begin(), probs.end());
            for(IdxType rep = 0; rep < repetition; ++rep)
                results[rep] = dist(rng);

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
        ValType                                 cpu_mem;

        itensor::SpinHalf sites_;
        itensor::ITensor   psi_full_;

        void one_qubit_gate(const SVGate& g)
        {
            int site = g.qubit + 1;
            auto I  = sites_(site);
            auto Ip = itensor::prime(I);
            itensor::ITensor G(I, Ip);
            // TODO: populate G from g.matrix
            psi_full_ = psi_full_ * G;
            psi_full_.noPrime();
        }

        void two_qubit_gate(const SVGate& g)
        {
            int i1  = g.ctrl + 1;
            int i2  = g.qubit + 1;
            auto I1 = sites_(i1);
            auto I2 = sites_(i2);
            auto Ip1 = itensor::prime(I1);
            auto Ip2 = itensor::prime(I2);
            itensor::ITensor G(I1, I2, Ip1, Ip2);
            // TODO: populate G from g.matrix
            psi_full_ = psi_full_ * G;
            psi_full_.noPrime();
        }
    };

} // namespace NWQSim
