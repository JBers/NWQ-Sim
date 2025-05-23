#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <memory>
#include "../include/backendManager.hpp"
#include "../include/state.hpp"
#include "../include/circuit.hpp"
#include "../include/nwq_util.hpp"

using namespace NWQSim;
using ValType = double;

static constexpr int NQ        = 16;
static constexpr int SHOTS     = 1024;
static constexpr int CIRCUITS  = 20;
static constexpr double THRESHOLD = 2.0/3.0;

static ValType uniform01() {
    return randomval();
}

static void apply_random_su2(Circuit &circ, IdxType q) {
    ValType u1 = uniform01();
    ValType u2 = uniform01();
    ValType u3 = uniform01();
    ValType theta = std::acos(1.0 - 2.0 * u1);
    ValType phi   = 2.0 * PI * u2;
    ValType lam   = 2.0 * PI * u3;
    circ.U(theta, phi, lam, q);
}

static std::shared_ptr<Circuit> build_qv_circuit(int k, std::mt19937_64 &rng) {
    auto circ = std::make_shared<Circuit>(k);
    std::vector<IdxType> perm(k);
    for (IdxType i = 0; i < k; ++i) perm[i] = i;
    for (int layer = 0; layer < k; ++layer) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i + 1 < k; i += 2) {
            IdxType q1 = perm[i];
            IdxType q2 = perm[i+1];
            apply_random_su2(*circ, q1);
            apply_random_su2(*circ, q2);
            circ->CX(q1, q2);
            apply_random_su2(*circ, q1);
            apply_random_su2(*circ, q2);
            circ->CX(q1, q2);
            apply_random_su2(*circ, q1);
            apply_random_su2(*circ, q2);
            circ->CX(q1, q2);
            apply_random_su2(*circ, q1);
            apply_random_su2(*circ, q2);
        }
    }
    return circ;
}

int main() {
    std::mt19937_64 rng(std::random_device{}());
    std::vector<int> bond_dims = {1,10,20,30,40,50,60,70,80,90,100};
    std::ofstream csv("bond_vs_qv.csv");
    csv << "bond_dimension,quantum_volume\n";
    for (int bd : bond_dims) {
        int max_pass = 0;
        for (int k = 2; k <= NQ; ++k) {
            double sum_hop = 0.0;
            size_t dim = size_t(1) << k;
            for (int c = 0; c < CIRCUITS; ++c) {
                auto circ = build_qv_circuit(k, rng);
                printf("Making circuit!");
                auto st   = BackendManager::create_state("CPU", k, "sv");
                printf("Made state!");
                circ->MA(SHOTS);
                st->sim(circ);
                printf("Simulated!");
                long long int *result = st->get_results();
                std::vector<long long> counts(dim, 0);
                for (int s = 0; s < SHOTS; ++s) {
                    counts[result[s]]++;
                }
                delete[] result;
                std::vector<ValType> phat(dim);
                for (size_t i = 0; i < dim; ++i) {
                    phat[i] = ValType(counts[i]) / ValType(SHOTS);
                }
                auto sorted = phat;
                std::sort(sorted.begin(), sorted.end());
                ValType median = (dim & 1)
                    ? sorted[dim/2]
                    : 0.5 * (sorted[dim/2 - 1] + sorted[dim/2]);
                std::vector<bool> is_heavy(dim);
                for (size_t i = 0; i < dim; ++i) {
                    is_heavy[i] = (phat[i] > median);
                }
                long long hits = 0;
                for (size_t i = 0; i < dim; ++i) {
                    if (is_heavy[i]) hits += counts[i];
                }
                double hop = double(hits) / double(SHOTS);
                sum_hop += hop;
            }
            double avg_hop = sum_hop / double(CIRCUITS);
            if (avg_hop > THRESHOLD) {
                max_pass = k;
            } else {
                break;
            }
        }
        int qv = (max_pass > 0) ? (1 << max_pass) : 0;
        std::cout << "bond_dim=" << bd
                  << " log2(QV)=" << max_pass
                  << " QV=" << qv << "\n";
        csv << bd << "," << qv << "\n";
    }
    csv.close();
    return 0;
}
