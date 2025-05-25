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

int main() {
    int n_qubits = 2;
    auto circuit = std::make_shared<NWQSim::Circuit>(n_qubits);
    circuit->H(0);
    circuit->CX(0, 1);
    circuit->RZ(0.125, 0);

    std::string backend = "CPU";
    std::string sim_method = "sv";
    auto state = BackendManager::create_state(backend, n_qubits, sim_method);

    int shots = 1024;
    circuit->MA(shots);
    state->sim(circuit);
    long long int *result = state->get_results();

    for (int i = 0; i < (1 << n_qubits); ++i) {
        printf("%lld\n", result[i]);
    }

    return 0;
}

