#include <cutensornet.h>
#include <iostream>

int main() {
    cutensornetHandle_t handle;
    // initialize handle
    if (cutensornetCreate(&handle) != CUTENSORNET_STATUS_SUCCESS) {
        std::cerr << "cuTensorNet init failed\n";
        return EXIT_FAILURE;
    }
    // clean up
    cutensornetDestroy(handle);
    std::cout << "cuTensorNet linked and callable\n";
    return EXIT_SUCCESS;
}
