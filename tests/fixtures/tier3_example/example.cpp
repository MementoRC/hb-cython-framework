#include <vector>
#include <algorithm>

extern "C" {
    double compute_sum(double* data, int n) {
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            total += data[i];
        }
        return total;
    }
}
