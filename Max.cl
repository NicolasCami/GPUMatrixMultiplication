__kernel void Multiply(__global __read_only int* a, __global __read_only int* b, __global __write_only int* c, int size) {
    int gridX = get_global_id(0);
    int gridY = get_global_id(1);

    int sum = 0;
    for (int k = 0; k < size; k++) {
        sum += a[size * gridY + k] * b[size * k + gridX];
    }

    c[size * gridY + gridX] = sum;
}

