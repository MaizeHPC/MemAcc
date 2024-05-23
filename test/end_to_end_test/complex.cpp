#include <stdio.h>

int compute(int x);

void kernel(int *a1, int* a2, int *b1, int* b2, int N, int* c, int* d, int* e) {
    for (int i = 0; i < N; i++) {
        const int  x = b1[i];
        const int  y = b2[i];

        a1[y] += c[i]; //a1[b2[i]] = c[i];
        a2[x] = d[i] * e[x]; //a2[b1[i]] = d[i] * e[b1[i]];
    }
}

int main() {
    int N = 4;
    int a1[4] = {0, 0, 0, 0};
    int a2[4] = {0, 0, 0, 0};
    int b1[4] = {3, 2, 1, 0};
    int b2[4] = {0, 1, 2, 3};
    int c[4] = {1, 2, 3, 4};
    int d[4] = {4, 3, 2, 1};
    int e[4] = {1, 2, 3, 4};

    kernel(a1, a2, b1, b2, N, c, d, e);

    // verification
    for (int i = 0; i < 4; i++) {
        printf("a1[%d] = %d\n", i, a1[i]);
        printf("a2[%d] = %d\n", i, a2[i]);
    }
    return 0;
}
