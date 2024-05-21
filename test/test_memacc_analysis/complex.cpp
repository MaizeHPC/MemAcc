#include <stdio.h>

int compute(int x);

void kernel(int *a1, int* a2, int *b1, int* b2, int N, int* c, int* d, int* e) {
    for (int i = 0; i < N; i++) {
        const int  x = b1[i];
        const int  y = b2[i];

        a1[y] += compute(c[i]); //a1[b2[i]] = c[i];
        a2[x] = compute(d[i]) * e[x]; //a2[b1[i]] = d[i] * e[b1[i]];
    }
}
