#include <stdio.h>

void gather(int *a, int *b, int*c, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]] + c[idx[i]];
  }
}

