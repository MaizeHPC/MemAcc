#include <stdio.h>

void gather(int *a, int *b, int*idx1, int *idx2, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx1[idx2[i]]];
  }
}

