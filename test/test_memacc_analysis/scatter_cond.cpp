#include <stdio.h>

void gather(int *a, int *b,int* f, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    
    if (f[i] < 2)
      a[idx[i]] = b[i];
    else
      a[idx[i]] = b[idx[i]];
  }
}