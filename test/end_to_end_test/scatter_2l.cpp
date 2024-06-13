#include <stdio.h>

void scatter2l(int *a, int *b, int *idx, int* idx2, int n) {
  for (int i = 0; i < n; i++) {
    a[idx[idx2[i]]] = b[i];
  }
}

int main(){
  int a[4] = {0, 0, 0, 0};
  int b[4] = {1, 2, 3, 4};
  int idx[4] = {3, 2, 1, 0};
  int idx2[4] = {0, 1, 2, 3};
  scatter2l(a, b, idx, idx2, 4);
  // verification
  for (int i = 0; i < 4; i++) {
    printf("%d ", a[i]);
  }
  return 0;
}
