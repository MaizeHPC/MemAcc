#include <stdio.h>

void gather(int *a, int *b, int*idx1, int *idx2, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx1[idx2[i]]];
  }
}

int main(){
  int a[4] = {0, 0, 0, 0};
  int b[4] = {1, 2, 3, 4};
  int idx1[4] = {3, 2, 1, 0};
  int idx2[4] = {0, 1, 2, 3};
  gather(a, b, idx1, idx2, 4);
  // verification
  for (int i = 0; i < 4; i++) {
    printf("a[%d] = %d\n", i, a[i]);
  }
  return 0;
}
