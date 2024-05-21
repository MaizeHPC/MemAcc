#include <stdio.h>

void gather(int *a, int *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[idx[i]] += b[idx[i]];
  }
}

int main(){
  int a[4] = {0, 0, 0, 0};
  int b[4] = {1, 2, 3, 4};
  int idx[4] = {3, 2, 1, 0};
  gather(a, b, idx, 4);
  int a_expected[4] = {4, 3, 2, 1};
  // verification
  for (int i = 0; i < 4; i++) {
    if (a[i] != a_expected[i]) {
      printf("Error: a[%d] = %d, expected: %d\n", i, a[i], a_expected[i]);
      return 1;
    }
  }
  printf("Success\n");
  return 0;
}
