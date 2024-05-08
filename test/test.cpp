#include <stdio.h>

void gather(int *a, int *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]];
  }
}

int main(){
  int a[4] = {0, 0, 0, 0};
  int b[4] = {1, 2, 3, 4};
  int idx[4] = {3, 2, 1, 0};
  gather(a, b, idx, 4);

  //@expect: 4, 3, 2, 1
  for (int i = 0; i < 4; i++) {
    printf("%d\n", a[i]);
  }
  return 0;
}
