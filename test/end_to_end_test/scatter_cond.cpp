#include <stdio.h>

void scatter(int *a, int *b,int* f, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    
    if (f[i] < 1)
      a[idx[i]] = b[i];
    else
      a[idx[i]] = b[idx[i]];
  }
}

int main(){
  int a[4] = {0, 0, 0, 0};
  int b[4] = {1, 2, 3, 4};
  int idx[4] = {3, 2, 1, 0};
  int f[4] = {1, 1, 0, 0};
  scatter(a, b, f, idx, 4);
  // verification
  for (int i = 0; i < 4; i++) {
    printf("%d ", a[i]);
  }
  return 0;
}