#include <stdio.h>

void rmw(int *a, int *b, int *idx, int n1, int n2) {
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
        a[j] += b[idx[j]];
    }
  }
}

int main(){
    int a[4] = {0, 0, 0, 0};
    int b[4] = {1, 2, 3, 4};
    int idx[4] = {3, 2, 1, 0};
    rmw(a, b, idx, 4, 4);
  return 0;
}