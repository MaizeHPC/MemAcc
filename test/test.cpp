void gather(double *a, double *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]];
  }
}