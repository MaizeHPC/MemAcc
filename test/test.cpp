void gather(double *a, double *b, int *idx, int n, const int C) {
  static double dest[100];
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]];
    dest[i] = a[i] + C;
  }
}