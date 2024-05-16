#include <stdio.h>
#include "MAA.hh"

// very heavy compute kernel
extern int compute(int data);
extern int* alloc_spd(int n);

MAA_t* MAA_config(int* spd_buffer, int* b, int* idx, int n){
    MAA_t* maa = new MAA_t();
    // for (int i = 0; i < n; i++)
    int i_loop = maa->loop(/*lower bound=*/0, /*lower bound=*/n, /*step=*/1);
    // idx[i]
    int idx_i = maa->load_internal(/*dependent idx=*/i_loop, /*data=*/idx);
    // b[idx[i]]
    maa->load_external(/*dependent idx=*/idx_i, /*data=*/b, /*buf=*/spd_buffer);
    return maa;
}

void gather_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer = alloc_spd(n);

  // Step1: configure MAA's worklist
  MAA_t* maa = MAA_config(spd_buffer, b, idx, n);

  // Step2: start MAA: perform the gather operation in MAA
  maa->start();
  for (int i = 0; i < n; i++) {
    a[i] = compute(b[idx[i]]);
  }
  // Step3: finish MAA: perform the scatter operation in MAA
  maa->finish();
}


void gather(int *a, int *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]];
  }
}

void scatter(int *a, int *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[idx[i]] = b[i];
  }
}

void scatter_maa(int*a, int *b, int *idx, int n){
  int* spd_buffer = alloc_spd(n);
  
  for (int i = 0; i < n; i++) {
    spd_buffer[i] = b[i];
  }

  packed_generic_store(in:spd_buffer, loop_lower_bound:0, 
                       loop_upper_bound:n, step:1){
    iteration arg: i;
    idx_i = load(idx, i);
    b_i = load(spd_buffer, i);
    write(a, idx_i, b_i);
  }
}

void gather_and_scatter(int *a, int *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[idx[i]] = b[idx[i]];
  }
}

void gather_and_scatter_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer_gather = alloc_spd(n);
  int* spd_buffer_scatter = alloc_spd(n);
  packed_generic_load(out:spd_buffer_gather, loop_lower_bound:0, 
                      loop_upper_bound:n, step:1){
    iteration arg: i;
    idx_i = load(idx, i);
    b_i = load(b, idx_i);
    write(spd_buffer_gather, b_i);
  }
  for (int i = 0; i < n; i++) {
    int data = spd_buffer_gather[i];
    spd_buffer_scatter[i] = data;
  }
  packed_generic_store(in:spd_buffer_scatter, loop_lower_bound:0, 
                       loop_upper_bound:n, step:1){
    iteration arg: i;
    idx_i = load(idx, i);
    b_i = load(spd_buffer_scatter, i);
    write(a, idx_i, b_i);
  }
}

void gather(int *a1, int *a2, int *b1, int *b2, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a1[i] = b1[idx[i]];
    a2[i] = b2[idx[i]];
  }
}

void gather(int *cell, int *cell_tallies, int *cell_indices, int n) {
  for (int i = 0; i < n; i++) {
    int res1=cell[cell_indices[i]];
    int res2=cell_tallies[cell_indices[i]];
  }
}

void gather_maa(int *a1, int *a2, int *b1, int* b2, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer1 = alloc_spd(n);
  int* spd_buffer2 = alloc_spd(n);

  packed_generic_load(out:spd_buffer1,spd_buffer2, loop_lower_bound:0, 
                      loop_upper_bound:n, step:1){
    iteration arg: i;
    idx_i = load(idx, i);
    b1_i = load(b, idx_i);
    b2_i = load(b, idx_i);
    write(spd_buffer1, b1_i);
    write(spd_buffer2, b2_i);
  }
  
  for (int i = 0; i < n; i++) {
    int data = spd_buffer[i];
    a[i] = data;
  }
}

void gather_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer = alloc_spd(n);

  packed_generic_load(out:spd_buffer, loop_lower_bound:0, 
                      loop_upper_bound:n, step:1){
    iteration arg: i;
    idx_i = load(idx, i);
    b_i = load(b, idx_i);
    write(spd_buffer, b_i);
  }
  
  for (int i = 0; i < n; i++) {
    int data = spd_buffer[i];
    a[i] = data;
  }
}

void gather_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer = alloc_spd(n);
  // MAA configuration
  maa = llvm.maa.init();
  loop_i = llvm.maa.loop(0, n, 1, maa);
  idx_i = llvm.maa.load_internal(loop_i, idx, maa);
  llvm.maa.load_external(idx_i, b, spd_buffer, maa);
  // MAA start performing the gather operation
  llvm.maa.start();
  for (int i = 0; i < n; i++) {
    int data = spd_buffer[i];
    a[i] = data;
  }
  // MAA start performing the scatter operation
  llvm.maa.finish();
}

void gather_scatter_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer_gather = alloc_spd(n);
  int* spd_buffer_scatter = alloc_spd(n);
  // MAA configuration
  maa = llvm.maa.init();
  loop_i = llvm.maa.loop(0, n, 1, maa);
  idx_i = llvm.maa.load_internal(loop_i, idx, maa);
  llvm.maa.load_external(idx_i, b, spd_buffer_gather, maa);
  loop_i = llvm.maa.loop(0, n, 1, maa);
  idx_i = llvm.maa.load_internal(loop_i, idx, maa);
  llvm.maa.store_external(idx_i, a,spd_buffer_scatter, maa);
  // MAA start performing the gather operation
  llvm.maa.start();
  for (int i = 0; i < n; i++) {
    int data = spd_buffer_gather[i];
    spd_buffer_scatter[i] = data;
  }
  // MAA start performing the scatter operation
  llvm.maa.finish();
}

void gather_scatter_maa(int *a, int *b, int *idx, int n) {
  // allocate buffer for b's data
  int* spd_buffer_gather = alloc_spd(n);
  int* spd_buffer_scatter = alloc_spd(n);
  // MAA configuration
  maa = llvm.maa.init();
  loop_i = llvm.maa.loop(0, n, 1, maa);
  idx_i = llvm.maa.load_internal(loop_i, idx, maa);
  llvm.maa.load_external(idx_i, b, spd_buffer_gather, maa);
  llvm.maa.store_external(idx_i, a,spd_buffer_scatter, maa);
  // MAA start performing the gather operation
  llvm.maa.start();
  for (int i = 0; i < n; i++) {
    int data = spd_buffer_gather[i];
    spd_buffer_scatter[i] = data;
  }
  // MAA start performing the scatter operation
  llvm.maa.finish();
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
