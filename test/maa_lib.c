#include <stdio.h>
#include <stdlib.h>

void* maa_spdalloc_i64(long size) {
    printf("Function: maa_spdalloc_i64, Argument: %ld\n", size);
    return malloc(size);
}

int maa_setloop_i64_i64_i32(long start, long size, int step) {
    printf("Function: maa_setloop_i64_i64_i32, Start: %ld, Size: %ld, Step: %d\n", start, size, step);
    return 0;  // Returning 0 as a placeholder
}

int maa_stream_i32_p0(int stream_id, void* ptr) {
    printf("Function: maa_stream_i32_p0, Stream ID: %d, Ptr: %p\n", stream_id, ptr);
    return stream_id;  // Returning the stream ID as a placeholder
}

int maa_indirect_ext_i32_p0_p0(int index, void* ptr, void* buf) {
    printf("Function: maa_indirect_i32_p0, Index: %d, Ptr: %p, buf: %p\n", index, ptr, buf);
    return index;  // Returning the index as a placeholder
}

int maa_start_i32(int root) {
    printf("Function: maa_start_i32 Root: %d\n", root);
    return 0;  // Returning 0 as a placeholder
}