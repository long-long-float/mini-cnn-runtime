int idx(int x, int y, int w) { return y * w + x; }

__kernel void Reshape(__global float* in, __global int* shape,
                      __global float* out) {
  int x = get_global_id(0);
  out[x] = in[x] * 2.0f;
}

__kernel void Conv(__global float* X, __global float* W, __global float* B,
                   __global float* Y) {}

#define PITCH (16 * 4)

__kernel void MatMul(__global float* A, int4 shapeA, __global float* B,
                     int4 shapeB, __global float* Y, int4 shapeY) {
#if 0
  for (int y = 0; y < shapeY.y; y++) {
    for (int x = 0; x < shapeY.x; x++) {
      float sum = 0.0f;
      for (int i = 0; i < shapeA.x; i++) {
        sum += A[idx(i, y, shapeA.x)] * B[idx(x, i, shapeB.x)];
      }
      Y[idx(x, y, shapeY.x)] = sum;
    }
  }
#else
  for (int y = 0; y < shapeY.y; y++) {
    for (int x = 0; x < shapeY.x; x++) {
      float sum = 0.0f;
      __global float* baseA = A + y * shapeA.x;
      for (int i = 0; i < shapeA.x / 16; i += 4) {
        float16 a16[4] = {
            vload16(i, baseA),
            vload16(i + 1, baseA),
            vload16(i + 2, baseA),
            vload16(i + 3, baseA),
        };
        float b16[PITCH];
        for (int j = 0; j < PITCH; j++) {
          b16[j] = B[idx(x, i * 16 + j, shapeB.x)];
        }
        /* slow
        for (int j = 0; j < 4; j++) {
          sum += dot(((float4*)&a16)[j], ((float4*)b16)[j]);
        }
        */
        for (int j = 0; j < PITCH; j++) {
          sum += ((float*)&a16)[j] * b16[j];
        }
      }
#endif
  Y[idx(x, y, shapeY.x)] = sum;
}
}
}
