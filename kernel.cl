int idx(int x, int y, int w) { return y * w + x; }

__kernel void Reshape(__global float* in, __global int* shape,
                      __global float* out) {
  int x = get_global_id(0);
  out[x] = in[x] * 2.0f;
}

__kernel void Conv(__global float* X, __global float* W, __global float* B,
                   __global float* Y) {}

__kernel void MatMul(__global float* A, int4 shapeA, __global float* B,
                     int4 shapeB, __global float* Y, int4 shapeY) {
  for (int y = 0; y < shapeY.y; y++) {
    for (int x = 0; x < shapeY.x; x++) {
      float sum = 0.0f;
      for (int i = 0; i < shapeA.x; i++) {
        sum += A[idx(i, y, shapeA.x)] * B[idx(x, i, shapeB.x)];
      }
      Y[idx(x, y, shapeY.x)] = sum;
    }
  }
}
