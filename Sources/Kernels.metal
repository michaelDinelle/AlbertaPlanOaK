#include <metal_stdlib>
using namespace metal;

kernel void fire(
    device float *W [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant uint &d [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= n || j >= d) return;
    float w = W[i * d + j];
    const float eps = sqrt(FLT_EPSILON);
    float prev;
    do {
        prev = w;
        float norm = 0;
        for (uint k = 0; k < d; k++) {
            norm += W[i * d + k] * W[i * d + k];
        }
        norm = sqrt(norm);
        w = w * (1.5f - 0.5f * norm * norm);
    } while (fabs(w - prev) > eps && isfinite(w));
    W[i * d + j] = w;
}

kernel void perception_forward(
    device const float *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &outDim [[buffer(3)]],
    constant uint &inDim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= outDim) return;
    float sum = 0;
    for (uint j = 0; j < inDim; j++) {
        sum += W[gid * inDim + j] * x[j];
    }
    y[gid] = tanh(sum);
}

kernel void perception_update(
    device float *W [[buffer(0)]],
    device float *alpha [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const float *err [[buffer(3)]],
    device const float *y [[buffer(4)]],
    constant uint &outDim [[buffer(5)]],
    constant uint &inDim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= outDim || j >= inDim) return;
    uint idx = i * inDim + j;
    float grad = err[i] * x[j] * (1 - y[i] * y[i]);
    alpha[idx] *= exp(grad * grad / float(inDim));
    W[idx] += alpha[idx] * grad;
}

kernel void value_update(
    device float *theta [[buffer(0)]],
    device const float *s [[buffer(1)]],
    device const float *sNext [[buffer(2)]],
    device float *rho [[buffer(3)]],
    device float *wealth [[buffer(4)]],
    device float *gSum [[buffer(5)]],
    constant float &reward [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    float v = 0;
    float vNext = 0;
    for (uint i = 0; i < dim; i++) {
        v += theta[i] * s[i];
        vNext += theta[i] * sNext[i];
    }
    float delta = reward - rho[0] + vNext - v;
    rho[0] += delta / (1 + fabs(delta));
    float grad = delta * s[gid];
    gSum[0] += fabs(grad);
    float eta = wealth[0] / (gSum[0] + FLT_EPSILON);
    theta[gid] += grad * eta;
    if (gid == 0) {
        wealth[0] += -grad * theta[gid] * eta;
    }
}

kernel void policy_act(
    device const float *W [[buffer(0)]],
    device const float *s [[buffer(1)]],
    device float *a [[buffer(2)]],
    constant uint &aDim [[buffer(3)]],
    constant uint &sDim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= aDim) return;
    float sum = 0;
    for (uint j = 0; j < sDim; j++) {
        sum += W[gid * sDim + j] * s[j];
    }
    a[gid] = tanh(sum);
}

kernel void policy_update(
    device float *W [[buffer(0)]],
    device const float *s [[buffer(1)]],
    constant float &adv [[buffer(2)]],
    constant uint &aDim [[buffer(3)]],
    constant uint &sDim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= aDim || j >= sDim) return;
    W[i * sDim + j] += adv * s[j] / float(sDim);
}

kernel void model_predict(
    device const float *W [[buffer(0)]],
    device const float *s [[buffer(1)]],
    device const float *a [[buffer(2)]],
    device const float *proj [[buffer(3)]],
    device float *next [[buffer(4)]],
    device float *phi [[buffer(5)]],
    constant uint &sDim [[buffer(6)]],
    constant uint &aDim [[buffer(7)]],
    constant uint &pDim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < pDim) {
        float sum = 0;
        for (uint j = 0; j < sDim; j++) {
            sum += s[j] * proj[j * pDim + gid];
        }
        for (uint j = 0; j < aDim; j++) {
            sum += a[j] * proj[(sDim + j) * pDim + gid];
        }
        phi[gid] = tanh(sum);
    } else if (gid < pDim + sDim) {
        uint i = gid - pDim;
        float sum = 0;
        for (uint j = 0; j < pDim; j++) {
            sum += W[i * pDim + j] * phi[j];
        }
        next[i] = sum;
    }
}

kernel void model_update(
    device float *W [[buffer(0)]],
    device const float *phi [[buffer(1)]],
    device const float *err [[buffer(2)]],
    constant uint &sDim [[buffer(3)]],
    constant uint &pDim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= sDim || j >= pDim) return;
    float norm = 0;
    for (uint k = 0; k < pDim; k++) {
        norm += phi[k] * phi[k];
    }
    float step = 1.0f / (norm + FLT_EPSILON);
    W[i * pDim + j] += step * err[i] * phi[j];
}
