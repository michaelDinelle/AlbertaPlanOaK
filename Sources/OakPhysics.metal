#ifndef OAK_PHYSICS_METAL_INCLUDED
#define OAK_PHYSICS_METAL_INCLUDED

#include <metal_stdlib>
#include "HalfCheetahModel.metal"

// Physics-scoped fast math (agent kernel math stays -fno-fast-math): the env
// stage is a per-lane serial latency chain (~33% of step wall); relaxed fp
// contraction/reassociation shortens it. Instrument change to the environment
// — re-certified by the paired population gate at the M-8192 equivalence tier
// (the F4 protocol). RNG helpers below are excluded (safe region).
#pragma METAL fp math_mode(fast)
using namespace metal;

constant float HC_BODY_SUBTREEMASS[8] = { 14.0f, 14.0f, 4.22635983f, 2.68284519f, 1.09539749f, 3.52343096f, 2.08535565f, 0.884518828f };
constant float HC_BODY_INVWEIGHT0[16] = { 0.0f, 0.0f, 0.0641575195f, 0.138782354f, 0.0969110156f, 1.00990763f, 0.127209225f, 1.47257736f, 0.243749283f, 3.77214058f, 0.0814837948f, 1.09180348f, 0.131996818f, 1.88718446f, 0.266144103f, 4.19715993f };
constant float HC_DOF_INVWEIGHT0[9] = { 0.106735728f, 0.0845922916f, 0.416347062f, 2.72978864f, 5.89324898f, 8.894877f, 3.08139261f, 6.88279596f, 9.46813496f };

constant int HC_M_ROWNNZ[9] = { 1, 2, 3, 4, 5, 6, 4, 5, 6 };
constant int HC_M_ROWADR[9] = { 0, 1, 3, 6, 10, 15, 21, 25, 30 };
constant int HC_M_COLIND[36] = {
    0,
    0, 1,
    0, 1, 2,
    0, 1, 2, 3,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 5,
    0, 1, 2, 6,
    0, 1, 2, 6, 7,
    0, 1, 2, 6, 7, 8 };
constant int HC_DOF_BODY[9] = { 1, 1, 1, 2, 3, 4, 5, 6, 7 };
constant int HC_BODY_DOFADR[8] = { 0, 0, 3, 4, 5, 6, 7, 8 };
constant int HC_BODY_DOFNUM[8] = { 0, 3, 1, 1, 1, 1, 1, 1 };
constant int HC_BODY_LASTDOF[8] = { -1, 2, 3, 4, 5, 6, 7, 8 };

constant float HC_MINVAL = 1e-15f;
constant float HC_MAXVAL = 1e10f;
constant float HC_MINMU = 1e-5f;
constant float HC_MINIMP = 0.0001f;
constant float HC_MAXIMP = 0.9999f;
constant float HC_CHOL_EPS = 1e-12f;
constant int HC_NEFC_MAX = 70;

struct HCState {
    float qpos[9];
    float qvel[9];
    float ctrl[6];
    float qacc_warmstart[9];
};

struct HCWork {
    float xipos[24];
    float ximat[72];
    float xanchor[27];
    float xaxis[27];
    float geom_xpos[27];
    float geom_xaxis[27];
    float com1[3];
    float cinert[80];
    float cdof[54];
    float cdofdot[54];
    float cvel[48];
    float M[36], qLD[36], qLDiagInv[9];
    float scale;
    int ncon;
    float con_dist[16];
    int nefc;
    float efc_J[630];
    float efc_pos[70];
    float efc_D[70];
    float efc_aref[70];
    float qfrc_bias[9];
    float qfrc_passive[9];
    float qfrc_smooth[9];
    float qacc_smooth[9];
    float qacc[9];
    float qfrc_constraint[9];
};

struct HCContacts {
    float con_pos[48];
    float con_frame[144];
    int con_exclude[16];
    int con_geom[16];
};

struct HCRefCoef {
    float efc_imp[70];
    float efc_K[70];
    float efc_B[70];
};

inline bool hc_isBad(float x) {
    return (x != x) || (x > HC_MAXVAL) || (x < -HC_MAXVAL);
}

inline float hc_dot3(thread const float *a, thread const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
inline float hc_dot6(thread const float *a, thread const float *b) {
    float r = 0.0f;
    for (int i = 0; i < 6; i++) r += a[i]*b[i];
    return r;
}
inline float hc_dot9(thread const float *a, thread const float *b) {
    float r = 0.0f;
    for (int i = 0; i < 9; i++) r += a[i]*b[i];
    return r;
}
inline float hc_twoSumErr(float a, float b, float s) {
    return (fabs(a) >= fabs(b)) ? ((a - s) + b) : ((b - s) + a);
}
struct HCAcc {
    float s, c;
};
inline void hc_accInit(thread HCAcc &a, float v) { a.s = v; a.c = 0.0f; }
inline void hc_accAdd(thread HCAcc &a, float v) {
    float t = a.s + v;
    a.c += hc_twoSumErr(a.s, v, t);
    a.s = t;
}
inline void hc_accMulAdd(thread HCAcc &a, float x, float y) {
    float p = x*y;
    float e = fma(x, y, -p);
    float t = a.s + p;
    a.c += hc_twoSumErr(a.s, p, t) + e;
    a.s = t;
}
inline float hc_accGet(thread const HCAcc &a) { return a.s + a.c; }
inline float hc_dot9c(thread const float *a, thread const float *b, float init) {
    HCAcc acc;
    hc_accInit(acc, init);
    for (int i = 0; i < 9; i++) hc_accMulAdd(acc, a[i], b[i]);
    return hc_accGet(acc);
}
inline void hc_cross(thread float *res, thread const float *a, thread const float *b) {
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}
inline void hc_normalize3(thread float *v) {
    float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm < HC_MINVAL) {
        v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f;
    } else {
        float inv = 1.0f / norm;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
}
inline void hc_normalize4(thread float *q) {
    float norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < HC_MINVAL) {
        q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
    } else {
        float inv = 1.0f / norm;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}
inline void hc_mulQuat(thread float *res, thread const float *a, thread const float *b) {
    float r0 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    float r1 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    float r2 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    float r3 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    res[0] = r0; res[1] = r1; res[2] = r2; res[3] = r3;
}
inline void hc_rotVecQuat(thread float *res, thread const float *vec, thread const float *q) {
    if (vec[0] == 0.0f && vec[1] == 0.0f && vec[2] == 0.0f) {
        res[0] = 0.0f; res[1] = 0.0f; res[2] = 0.0f;
    } else if (q[0] == 1.0f && q[1] == 0.0f && q[2] == 0.0f && q[3] == 0.0f) {
        res[0] = vec[0]; res[1] = vec[1]; res[2] = vec[2];
    } else {
        float t0 = q[0]*vec[0] + q[2]*vec[2] - q[3]*vec[1];
        float t1 = q[0]*vec[1] + q[3]*vec[0] - q[1]*vec[2];
        float t2 = q[0]*vec[2] + q[1]*vec[1] - q[2]*vec[0];
        float r0 = vec[0] + 2.0f*(q[2]*t2 - q[3]*t1);
        float r1 = vec[1] + 2.0f*(q[3]*t0 - q[1]*t2);
        float r2 = vec[2] + 2.0f*(q[1]*t1 - q[2]*t0);
        res[0] = r0; res[1] = r1; res[2] = r2;
    }
}
inline void hc_axisAngle2Quat(thread float *res, thread const float *axis, float angle) {
    if (angle == 0.0f) {
        res[0] = 1.0f; res[1] = 0.0f; res[2] = 0.0f; res[3] = 0.0f;
    } else {
        float half_ = angle * 0.5f;
        float s = sin(half_);
        res[0] = cos(half_);
        res[1] = axis[0]*s; res[2] = axis[1]*s; res[3] = axis[2]*s;
    }
}
inline void hc_quat2Mat(thread float *res, thread const float *q) {
    if (q[0] == 1.0f && q[1] == 0.0f && q[2] == 0.0f && q[3] == 0.0f) {
        res[0] = 1.0f; res[1] = 0.0f; res[2] = 0.0f;
        res[3] = 0.0f; res[4] = 1.0f; res[5] = 0.0f;
        res[6] = 0.0f; res[7] = 0.0f; res[8] = 1.0f;
    } else {
        float q00 = q[0]*q[0], q01 = q[0]*q[1], q02 = q[0]*q[2], q03 = q[0]*q[3];
        float q11 = q[1]*q[1], q12 = q[1]*q[2], q13 = q[1]*q[3];
        float q22 = q[2]*q[2], q23 = q[2]*q[3], q33 = q[3]*q[3];
        res[0] = q00 + q11 - q22 - q33;
        res[4] = q00 - q11 + q22 - q33;
        res[8] = q00 - q11 - q22 + q33;
        res[1] = 2.0f*(q12 - q03);
        res[2] = 2.0f*(q13 + q02);
        res[3] = 2.0f*(q12 + q03);
        res[5] = 2.0f*(q23 - q01);
        res[6] = 2.0f*(q13 - q02);
        res[7] = 2.0f*(q23 + q01);
    }
}
inline void hc_mulInertVec(thread float *res, thread const float *i, thread const float *v) {
    res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
    res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
    res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
    res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
    res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
    res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}
inline void hc_crossMotion(thread float *res, thread const float *v, thread const float *m) {
    res[0] = v[1]*m[2] - v[2]*m[1];
    res[1] = v[2]*m[0] - v[0]*m[2];
    res[2] = v[0]*m[1] - v[1]*m[0];
    res[3] = v[1]*m[5] - v[2]*m[4] + v[4]*m[2] - v[5]*m[1];
    res[4] = v[2]*m[3] - v[0]*m[5] + v[5]*m[0] - v[3]*m[2];
    res[5] = v[0]*m[4] - v[1]*m[3] + v[3]*m[1] - v[4]*m[0];
}
inline void hc_crossForce(thread float *res, thread const float *v, thread const float *f) {
    res[0] = v[1]*f[2] - v[2]*f[1] + v[4]*f[5] - v[5]*f[4];
    res[1] = v[2]*f[0] - v[0]*f[2] + v[5]*f[3] - v[3]*f[5];
    res[2] = v[0]*f[1] - v[1]*f[0] + v[3]*f[4] - v[4]*f[3];
    res[3] = v[1]*f[5] - v[2]*f[4];
    res[4] = v[2]*f[3] - v[0]*f[5];
    res[5] = v[0]*f[4] - v[1]*f[3];
}
inline void hc_inertCom(thread float *res, thread const float *inert, thread const float *mat,
                        thread const float *off, float mass) {
    float tmp[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            tmp[3*i + j] = inert[i] * mat[3*j + i];
    res[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6];
    res[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7];
    res[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8];
    res[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7];
    res[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8];
    res[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8];
    res[0] += mass*(off[1]*off[1] + off[2]*off[2]);
    res[1] += mass*(off[0]*off[0] + off[2]*off[2]);
    res[2] += mass*(off[0]*off[0] + off[1]*off[1]);
    res[3] -= mass*off[0]*off[1];
    res[4] -= mass*off[0]*off[2];
    res[5] -= mass*off[1]*off[2];
    res[6] = mass*off[0];
    res[7] = mass*off[1];
    res[8] = mass*off[2];
    res[9] = mass;
}
inline void hc_makeFrame(thread float *f) {
    hc_normalize3(f);
    float ynorm = sqrt(f[3]*f[3] + f[4]*f[4] + f[5]*f[5]);
    if (ynorm < 0.5f) {
        f[3] = 0.0f; f[4] = 0.0f; f[5] = 0.0f;
        if (fabs(f[1]) < 0.9f) f[4] = 1.0f;
        else f[5] = 1.0f;
    }
    float d = hc_dot3(f, f + 3);
    float tmp[3] = { f[3] - f[0]*d, f[4] - f[1]*d, f[5] - f[2]*d };
    hc_normalize3(tmp);
    f[3] = tmp[0]; f[4] = tmp[1]; f[5] = tmp[2];
    hc_cross(f + 6, f, f + 3);
}

inline void hc_factorI(thread float *LD, thread float *diagInv) {
    for (int j = 8; j >= 0; j--) {
        int start = HC_M_ROWADR[j];
        int diagidx = HC_M_ROWNNZ[j] - 1;
        int end = start + diagidx;
        float invD = 1.0f / LD[end];
        diagInv[j] = invD;
        for (int adr = end - 1; adr >= start; adr--) {
            int i = HC_M_COLIND[adr];
            float a = -LD[adr] * invD;
            int ri = HC_M_ROWADR[i];
            int ni = HC_M_ROWNNZ[i];
            for (int k = 0; k < ni; k++) LD[ri + k] += LD[start + k] * a;
        }
        for (int adr = start; adr < end; adr++) LD[adr] *= invD;
    }
}
inline void hc_solveLD(thread float *x, thread const float *LD, thread const float *diagInv) {
    for (int i = 8; i >= 0; i--) {
        if (HC_M_ROWNNZ[i] == 1) continue;
        float xi = x[i];
        if (xi != 0.0f) {
            int start = HC_M_ROWADR[i];
            int nn = HC_M_ROWNNZ[i] - 1;
            for (int k = 0; k < nn; k++) x[HC_M_COLIND[start + k]] -= LD[start + k] * xi;
        }
    }
    for (int i = 0; i < 9; i++) x[i] *= diagInv[i];
    for (int i = 0; i < 9; i++) {
        if (HC_M_ROWNNZ[i] == 1) continue;
        int start = HC_M_ROWADR[i];
        int nn = HC_M_ROWNNZ[i] - 1;
        float acc = 0.0f;
        for (int k = 0; k < nn; k++) acc += LD[start + k] * x[HC_M_COLIND[start + k]];
        x[i] -= acc;
    }
}
inline void hc_mulSymVecSparse(thread const float *M, thread const float *x, thread float *res) {
    HCAcc acc[9];
    for (int i = 0; i < 9; i++) hc_accInit(acc[i], 0.0f);
    for (int i = 0; i < 9; i++) {
        int adr = HC_M_ROWADR[i];
        int nnz = HC_M_ROWNNZ[i];
        hc_accMulAdd(acc[i], M[adr + nnz - 1], x[i]);
        for (int k = 0; k < nnz - 1; k++) {
            int j = HC_M_COLIND[adr + k];
            hc_accMulAdd(acc[i], M[adr + k], x[j]);
            hc_accMulAdd(acc[j], M[adr + k], x[i]);
        }
    }
    for (int i = 0; i < 9; i++) res[i] = hc_accGet(acc[i]);
}

inline void hc_kinematics(thread const HCState &s, thread HCWork &w) {
    float xpos[24];
    float xquat[32];
    float xmat[72];
    for (int k = 0; k < 3; k++) { xpos[k] = 0.0f; w.xipos[k] = 0.0f; }
    xquat[0] = 1.0f; xquat[1] = 0.0f; xquat[2] = 0.0f; xquat[3] = 0.0f;
    for (int k = 0; k < 9; k++) {
        xmat[k] = (k % 4 == 0) ? 1.0f : 0.0f;
        w.ximat[k] = (k % 4 == 0) ? 1.0f : 0.0f;
    }
    for (int i = 1; i < 8; i++) {
        int pid = HC_BODY_PARENT[i];
        float bpos[3] = { HC_BODY_POS[3*i], HC_BODY_POS[3*i + 1], HC_BODY_POS[3*i + 2] };
        float bquat[4] = { HC_BODY_QUAT[4*i], HC_BODY_QUAT[4*i + 1], HC_BODY_QUAT[4*i + 2], HC_BODY_QUAT[4*i + 3] };
        float pos[3], quat[4];
        if (pid == 0) {
            pos[0] = bpos[0]; pos[1] = bpos[1]; pos[2] = bpos[2];
            quat[0] = bquat[0]; quat[1] = bquat[1]; quat[2] = bquat[2]; quat[3] = bquat[3];
        } else {
            thread const float *m = xmat + 9*pid;
            pos[0] = m[0]*bpos[0] + m[1]*bpos[1] + m[2]*bpos[2] + xpos[3*pid + 0];
            pos[1] = m[3]*bpos[0] + m[4]*bpos[1] + m[5]*bpos[2] + xpos[3*pid + 1];
            pos[2] = m[6]*bpos[0] + m[7]*bpos[1] + m[8]*bpos[2] + xpos[3*pid + 2];
            float pq[4] = { xquat[4*pid], xquat[4*pid + 1], xquat[4*pid + 2], xquat[4*pid + 3] };
            hc_mulQuat(quat, pq, bquat);
        }
        for (int j = 0; j < 9; j++) {
            if (HC_JNT_BODY[j] != i) continue;
            float axisLocal[3] = { HC_JNT_AXIS[3*j], HC_JNT_AXIS[3*j + 1], HC_JNT_AXIS[3*j + 2] };
            float jp[3] = { HC_JNT_POS[3*j], HC_JNT_POS[3*j + 1], HC_JNT_POS[3*j + 2] };
            float xax[3], anc[3];
            hc_rotVecQuat(xax, axisLocal, quat);
            hc_rotVecQuat(anc, jp, quat);
            anc[0] += pos[0]; anc[1] += pos[1]; anc[2] += pos[2];
            if (HC_JNT_TYPE[j] == 2) {
                float dq = s.qpos[j] - HC_QPOS0[j];
                pos[0] += xax[0]*dq; pos[1] += xax[1]*dq; pos[2] += xax[2]*dq;
            } else {
                float qloc[4], qn[4];
                hc_axisAngle2Quat(qloc, axisLocal, s.qpos[j] - HC_QPOS0[j]);
                hc_mulQuat(qn, quat, qloc);
                quat[0] = qn[0]; quat[1] = qn[1]; quat[2] = qn[2]; quat[3] = qn[3];
                float off[3];
                hc_rotVecQuat(off, jp, quat);
                pos[0] = anc[0] - off[0]; pos[1] = anc[1] - off[1]; pos[2] = anc[2] - off[2];
            }
            w.xanchor[3*j] = anc[0]; w.xanchor[3*j + 1] = anc[1]; w.xanchor[3*j + 2] = anc[2];
            w.xaxis[3*j] = xax[0]; w.xaxis[3*j + 1] = xax[1]; w.xaxis[3*j + 2] = xax[2];
        }
        hc_normalize4(quat);
        xpos[3*i] = pos[0]; xpos[3*i + 1] = pos[1]; xpos[3*i + 2] = pos[2];
        xquat[4*i] = quat[0]; xquat[4*i + 1] = quat[1]; xquat[4*i + 2] = quat[2]; xquat[4*i + 3] = quat[3];
        hc_quat2Mat(xmat + 9*i, quat);
    }
    for (int i = 1; i < 8; i++) {
        thread const float *m = xmat + 9*i;
        float ip[3] = { HC_BODY_IPOS[3*i], HC_BODY_IPOS[3*i + 1], HC_BODY_IPOS[3*i + 2] };
        w.xipos[3*i + 0] = m[0]*ip[0] + m[1]*ip[1] + m[2]*ip[2] + xpos[3*i + 0];
        w.xipos[3*i + 1] = m[3]*ip[0] + m[4]*ip[1] + m[5]*ip[2] + xpos[3*i + 1];
        w.xipos[3*i + 2] = m[6]*ip[0] + m[7]*ip[1] + m[8]*ip[2] + xpos[3*i + 2];
        float iq[4] = { HC_BODY_IQUAT[4*i], HC_BODY_IQUAT[4*i + 1], HC_BODY_IQUAT[4*i + 2], HC_BODY_IQUAT[4*i + 3] };
        float bq[4] = { xquat[4*i], xquat[4*i + 1], xquat[4*i + 2], xquat[4*i + 3] };
        float q[4];
        hc_mulQuat(q, bq, iq);
        hc_quat2Mat(w.ximat + 9*i, q);
    }
    for (int g = 0; g < 9; g++) {
        int b = HC_GEOM_BODY[g];
        thread const float *m = xmat + 9*b;
        float gp[3] = { HC_GEOM_POS[3*g], HC_GEOM_POS[3*g + 1], HC_GEOM_POS[3*g + 2] };
        w.geom_xpos[3*g + 0] = m[0]*gp[0] + m[1]*gp[1] + m[2]*gp[2] + xpos[3*b + 0];
        w.geom_xpos[3*g + 1] = m[3]*gp[0] + m[4]*gp[1] + m[5]*gp[2] + xpos[3*b + 1];
        w.geom_xpos[3*g + 2] = m[6]*gp[0] + m[7]*gp[1] + m[8]*gp[2] + xpos[3*b + 2];
        float gq[4] = { HC_GEOM_QUAT[4*g], HC_GEOM_QUAT[4*g + 1], HC_GEOM_QUAT[4*g + 2], HC_GEOM_QUAT[4*g + 3] };
        float bq[4] = { xquat[4*b], xquat[4*b + 1], xquat[4*b + 2], xquat[4*b + 3] };
        float q[4], gm[9];
        hc_mulQuat(q, bq, gq);
        hc_quat2Mat(gm, q);
        w.geom_xaxis[3*g + 0] = gm[2];
        w.geom_xaxis[3*g + 1] = gm[5];
        w.geom_xaxis[3*g + 2] = gm[8];
    }
}

inline void hc_comPos(thread HCWork &w) {
    float com[24];
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 3; k++)
            com[3*i + k] = w.xipos[3*i + k] * HC_BODY_MASS[i];
    for (int i = 7; i >= 1; i--) {
        int p = HC_BODY_PARENT[i];
        for (int k = 0; k < 3; k++) com[3*p + k] += com[3*i + k];
    }
    for (int i = 0; i < 8; i++) {
        if (HC_BODY_SUBTREEMASS[i] < HC_MINVAL) {
            for (int k = 0; k < 3; k++) com[3*i + k] = w.xipos[3*i + k];
        } else {
            float inv = 1.0f / HC_BODY_SUBTREEMASS[i];
            for (int k = 0; k < 3; k++) com[3*i + k] *= inv;
        }
    }
    w.com1[0] = com[3]; w.com1[1] = com[4]; w.com1[2] = com[5];
    for (int k = 0; k < 10; k++) w.cinert[k] = 0.0f;
    for (int i = 1; i < 8; i++) {
        float inert[3] = { HC_BODY_INERTIA[3*i], HC_BODY_INERTIA[3*i + 1], HC_BODY_INERTIA[3*i + 2] };
        float off[3] = { w.xipos[3*i] - w.com1[0], w.xipos[3*i + 1] - w.com1[1], w.xipos[3*i + 2] - w.com1[2] };
        hc_inertCom(w.cinert + 10*i, inert, w.ximat + 9*i, off, HC_BODY_MASS[i]);
    }
    for (int j = 0; j < 9; j++) {
        thread const float *ax = w.xaxis + 3*j;
        if (HC_JNT_TYPE[j] == 2) {
            w.cdof[6*j + 0] = 0.0f; w.cdof[6*j + 1] = 0.0f; w.cdof[6*j + 2] = 0.0f;
            w.cdof[6*j + 3] = ax[0]; w.cdof[6*j + 4] = ax[1]; w.cdof[6*j + 5] = ax[2];
        } else {
            float off[3] = { w.com1[0] - w.xanchor[3*j], w.com1[1] - w.xanchor[3*j + 1], w.com1[2] - w.xanchor[3*j + 2] };
            w.cdof[6*j + 0] = ax[0]; w.cdof[6*j + 1] = ax[1]; w.cdof[6*j + 2] = ax[2];
            hc_cross(w.cdof + 6*j + 3, w.xaxis + 3*j, off);
        }
    }
}

inline void hc_makeM(thread HCWork &w) {
    float crb[80];
    for (int k = 0; k < 80; k++) crb[k] = w.cinert[k];
    for (int i = 7; i >= 1; i--) {
        int p = HC_BODY_PARENT[i];
        if (p > 0)
            for (int k = 0; k < 10; k++) crb[10*p + k] += crb[10*i + k];
    }
    for (int k = 0; k < 36; k++) w.M[k] = 0.0f;
    for (int i = 0; i < 9; i++) {
        int adrDiag = HC_M_ROWADR[i] + HC_M_ROWNNZ[i] - 1;
        w.M[adrDiag] = HC_DOF_ARMATURE[i];
        float buf[6];
        hc_mulInertVec(buf, crb + 10*HC_DOF_BODY[i], w.cdof + 6*i);
        int a = adrDiag;
        int j = i;
        while (j >= 0) {
            w.M[a] += hc_dot6(w.cdof + 6*j, buf);
            a--;
            j = HC_DOF_PARENT[j];
        }
    }
}

struct HCDf { float hi, lo; };
inline HCDf hc_dfMake(float h) { HCDf r; r.hi = h; r.lo = 0.0f; return r; }
inline HCDf hc_dfTwoSum(float a, float b) {
    HCDf r;
    r.hi = a + b;
    float bb = r.hi - a;
    r.lo = (a - (r.hi - bb)) + (b - bb);
    return r;
}
inline HCDf hc_dfQuickSum(float a, float b) {
    HCDf r;
    r.hi = a + b;
    r.lo = b - (r.hi - a);
    return r;
}
inline HCDf hc_dfAdd(HCDf a, HCDf b) {
    HCDf s = hc_dfTwoSum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return hc_dfQuickSum(s.hi, s.lo);
}
inline HCDf hc_dfNeg(HCDf a) { HCDf r; r.hi = -a.hi; r.lo = -a.lo; return r; }
inline HCDf hc_dfSub(HCDf a, HCDf b) { return hc_dfAdd(a, hc_dfNeg(b)); }
inline HCDf hc_dfProd(float a, float b) {
    HCDf r;
    r.hi = a*b;
    r.lo = fma(a, b, -r.hi);
    return r;
}
inline HCDf hc_dfMulF(HCDf a, float b) {
    HCDf p = hc_dfProd(a.hi, b);
    p.lo += a.lo*b;
    return hc_dfQuickSum(p.hi, p.lo);
}
inline HCDf hc_dfMul(HCDf a, HCDf b) {
    HCDf p = hc_dfProd(a.hi, b.hi);
    p.lo += a.hi*b.lo + a.lo*b.hi;
    return hc_dfQuickSum(p.hi, p.lo);
}
inline HCDf hc_dfRsqrt(HCDf x) {
    float r0 = rsqrt(x.hi);
    HCDf t = hc_dfMulF(hc_dfMulF(x, r0), r0);
    HCDf u = hc_dfSub(hc_dfMake(3.0f), t);
    return hc_dfMulF(hc_dfMulF(u, r0), 0.5f);
}
inline void hc_dfQuatMulF(thread const HCDf *a, thread const float *b, thread HCDf *res) {
    HCDf r0 = hc_dfSub(hc_dfSub(hc_dfSub(hc_dfMulF(a[0], b[0]), hc_dfMulF(a[1], b[1])),
                                hc_dfMulF(a[2], b[2])), hc_dfMulF(a[3], b[3]));
    HCDf r1 = hc_dfSub(hc_dfAdd(hc_dfAdd(hc_dfMulF(a[0], b[1]), hc_dfMulF(a[1], b[0])),
                                hc_dfMulF(a[2], b[3])), hc_dfMulF(a[3], b[2]));
    HCDf r2 = hc_dfAdd(hc_dfAdd(hc_dfSub(hc_dfMulF(a[0], b[2]), hc_dfMulF(a[1], b[3])),
                                hc_dfMulF(a[2], b[0])), hc_dfMulF(a[3], b[1]));
    HCDf r3 = hc_dfAdd(hc_dfSub(hc_dfAdd(hc_dfMulF(a[0], b[3]), hc_dfMulF(a[1], b[2])),
                                hc_dfMulF(a[2], b[1])), hc_dfMulF(a[3], b[0]));
    res[0] = r0; res[1] = r1; res[2] = r2; res[3] = r3;
}
inline void hc_dfRotRow3(thread const HCDf *q, thread HCDf *row) {
    row[0] = hc_dfMulF(hc_dfSub(hc_dfMul(q[1], q[3]), hc_dfMul(q[0], q[2])), 2.0f);
    row[1] = hc_dfMulF(hc_dfAdd(hc_dfMul(q[2], q[3]), hc_dfMul(q[0], q[1])), 2.0f);
    row[2] = hc_dfAdd(hc_dfSub(hc_dfSub(hc_dfMul(q[0], q[0]), hc_dfMul(q[1], q[1])),
                               hc_dfMul(q[2], q[2])), hc_dfMul(q[3], q[3]));
}
inline void hc_capsuleDistsDf(thread const HCState &s, thread float *distOut) {
    HCDf q[8][4];
    HCDf z[8];
    q[0][0] = hc_dfMake(1.0f); q[0][1] = hc_dfMake(0.0f);
    q[0][2] = hc_dfMake(0.0f); q[0][3] = hc_dfMake(0.0f);
    z[0] = hc_dfMake(0.0f);
    for (int i = 1; i < 8; i++) {
        int pid = HC_BODY_PARENT[i];
        HCDf zi;
        if (pid == 0) {
            zi = hc_dfMake(HC_BODY_POS[3*i + 2]);
            for (int k = 0; k < 4; k++) q[i][k] = q[0][k];
        } else {
            HCDf row[3];
            hc_dfRotRow3(q[pid], row);
            zi = z[pid];
            for (int k = 0; k < 3; k++)
                zi = hc_dfAdd(zi, hc_dfMulF(row[k], HC_BODY_POS[3*i + k]));
            for (int k = 0; k < 4; k++) q[i][k] = q[pid][k];
        }
        for (int j = 0; j < 9; j++) {
            if (HC_JNT_BODY[j] != i) continue;
            if (HC_JNT_TYPE[j] == 2) {
                HCDf row[3];
                hc_dfRotRow3(q[i], row);
                HCDf axz = hc_dfMake(0.0f);
                for (int k = 0; k < 3; k++)
                    axz = hc_dfAdd(axz, hc_dfMulF(row[k], HC_JNT_AXIS[3*j + k]));
                zi = hc_dfAdd(zi, hc_dfMulF(axz, s.qpos[j] - HC_QPOS0[j]));
            } else {
                float a = s.qpos[j] - HC_QPOS0[j];
                float qloc[4];
                if (a == 0.0f) {
                    qloc[0] = 1.0f; qloc[1] = 0.0f; qloc[2] = 0.0f; qloc[3] = 0.0f;
                } else {
                    float half_ = a*0.5f;
                    float sn = sin(half_);
                    qloc[0] = cos(half_);
                    qloc[1] = HC_JNT_AXIS[3*j + 0]*sn;
                    qloc[2] = HC_JNT_AXIS[3*j + 1]*sn;
                    qloc[3] = HC_JNT_AXIS[3*j + 2]*sn;
                }
                HCDf qn[4];
                hc_dfQuatMulF(q[i], qloc, qn);
                for (int k = 0; k < 4; k++) q[i][k] = qn[k];
            }
        }
        HCDf n2 = hc_dfMake(0.0f);
        for (int k = 0; k < 4; k++) n2 = hc_dfAdd(n2, hc_dfMul(q[i][k], q[i][k]));
        HCDf rinv = hc_dfRsqrt(n2);
        for (int k = 0; k < 4; k++) q[i][k] = hc_dfMul(q[i][k], rinv);
        z[i] = zi;
    }
    for (int g = 1; g < 9; g++) {
        int b = HC_GEOM_BODY[g];
        HCDf row[3];
        hc_dfRotRow3(q[b], row);
        HCDf zg = z[b];
        for (int k = 0; k < 3; k++)
            zg = hc_dfAdd(zg, hc_dfMulF(row[k], HC_GEOM_POS[3*g + k]));
        float gq[4] = { HC_GEOM_QUAT[4*g], HC_GEOM_QUAT[4*g + 1], HC_GEOM_QUAT[4*g + 2], HC_GEOM_QUAT[4*g + 3] };
        HCDf qg[4];
        hc_dfQuatMulF(q[b], gq, qg);
        HCDf grow[3];
        hc_dfRotRow3(qg, grow);
        float hl = HC_GEOM_SIZE[3*g + 1];
        float r = HC_GEOM_SIZE[3*g + 0];
        HCDf segz = hc_dfMulF(grow[2], hl);
        HCDf dPlus = hc_dfSub(hc_dfAdd(zg, segz), hc_dfMake(r));
        HCDf dMinus = hc_dfSub(hc_dfSub(zg, segz), hc_dfMake(r));
        distOut[2*(g - 1) + 0] = dPlus.hi + dPlus.lo;
        distOut[2*(g - 1) + 1] = dMinus.hi + dMinus.lo;
    }
}

inline void hc_collision(thread const HCState &s, thread HCWork &w,
                         thread HCContacts &con) {
    float distDf[16];
    hc_capsuleDistsDf(s, distDf);
    w.ncon = 0;
    thread const float *n = w.geom_xaxis;
    thread const float *ppos = w.geom_xpos;
    for (int g = 1; g < 9; g++) {
        float r = HC_GEOM_SIZE[3*g + 0];
        float hl = HC_GEOM_SIZE[3*g + 1];
        thread const float *cpos = w.geom_xpos + 3*g;
        float dvec[3] = { cpos[0] - ppos[0], cpos[1] - ppos[1], cpos[2] - ppos[2] };
        float pdist = hc_dot3(n, dvec);
        if (pdist > r + hl) continue;
        thread const float *axis = w.geom_xaxis + 3*g;
        float seg[3] = { hl*axis[0], hl*axis[1], hl*axis[2] };
        for (int e = 0; e < 2; e++) {
            float sgn = (e == 0) ? 1.0f : -1.0f;
            float center[3] = { cpos[0] + sgn*seg[0], cpos[1] + sgn*seg[1], cpos[2] + sgn*seg[2] };
            float dist = distDf[2*(g - 1) + e];
            if (dist > 0.0f) continue;
            int c = w.ncon;
            w.con_dist[c] = dist;
            float scl = dist*0.5f + r;
            con.con_pos[3*c + 0] = center[0] - n[0]*scl;
            con.con_pos[3*c + 1] = center[1] - n[1]*scl;
            con.con_pos[3*c + 2] = center[2] - n[2]*scl;
            float fr[9] = { n[0], n[1], n[2], axis[0], axis[1], axis[2], 0.0f, 0.0f, 0.0f };
            hc_makeFrame(fr);
            for (int k = 0; k < 9; k++) con.con_frame[9*c + k] = fr[k];
            con.con_exclude[c] = (dist >= 0.0f) ? 1 : 0;
            con.con_geom[c] = g;
            w.ncon = c + 1;
        }
    }
}

inline float hc_getimpedance(float pos, float d0, float d1, float width, float mid) {
    if (d0 == d1 || width <= HC_MINVAL) return d0;
    float x = pos / width;
    x = fabs(x);
    if (x >= 1.0f) return d1;
    if (x <= 0.0f) return d0;
    float y;
    if (x <= mid) {
        float a = 1.0f / mid;
        y = a*x*x;
    } else {
        float b = 1.0f / (1.0f - mid);
        y = 1.0f - b*(1.0f - x)*(1.0f - x);
    }
    return d0 + y*(d1 - d0);
}

inline void hc_makeConstraint(thread const HCState &s, thread HCWork &w,
                              thread const HCContacts &con, thread HCRefCoef &ref) {
    int nefc = 0;
    for (int j = 0; j < 9; j++) {
        if (!HC_JNT_LIMITED[j]) continue;
        float v = s.qpos[j];
        for (int side = -1; side <= 1; side += 2) {
            float dist = (side == -1) ? (v - HC_JNT_RANGE[2*j + 0])
                                      : (HC_JNT_RANGE[2*j + 1] - v);
            if (dist < HC_JNT_MARGIN[j]) {
                for (int k = 0; k < 9; k++) w.efc_J[9*nefc + k] = 0.0f;
                w.efc_J[9*nefc + j] = (float)(-side);
                w.efc_pos[nefc] = dist;
                float dA = HC_DOF_INVWEIGHT0[j];
                float solref0 = max(HC_JNT_SOLREF[2*j + 0], 2.0f*HC_TIMESTEP);
                float solref1 = HC_JNT_SOLREF[2*j + 1];
                float d0 = clamp(HC_JNT_SOLIMP[5*j + 0], HC_MINIMP, HC_MAXIMP);
                float d1 = clamp(HC_JNT_SOLIMP[5*j + 1], HC_MINIMP, HC_MAXIMP);
                float width = HC_JNT_SOLIMP[5*j + 2];
                float mid = HC_JNT_SOLIMP[5*j + 3];
                float imp = hc_getimpedance(dist, d0, d1, width, mid);
                float R = max(HC_MINVAL, (1.0f - imp)*dA/imp);
                ref.efc_imp[nefc] = imp;
                ref.efc_K[nefc] = 1.0f / max(HC_MINVAL, d1*d1*solref0*solref0*solref1*solref1);
                ref.efc_B[nefc] = 2.0f / max(HC_MINVAL, d1*solref0);
                w.efc_D[nefc] = 1.0f / R;
                nefc++;
            }
        }
    }
    for (int c = 0; c < w.ncon; c++) {
        if (con.con_exclude[c]) continue;
        int g = con.con_geom[c];
        int b = HC_GEOM_BODY[g];
        float fri[3];
        for (int k = 0; k < 3; k++)
            fri[k] = max(HC_GEOM_FRICTION[0 + k], HC_GEOM_FRICTION[3*g + k]);
        float friction0 = max(HC_MINMU, fri[0]);
        float friction1 = max(HC_MINMU, fri[0]);
        float solref0 = 0.5f*HC_GEOM_SOLREF[0] + 0.5f*HC_GEOM_SOLREF[2*g + 0];
        float solref1 = 0.5f*HC_GEOM_SOLREF[1] + 0.5f*HC_GEOM_SOLREF[2*g + 1];
        float solimp[5];
        for (int k = 0; k < 5; k++)
            solimp[k] = 0.5f*HC_GEOM_SOLIMP[k] + 0.5f*HC_GEOM_SOLIMP[5*g + k];
        solref0 = max(solref0, 2.0f*HC_TIMESTEP);
        float d0 = clamp(solimp[0], HC_MINIMP, HC_MAXIMP);
        float d1 = clamp(solimp[1], HC_MINIMP, HC_MAXIMP);
        float jac[27];
        for (int k = 0; k < 27; k++) jac[k] = 0.0f;
        float off[3] = { con.con_pos[3*c] - w.com1[0], con.con_pos[3*c + 1] - w.com1[1], con.con_pos[3*c + 2] - w.com1[2] };
        int i = HC_BODY_LASTDOF[b];
        while (i >= 0) {
            thread const float *cd = w.cdof + 6*i;
            jac[0*9 + i] = cd[3] + (cd[1]*off[2] - cd[2]*off[1]);
            jac[1*9 + i] = cd[4] + (cd[2]*off[0] - cd[0]*off[2]);
            jac[2*9 + i] = cd[5] + (cd[0]*off[1] - cd[1]*off[0]);
            i = HC_DOF_PARENT[i];
        }
        float jacC[27];
        thread const float *fr = con.con_frame + 9*c;
        for (int rr = 0; rr < 3; rr++)
            for (int col = 0; col < 9; col++)
                jacC[9*rr + col] = fr[3*rr + 0]*jac[0 + col] + fr[3*rr + 1]*jac[9 + col]
                                 + fr[3*rr + 2]*jac[18 + col];
        int first = nefc;
        for (int k = 1; k <= 2; k++) {
            float mu = (k == 1) ? friction0 : friction1;
            for (int sgn = 0; sgn < 2; sgn++) {
                float sg = (sgn == 0) ? 1.0f : -1.0f;
                for (int col = 0; col < 9; col++)
                    w.efc_J[9*nefc + col] = jacC[col] + sg*mu*jacC[9*k + col];
                w.efc_pos[nefc] = w.con_dist[c];
                nefc++;
            }
        }
        float tran = HC_BODY_INVWEIGHT0[0] + HC_BODY_INVWEIGHT0[2*b + 0];
        float dA0 = tran + friction0*friction0*tran;
        float imp = hc_getimpedance(w.con_dist[c], d0, d1, solimp[2], solimp[3]);
        float K = 1.0f / max(HC_MINVAL, d1*d1*solref0*solref0*solref1*solref1);
        float B = 2.0f / max(HC_MINVAL, d1*solref0);
        float R0 = max(HC_MINVAL, (1.0f - imp)*dA0/imp);
        float R1 = R0 / max(HC_MINVAL, 1.0f);
        float conMu = friction0 * sqrt(R1/R0);
        float Rpy = 2.0f*conMu*conMu*R0;
        for (int row = first; row < first + 4; row++) {
            ref.efc_imp[row] = imp;
            ref.efc_K[row] = K;
            ref.efc_B[row] = B;
            w.efc_D[row] = 1.0f / Rpy;
        }
    }
    w.nefc = nefc;
}

inline void hc_comVel(thread const HCState &s, thread HCWork &w) {
    for (int k = 0; k < 6; k++) w.cvel[k] = 0.0f;
    for (int i = 1; i < 8; i++) {
        int p = HC_BODY_PARENT[i];
        float v6[6];
        for (int k = 0; k < 6; k++) v6[k] = w.cvel[6*p + k];
        int bda = HC_BODY_DOFADR[i];
        int nn = HC_BODY_DOFNUM[i];
        for (int k = 0; k < nn; k++) {
            int j = bda + k;
            hc_crossMotion(w.cdofdot + 6*j, v6, w.cdof + 6*j);
            for (int t = 0; t < 6; t++) v6[t] += w.cdof[6*j + t] * s.qvel[j];
        }
        for (int k = 0; k < 6; k++) w.cvel[6*i + k] = v6[k];
    }
}

inline void hc_rne(thread const HCState &s, thread HCWork &w) {
    float cacc[48], cfrc[48];
    cacc[0] = 0.0f; cacc[1] = 0.0f; cacc[2] = 0.0f;
    cacc[3] = 0.0f; cacc[4] = 0.0f; cacc[5] = -HC_GRAVITY_Z;
    for (int k = 0; k < 6; k++) cfrc[k] = 0.0f;
    for (int i = 1; i < 8; i++) {
        int p = HC_BODY_PARENT[i];
        int bda = HC_BODY_DOFADR[i];
        int nn = HC_BODY_DOFNUM[i];
        float acc[6];
        for (int k = 0; k < 6; k++) acc[k] = cacc[6*p + k];
        for (int k = 0; k < nn; k++) {
            int j = bda + k;
            for (int t = 0; t < 6; t++) acc[t] += w.cdofdot[6*j + t] * s.qvel[j];
        }
        for (int k = 0; k < 6; k++) cacc[6*i + k] = acc[k];
        float tmp[6], tmp1[6], cf[6];
        hc_mulInertVec(tmp, w.cinert + 10*i, acc);
        hc_mulInertVec(tmp1, w.cinert + 10*i, w.cvel + 6*i);
        hc_crossForce(cf, w.cvel + 6*i, tmp1);
        for (int k = 0; k < 6; k++) cfrc[6*i + k] = tmp[k] + cf[k];
    }
    for (int i = 7; i >= 1; i--) {
        int p = HC_BODY_PARENT[i];
        if (p != 0)
            for (int k = 0; k < 6; k++) cfrc[6*p + k] += cfrc[6*i + k];
    }
    for (int v = 0; v < 9; v++)
        w.qfrc_bias[v] = hc_dot6(w.cdof + 6*v, cfrc + 6*HC_DOF_BODY[v]);
}

inline float hc_constraintUpdate(int nefc, thread const float *Jaref, thread const float *D,
                                 thread float *force, thread int *st) {
    HCAcc cost;
    hc_accInit(cost, 0.0f);
    for (int i = 0; i < nefc; i++) {
        float jar = Jaref[i];
        if (jar >= 0.0f) {
            force[i] = 0.0f;
            st[i] = 0;
        } else {
            force[i] = -D[i]*jar;
            st[i] = 1;
            hc_accMulAdd(cost, 0.5f*D[i]*jar, jar);
        }
    }
    return hc_accGet(cost);
}
inline void hc_JTforce(int nefc, thread const float *J, thread const float *force, thread float *res) {
    HCAcc acc[9];
    for (int v = 0; v < 9; v++) hc_accInit(acc[v], 0.0f);
    for (int r = 0; r < nefc; r++) {
        float f = force[r];
        if (f != 0.0f)
            for (int v = 0; v < 9; v++) hc_accMulAdd(acc[v], J[9*r + v], f);
    }
    for (int v = 0; v < 9; v++) res[v] = hc_accGet(acc[v]);
}
inline float hc_gaussCost(thread const HCWork &w, thread const float *Ma, thread const float *qacc) {
    HCAcc g;
    hc_accInit(g, 0.0f);
    for (int i = 0; i < 9; i++)
        hc_accMulAdd(g, 0.5f*(Ma[i] - w.qfrc_smooth[i]), qacc[i] - w.qacc_smooth[i]);
    return hc_accGet(g);
}

inline void hc_cholFactor(thread float *L) {
    for (int j = 0; j < 9; j++) {
        float t = L[j*9 + j];
        for (int k = 0; k < j; k++) t -= L[j*9 + k]*L[j*9 + k];
        if (t < HC_MINVAL) t = HC_MINVAL;
        L[j*9 + j] = sqrt(t);
        float inv = 1.0f / L[j*9 + j];
        for (int i = j + 1; i < 9; i++) {
            float v = L[i*9 + j];
            for (int k = 0; k < j; k++) v -= L[i*9 + k]*L[j*9 + k];
            L[i*9 + j] = v*inv;
        }
    }
}
inline void hc_makeHessian(thread const HCWork &w, thread const int *st, thread float *L) {
    for (int k = 0; k < 81; k++) L[k] = 0.0f;
    for (int r = 0; r < w.nefc; r++) {
        if (st[r] != 1) continue;
        float Dj = w.efc_D[r];
        for (int i = 0; i < 9; i++) {
            float t = w.efc_J[9*r + i];
            if (t == 0.0f) continue;
            float sc = t*Dj;
            for (int c = 0; c <= i; c++) L[i*9 + c] += w.efc_J[9*r + c]*sc;
        }
    }
    for (int i = 0; i < 9; i++) {
        int adr = HC_M_ROWADR[i];
        for (int k = 0; k < HC_M_ROWNNZ[i]; k++)
            L[i*9 + HC_M_COLIND[adr + k]] += w.M[adr + k];
    }
    hc_cholFactor(L);
}
inline void hc_cholSolve(thread const float *L, thread const float *x, thread float *res) {
    for (int i = 0; i < 9; i++) {
        float t = x[i];
        for (int j = 0; j < i; j++) t -= L[i*9 + j]*res[j];
        res[i] = t / L[i*9 + i];
    }
    for (int i = 8; i >= 0; i--) {
        float t = res[i];
        for (int j = i + 1; j < 9; j++) t -= L[j*9 + i]*res[j];
        res[i] = t / L[i*9 + i];
    }
}
inline int hc_cholUpdate(thread float *L, thread float *vec, int flgPlus) {
    int rank = 9;
    for (int i = 0; i < 9; i++) {
        if (vec[i] != 0.0f) {
            float Lii = L[i*9 + i];
            float t = Lii*Lii + (flgPlus ? vec[i]*vec[i] : -(vec[i]*vec[i]));
            if (t < HC_CHOL_EPS) { t = HC_CHOL_EPS; rank--; }
            float r = sqrt(t);
            float c = r / Lii;
            float sf = vec[i] / Lii;
            L[i*9 + i] = r;
            if (flgPlus) {
                for (int j = i + 1; j < 9; j++) L[j*9 + i] = (L[j*9 + i] + sf*vec[j]) / c;
            } else {
                for (int j = i + 1; j < 9; j++) L[j*9 + i] = (L[j*9 + i] - sf*vec[j]) / c;
            }
            for (int j = i + 1; j < 9; j++) vec[j] = c*vec[j] - sf*L[j*9 + i];
        }
    }
    return rank;
}

struct HCLS { float alpha, cost, deriv0, deriv1; };

inline HCLS hc_lsEval(float alpha, int nefc, thread const float *Jaref, thread const float *Jv,
                      thread const float *quad, thread const float *quadGauss, thread int &lsiter) {
    HCAcc a0, a1, a2;
    hc_accInit(a0, quadGauss[0]);
    hc_accInit(a1, quadGauss[1]);
    hc_accInit(a2, quadGauss[2]);
    for (int i = 0; i < nefc; i++) {
        float x = Jaref[i] + alpha*Jv[i];
        if (x < 0.0f) {
            hc_accAdd(a0, quad[3*i]);
            hc_accAdd(a1, quad[3*i + 1]);
            hc_accAdd(a2, quad[3*i + 2]);
        }
    }
    float q0 = hc_accGet(a0), q1 = hc_accGet(a1), q2 = hc_accGet(a2);
    HCLS p;
    p.alpha = alpha;
    p.cost = q2*alpha*alpha + q1*alpha + q0;
    p.deriv0 = 2.0f*q2*alpha + q1;
    p.deriv1 = 2.0f*q2;
    if (p.deriv1 <= 0.0f) p.deriv1 = HC_MINVAL;
    lsiter++;
    return p;
}

inline bool hc_updateBracket(thread HCLS &p, thread const HCLS *cand, thread HCLS &pnext,
                             int nefc, thread const float *Jaref, thread const float *Jv,
                             thread const float *quad, thread const float *quadGauss,
                             thread int &lsiter) {
    bool flag = false;
    for (int k = 0; k < 3; k++) {
        if (p.deriv0 < 0.0f && cand[k].deriv0 < 0.0f && cand[k].deriv0 > p.deriv0) {
            p = cand[k]; flag = true;
        } else if (p.deriv0 > 0.0f && cand[k].deriv0 > 0.0f && cand[k].deriv0 < p.deriv0) {
            p = cand[k]; flag = true;
        }
    }
    if (flag)
        pnext = hc_lsEval(p.alpha - p.deriv0/p.deriv1, nefc, Jaref, Jv, quad, quadGauss, lsiter);
    return flag;
}

inline float hc_primalSearch(int nefc, thread const float *Jaref, thread const float *Jv,
                             thread const float *quad, thread const float *quadGauss,
                             float snorm, float scale) {
    if (snorm < HC_MINVAL) return 0.0f;
    float gtol = HC_SOLVER_TOL * HC_LS_TOL * snorm / scale;
    int lsiter = 0;
    HCLS p0 = hc_lsEval(0.0f, nefc, Jaref, Jv, quad, quadGauss, lsiter);
    HCLS p1 = hc_lsEval(p0.alpha - p0.deriv0/p0.deriv1, nefc, Jaref, Jv, quad, quadGauss, lsiter);
    if (fabs(p1.deriv0) < gtol) return p1.alpha;
    float dir = (p1.deriv0 < 0.0f) ? 1.0f : -1.0f;
    HCLS p2 = p0;
    while (p1.deriv0*dir <= -gtol && lsiter < HC_LS_ITER) {
        p2 = p1;
        p1 = hc_lsEval(p1.alpha - p1.deriv0/p1.deriv1, nefc, Jaref, Jv, quad, quadGauss, lsiter);
        if (fabs(p1.deriv0) < gtol) return p1.alpha;
    }
    if (lsiter >= HC_LS_ITER) return p1.alpha;
    HCLS p2next = p1;
    HCLS p1next = hc_lsEval(p1.alpha - p1.deriv0/p1.deriv1, nefc, Jaref, Jv, quad, quadGauss, lsiter);
    while (lsiter < HC_LS_ITER) {
        HCLS pmid = hc_lsEval(0.5f*(p1.alpha + p2.alpha), nefc, Jaref, Jv, quad, quadGauss, lsiter);
        HCLS cand[3] = { p1next, p2next, pmid };
        bool any = false;
        HCLS best = pmid;
        for (int k = 0; k < 3; k++) {
            if (fabs(cand[k].deriv0) < gtol && (!any || cand[k].cost < best.cost)) {
                best = cand[k];
                any = true;
            }
        }
        if (any) return best.alpha;
        bool b1 = hc_updateBracket(p1, cand, p1next, nefc, Jaref, Jv, quad, quadGauss, lsiter);
        bool b2 = hc_updateBracket(p2, cand, p2next, nefc, Jaref, Jv, quad, quadGauss, lsiter);
        if (!b1 && !b2) return pmid.alpha;
    }
    if (p1.cost <= p2.cost && p1.cost < p0.cost) return p1.alpha;
    if (p2.cost <= p1.cost && p2.cost < p0.cost) return p2.alpha;
    return 0.0f;
}

inline void hc_newton(thread HCWork &w) {
    int nefc = w.nefc;
    float Ma[9];
    hc_mulSymVecSparse(w.M, w.qacc, Ma);
    float Jaref[HC_NEFC_MAX];
    for (int i = 0; i < nefc; i++) Jaref[i] = hc_dot9c(w.efc_J + 9*i, w.qacc, -w.efc_aref[i]);
    float force[HC_NEFC_MAX];
    int st[HC_NEFC_MAX], oldst[HC_NEFC_MAX];
    float costC = hc_constraintUpdate(nefc, Jaref, w.efc_D, force, st);
    hc_JTforce(nefc, w.efc_J, force, w.qfrc_constraint);
    float gauss = hc_gaussCost(w, Ma, w.qacc);
    float cost = costC + gauss;
    float L[81];
    hc_makeHessian(w, st, L);
    float grad[9], Mgrad[9], search[9];
    for (int v = 0; v < 9; v++) grad[v] = Ma[v] - w.qfrc_smooth[v] - w.qfrc_constraint[v];
    hc_cholSolve(L, grad, Mgrad);
    for (int v = 0; v < 9; v++) search[v] = -Mgrad[v];

    int iter = 0;
    while (iter < HC_SOLVER_ITER) {
        float snorm = sqrt(hc_dot9(search, search));
        float Mv[9];
        hc_mulSymVecSparse(w.M, search, Mv);
        float Jv[HC_NEFC_MAX];
        for (int i = 0; i < nefc; i++) Jv[i] = hc_dot9c(w.efc_J + 9*i, search, 0.0f);
        float quadGauss[3];
        quadGauss[0] = gauss;
        {
            HCAcc aq;
            hc_accInit(aq, 0.0f);
            for (int v = 0; v < 9; v++) hc_accMulAdd(aq, search[v], Ma[v]);
            for (int v = 0; v < 9; v++) hc_accMulAdd(aq, -w.qfrc_smooth[v], search[v]);
            quadGauss[1] = hc_accGet(aq);
        }
        quadGauss[2] = 0.5f*hc_dot9c(search, Mv, 0.0f);
        float quad[3*HC_NEFC_MAX];
        for (int i = 0; i < nefc; i++) {
            float Dj = w.efc_D[i];
            quad[3*i + 0] = 0.5f*Dj*Jaref[i]*Jaref[i];
            quad[3*i + 1] = Dj*Jaref[i]*Jv[i];
            quad[3*i + 2] = 0.5f*Dj*Jv[i]*Jv[i];
        }
        float alpha = hc_primalSearch(nefc, Jaref, Jv, quad, quadGauss, snorm, w.scale);
        if (alpha == 0.0f) break;
        for (int v = 0; v < 9; v++) {
            w.qacc[v] += alpha*search[v];
            Ma[v] += alpha*Mv[v];
        }
        for (int i = 0; i < nefc; i++) Jaref[i] += alpha*Jv[i];
        for (int i = 0; i < nefc; i++) oldst[i] = st[i];
        float oldcost = cost;
        costC = hc_constraintUpdate(nefc, Jaref, w.efc_D, force, st);
        hc_JTforce(nefc, w.efc_J, force, w.qfrc_constraint);
        gauss = hc_gaussCost(w, Ma, w.qacc);
        cost = costC + gauss;
        for (int i = 0; i < nefc; i++) {
            int up;
            if (oldst[i] != 1 && st[i] == 1) up = 1;
            else if (oldst[i] == 1 && st[i] != 1) up = 0;
            else continue;
            float sd = sqrt(w.efc_D[i]);
            float vec[9];
            for (int v = 0; v < 9; v++) vec[v] = w.efc_J[9*i + v]*sd;
            int rank = hc_cholUpdate(L, vec, up);
            if (rank < 9) {
                hc_makeHessian(w, st, L);
                break;
            }
        }
        for (int v = 0; v < 9; v++) grad[v] = Ma[v] - w.qfrc_smooth[v] - w.qfrc_constraint[v];
        hc_cholSolve(L, grad, Mgrad);
        float improvement = w.scale*(oldcost - cost);
        float gradient = w.scale*sqrt(hc_dot9(grad, grad));
        iter++;
        if (improvement < HC_SOLVER_TOL || gradient < HC_SOLVER_TOL) break;
        for (int v = 0; v < 9; v++) search[v] = -Mgrad[v];
    }
}

inline void hc_fwdConstraint(thread const HCState &s, thread HCWork &w) {
    for (int v = 0; v < 9; v++) w.qfrc_constraint[v] = 0.0f;
    if (w.nefc == 0) {
        for (int v = 0; v < 9; v++) w.qacc[v] = w.qacc_smooth[v];
        return;
    }
    int nefc = w.nefc;
    float jar[HC_NEFC_MAX], forceTmp[HC_NEFC_MAX];
    int stTmp[HC_NEFC_MAX];
    for (int i = 0; i < nefc; i++)
        jar[i] = hc_dot9c(w.efc_J + 9*i, s.qacc_warmstart, -w.efc_aref[i]);
    float costWs = hc_constraintUpdate(nefc, jar, w.efc_D, forceTmp, stTmp);
    float Maw[9];
    {
        float wsLocal[9];
        for (int v = 0; v < 9; v++) wsLocal[v] = s.qacc_warmstart[v];
        hc_mulSymVecSparse(w.M, wsLocal, Maw);
        HCAcc aw;
        hc_accInit(aw, costWs);
        for (int v = 0; v < 9; v++)
            hc_accMulAdd(aw, 0.5f*(Maw[v] - w.qfrc_smooth[v]), wsLocal[v] - w.qacc_smooth[v]);
        costWs = hc_accGet(aw);
    }
    float efcB[HC_NEFC_MAX];
    for (int i = 0; i < nefc; i++)
        efcB[i] = hc_dot9c(w.efc_J + 9*i, w.qacc_smooth, -w.efc_aref[i]);
    float costSmooth = hc_constraintUpdate(nefc, efcB, w.efc_D, forceTmp, stTmp);
    if (costWs > costSmooth) {
        for (int v = 0; v < 9; v++) w.qacc[v] = w.qacc_smooth[v];
    } else {
        for (int v = 0; v < 9; v++) w.qacc[v] = s.qacc_warmstart[v];
    }
    hc_newton(w);
}

inline void hc_forward(thread const HCState &s, thread HCWork &w) {
    hc_kinematics(s, w);
    HCContacts con;
    HCRefCoef ref;
    // collision depends only on kinematics outputs (geom_xpos/geom_xaxis) and
    // s.qpos; it is writes-disjoint with comPos/comVel/rne/makeM/factorI, so
    // the Df contact chain issues ahead of them (ILP statement motion).
    hc_collision(s, w, con);
    hc_comPos(w);
    // ILP reorder (writes-disjoint, bit-identical statement motion): comVel
    // needs only comPos's cdof (+qvel) and writes cvel/cdofdot; rne consumes
    // those plus comPos's cinert and writes only qfrc_bias. Neither touches
    // M/con/efc_*, so both hoist above makeM/collision.
    hc_comVel(s, w);
    hc_rne(s, w);
    hc_makeM(w);
    for (int k = 0; k < 36; k++) w.qLD[k] = w.M[k];
    hc_factorI(w.qLD, w.qLDiagInv);
    hc_makeConstraint(s, w, con, ref);
    {
        float tr = 0.0f;
        for (int i = 0; i < 9; i++) tr += w.M[HC_M_ROWADR[i] + HC_M_ROWNNZ[i] - 1];
        w.scale = 1.0f / tr;
    }

    for (int j = 0; j < 9; j++) {
        float spring = -HC_JNT_STIFFNESS[j]*(s.qpos[j] - HC_QPOS_SPRING[j]);
        float damper = -HC_DOF_DAMPING[j]*s.qvel[j];
        w.qfrc_passive[j] = spring + damper;
    }
    {
        float qvLocal[9];
        for (int v = 0; v < 9; v++) qvLocal[v] = s.qvel[v];
        for (int i = 0; i < w.nefc; i++) {
            float vel = hc_dot9c(w.efc_J + 9*i, qvLocal, 0.0f);
            w.efc_aref[i] = -ref.efc_B[i]*vel - ref.efc_K[i]*ref.efc_imp[i]*w.efc_pos[i];
        }
    }

    float qfrcAct[9];
    for (int v = 0; v < 9; v++) qfrcAct[v] = 0.0f;
    {
        float cl[6];
        bool bad = false;
        for (int u = 0; u < 6; u++) {
            cl[u] = max(HC_ACT_CTRLRANGE[2*u + 0], min(HC_ACT_CTRLRANGE[2*u + 1], s.ctrl[u]));
            if (hc_isBad(cl[u])) bad = true;
        }
        if (bad)
            for (int u = 0; u < 6; u++) cl[u] = 0.0f;
        for (int u = 0; u < 6; u++)
            qfrcAct[HC_ACT_TRNID[u]] += HC_ACT_GEAR[u]*cl[u];
    }

    for (int v = 0; v < 9; v++) {
        w.qfrc_smooth[v] = w.qfrc_passive[v] - w.qfrc_bias[v] + qfrcAct[v];
        w.qacc_smooth[v] = w.qfrc_smooth[v];
    }
    hc_solveLD(w.qacc_smooth, w.qLD, w.qLDiagInv);

    hc_fwdConstraint(s, w);
}

inline void hc_resetData(thread HCState &s) {
    for (int v = 0; v < 9; v++) {
        s.qpos[v] = HC_QPOS0[v];
        s.qvel[v] = 0.0f;
        s.qacc_warmstart[v] = 0.0f;
    }
    for (int u = 0; u < 6; u++) s.ctrl[u] = 0.0f;
}
inline bool hc_isBadVec9(thread const float *x) {
    bool bad = false;
    for (int i = 0; i < 9; i++) bad = bad || hc_isBad(x[i]);
    return bad;
}

inline void hc_mj_step(thread HCState &s) {
    if (hc_isBadVec9(s.qpos)) hc_resetData(s);
    if (hc_isBadVec9(s.qvel)) hc_resetData(s);
    HCWork w;
    hc_forward(s, w);
    if (hc_isBadVec9(w.qacc)) {
        hc_resetData(s);
        hc_forward(s, w);
    }
    float qH[36], qHDiagInv[9];
    for (int k = 0; k < 36; k++) qH[k] = w.M[k];
    for (int i = 0; i < 9; i++)
        qH[HC_M_ROWADR[i] + HC_M_ROWNNZ[i] - 1] += HC_TIMESTEP*HC_DOF_DAMPING[i];
    hc_factorI(qH, qHDiagInv);
    float qaccInt[9];
    for (int v = 0; v < 9; v++) qaccInt[v] = w.qfrc_smooth[v] + w.qfrc_constraint[v];
    hc_solveLD(qaccInt, qH, qHDiagInv);
    for (int v = 0; v < 9; v++) s.qvel[v] += HC_TIMESTEP*qaccInt[v];
    for (int v = 0; v < 9; v++) s.qpos[v] += HC_TIMESTEP*s.qvel[v];
    for (int v = 0; v < 9; v++) s.qacc_warmstart[v] = w.qacc[v];
}

kernel void oak_physics_probe(device const float *in [[buffer(0)]],
                              device float *out [[buffer(1)]],
                              constant uint &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    device const float *p = in + gid*33;
    HCState s;
    for (int i = 0; i < 9; i++) s.qpos[i] = p[i];
    for (int i = 0; i < 9; i++) s.qvel[i] = p[9 + i];
    for (int i = 0; i < 6; i++) s.ctrl[i] = p[18 + i];
    for (int i = 0; i < 9; i++) s.qacc_warmstart[i] = p[24 + i];
    hc_mj_step(s);
    device float *o = out + gid*18;
    for (int i = 0; i < 9; i++) o[i] = s.qpos[i];
    for (int i = 0; i < 9; i++) o[9 + i] = s.qvel[i];
}

kernel void oak_physics_probe_debug(device const float *in [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    constant uint &count [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    device const float *p = in + gid*33;
    HCState s;
    for (int i = 0; i < 9; i++) s.qpos[i] = p[i];
    for (int i = 0; i < 9; i++) s.qvel[i] = p[9 + i];
    for (int i = 0; i < 6; i++) s.ctrl[i] = p[18 + i];
    for (int i = 0; i < 9; i++) s.qacc_warmstart[i] = p[24 + i];
    HCWork w;
    hc_forward(s, w);
    device float *o = out + gid*904;
    for (int i = 0; i < 904; i++) o[i] = 0.0f;
    for (int i = 0; i < 9; i++) o[i] = w.qacc_smooth[i];
    for (int i = 0; i < 9; i++) o[9 + i] = w.qacc[i];
    for (int i = 0; i < 9; i++) o[18 + i] = w.qfrc_constraint[i];
    for (int i = 0; i < 9; i++) o[27 + i] = w.qfrc_smooth[i];
    for (int i = 0; i < 9; i++) o[36 + i] = w.qfrc_bias[i];
    o[45] = (float)w.ncon;
    o[46] = (float)w.nefc;
    for (int i = 0; i < w.ncon; i++) o[48 + i] = w.con_dist[i];
    for (int i = 0; i < w.nefc; i++) o[64 + i] = w.efc_pos[i];
    for (int i = 0; i < w.nefc; i++) o[134 + i] = w.efc_D[i];
    for (int i = 0; i < w.nefc; i++) o[204 + i] = w.efc_aref[i];
    for (int i = 0; i < 9*w.nefc; i++) o[274 + i] = w.efc_J[i];
}

constant uint HC_PHILOX_M0 = 0xD2511F53u;
constant uint HC_PHILOX_M1 = 0xCD9E8D57u;
constant uint HC_PHILOX_W0 = 0x9E3779B9u;
constant uint HC_PHILOX_W1 = 0xBB67AE85u;
constant uint HC_RNG_STREAM_RESET = 0u;
constant uint HC_RNG_STREAM_CTRL = 1u;

#pragma METAL fp math_mode(safe)
inline uint4 hc_philox4x32(uint2 key, uint4 ctr) {
    for (int r = 0; r < 10; r++) {
        if (r > 0) { key.x += HC_PHILOX_W0; key.y += HC_PHILOX_W1; }
        uint hi0 = mulhi(HC_PHILOX_M0, ctr.x), lo0 = HC_PHILOX_M0 * ctr.x;
        uint hi1 = mulhi(HC_PHILOX_M1, ctr.z), lo1 = HC_PHILOX_M1 * ctr.z;
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    }
    return ctr;
}

inline float hc_u01(uint x) { return float(x >> 8) * 0x1p-24f; }
inline float hc_u01open(uint x) { return (float(x >> 8) + 1.0f) * 0x1p-24f; }
inline float hc_uniform(uint x, float lo, float hi) { return lo + (hi - lo)*hc_u01(x); }

inline void hc_rng_reset_noise(uint laneSeed, uint resetIdx,
                               thread float *qposNoise, thread float *qvelNoise) {
    uint2 key = uint2(laneSeed, HC_RNG_STREAM_RESET);
    uint raw[12];
    for (uint b = 0; b < 3; b++) {
        uint4 r = hc_philox4x32(key, uint4(resetIdx, b, 0u, 0u));
        raw[4*b + 0] = r.x; raw[4*b + 1] = r.y; raw[4*b + 2] = r.z; raw[4*b + 3] = r.w;
    }
    for (int i = 0; i < 9; i++)
        qposNoise[i] = hc_uniform(raw[i], -HC_RESET_NOISE, HC_RESET_NOISE);
    for (uint b = 0; b < 3; b++) {
        uint4 r = hc_philox4x32(key, uint4(resetIdx, 3u + b, 0u, 0u));
        raw[4*b + 0] = r.x; raw[4*b + 1] = r.y; raw[4*b + 2] = r.z; raw[4*b + 3] = r.w;
    }
    float z[12];
    for (int p = 0; p < 6; p++) {
        float u1 = hc_u01open(raw[2*p]);
        float u2 = hc_u01(raw[2*p + 1]);
        float rad = sqrt(-2.0f * log(u1));
        float ang = 2.0f * M_PI_F * u2;
        z[2*p] = rad * cos(ang);
        z[2*p + 1] = rad * sin(ang);
    }
    for (int i = 0; i < 9; i++) qvelNoise[i] = HC_RESET_NOISE * z[i];
}

inline void hc_rng_ctrl(uint laneSeed, uint stepIdx, thread float *ctrl) {
    uint2 key = uint2(laneSeed, HC_RNG_STREAM_CTRL);
    uint4 r0 = hc_philox4x32(key, uint4(stepIdx, 0u, 0u, 0u));
    uint4 r1 = hc_philox4x32(key, uint4(stepIdx, 1u, 0u, 0u));
    uint raw[6] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y };
    for (int u = 0; u < 6; u++) ctrl[u] = hc_uniform(raw[u], -1.0f, 1.0f);
}

inline void hc_env_reset(thread HCState &s, uint laneSeed,
                         thread uint &elapsed, thread uint &resets) {
    hc_resetData(s);
    float qn[9], vn[9];
    hc_rng_reset_noise(laneSeed, resets, qn, vn);
    for (int v = 0; v < 9; v++) {
        s.qpos[v] = HC_QPOS0[v] + qn[v];
        s.qvel[v] = vn[v];
    }
    resets++;
    elapsed = 0u;
}

inline float hc_env_step(thread HCState &s, thread uint &elapsed, thread uint &resets,
                         uint laneSeed, uint episodeLen,
                         thread const float *ctrl, thread float *obs, thread uint &doneOut) {
    float reward;
    if (episodeLen != 0u && elapsed >= episodeLen) {
        hc_env_reset(s, laneSeed, elapsed, resets);
        reward = 0.0f;
        doneOut = 0u;
    } else {
        float xBefore = s.qpos[0];
        for (int u = 0; u < 6; u++) s.ctrl[u] = ctrl[u];
        for (int k = 0; k < HC_FRAME_SKIP; k++) hc_mj_step(s);
        float xv = (s.qpos[0] - xBefore) / (float(HC_FRAME_SKIP) * HC_TIMESTEP);
        float ctrlCost = 0.0f;
        for (int u = 0; u < 6; u++) ctrlCost += 0.1f * ctrl[u] * ctrl[u];
        reward = xv * 1.0f - ctrlCost;
        elapsed++;
        doneOut = (episodeLen != 0u && elapsed >= episodeLen) ? 1u : 0u;
    }
    for (int i = 1; i < 9; i++) obs[i - 1] = s.qpos[i];
    for (int i = 0; i < 9; i++) obs[8 + i] = s.qvel[i];
    return reward;
}

inline float2 hc_energy(thread const HCState &s) {
    HCWork w;
    hc_kinematics(s, w);
    hc_comPos(w);
    hc_makeM(w);
    float pot = 0.0f;
    for (int i = 1; i < 8; i++)
        pot += HC_BODY_MASS[i] * (-HC_GRAVITY_Z) * w.xipos[3*i + 2];
    for (int j = 0; j < 9; j++) {
        float dq = s.qpos[j] - HC_QPOS_SPRING[j];
        pot += 0.5f * HC_JNT_STIFFNESS[j] * dq * dq;
    }
    float qv[9], Mv[9];
    for (int v = 0; v < 9; v++) qv[v] = s.qvel[v];
    hc_mulSymVecSparse(w.M, qv, Mv);
    float kin = 0.5f * hc_dot9(qv, Mv);
    return float2(pot, kin);
}

struct HCEnvLane {
    HCState s;
    uint elapsed;
    uint resets;
};

kernel void oak_physics_rollout(device const uint *hdr        [[buffer(0)]],
                                device const float *init      [[buffer(1)]],
                                device const float *ctrlSeq   [[buffer(2)]],
                                device float *out             [[buffer(3)]],
                                device HCEnvLane *lanes       [[buffer(4)]],
                                constant uint2 &chunk         [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    uint laneCount = hdr[0], steps = hdr[1], mode = hdr[2];
    uint seedBase = hdr[3], episodeLen = hdr[4];
    if (gid >= laneCount) return;
    uint laneSeed = seedBase + gid;
    HCState s;
    uint elapsed, resets;
    if (chunk.x == 0u) {
        if (mode == 0u || mode == 3u) {
            hc_resetData(s);
            device const float *p = init + gid*18;
            for (int i = 0; i < 9; i++) s.qpos[i] = p[i];
            for (int i = 0; i < 9; i++) s.qvel[i] = p[9 + i];
            elapsed = 0u;
            resets = 0u;
        } else {
            elapsed = 0u;
            resets = 0u;
            hc_env_reset(s, laneSeed, elapsed, resets);
        }
    } else {
        s = lanes[gid].s;
        elapsed = lanes[gid].elapsed;
        resets = lanes[gid].resets;
    }
    if (mode == 4u) {
        device float *o = out + (ulong)gid*24;
        for (int i = 0; i < 9; i++) o[i] = s.qpos[i];
        for (int i = 0; i < 9; i++) o[9 + i] = s.qvel[i];
        float c[6];
        hc_rng_ctrl(laneSeed, 0u, c);
        for (int u = 0; u < 6; u++) o[18 + u] = c[u];
        return;
    }
    uint tEnd = min(chunk.x + chunk.y, steps);
    for (uint t = chunk.x; t < tEnd; t++) {
        if (mode == 3u) {
            float2 en = hc_energy(s);
            device float *o = out + ((ulong)gid*steps + t)*2;
            o[0] = en.x;
            o[1] = en.y;
            hc_mj_step(s);
            continue;
        }
        float ctrl[6];
        if (mode == 0u) {
            device const float *c = ctrlSeq + ((ulong)gid*steps + t)*6;
            for (int u = 0; u < 6; u++) ctrl[u] = c[u];
        } else {
            hc_rng_ctrl(laneSeed, t, ctrl);
        }
        float obs[17];
        uint done;
        float reward = hc_env_step(s, elapsed, resets, laneSeed, episodeLen, ctrl, obs, done);
        if (mode != 2u) {
            device float *o = out + ((ulong)gid*steps + t)*37;
            for (int i = 0; i < 17; i++) o[i] = obs[i];
            o[17] = reward;
            o[18] = float(done);
            for (int i = 0; i < 9; i++) o[19 + i] = s.qpos[i];
            for (int i = 0; i < 9; i++) o[28 + i] = s.qvel[i];
        }
    }
    if (mode == 2u && tEnd == steps) {
        device float *o = out + (ulong)gid*18;
        for (int i = 0; i < 9; i++) o[i] = s.qpos[i];
        for (int i = 0; i < 9; i++) o[9 + i] = s.qvel[i];
    }
    lanes[gid].s = s;
    lanes[gid].elapsed = elapsed;
    lanes[gid].resets = resets;
}

#endif
