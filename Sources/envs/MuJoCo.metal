#pragma METAL fp math_mode(safe)

constant float MJ_MINVAL = 1e-15f;
constant float MJ_MAXVAL = 1e10f;
constant float MJ_MINMU = 1e-5f;
constant float MJ_MINIMP = 0.0001f;
constant float MJ_MAXIMP = 0.9999f;
constant float MJ_SOLVER_TOL = 1e-5f;
constant float MJ_LS_TOL = 0.01f;
constant int MJ_SOLVER_ITER = 10;
constant int MJ_LS_ITER = 15;

#pragma METAL fp math_mode(fast)

inline bool mj_isBad(float x) {
    return (x != x) || (x > MJ_MAXVAL) || (x < -MJ_MAXVAL);
}

inline float mj_peerSum(device const float *store, float v) {
    return simd_sum(v);
}
inline float mj_peerSum(thread const float *store, float v) {
    return v;
}

template<typename PA, typename PB>
inline float mj_dot3(PA a, PB b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
template<typename PA, typename PB>
inline float mj_dot6(PA a, PB b) {
    float r = 0.0f;
    for (int i = 0; i < 6; i++) r += a[i]*b[i];
    return r;
}
template<uint N, typename PA, typename PB>
inline float mj_dotN(PA a, PB b) {
    float r = 0.0f;
    for (int i = 0; i < int(N); i++) r += a[i]*b[i];
    return r;
}
template<uint N, typename PA, typename PB>
inline float mj_dotNf(PA a, PB b, float init) {
    float r = init;
    for (int i = 0; i < int(N); i++) r = fma(a[i], b[i], r);
    return r;
}
template<uint N>
inline float mj_dotDevRow(device const float *a, thread const float *b, float init) {
    device const packed_float4 *a4 = (device const packed_float4 *)a;
    float4 acc = float4(0.0f);
    constexpr int N4 = int(N)/4;
    for (int k = 0; k < N4; k++) {
        float4 av = float4(a4[k]);
        float4 bv = float4(b[4*k], b[4*k + 1], b[4*k + 2], b[4*k + 3]);
        acc = fma(av, bv, acc);
    }
    float r = init + acc.x + acc.y + acc.z + acc.w;
    for (int k = 4*N4; k < int(N); k++) r = fma(a[k], b[k], r);
    return r;
}
template<typename PR, typename PA, typename PB>
inline void mj_cross(PR res, PA a, PB b) {
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}
template<typename PV3>
inline void mj_normalize3(PV3 v) {
    float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm < MJ_MINVAL) {
        v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f;
    } else {
        float inv = 1.0f / norm;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
}
inline void mj_normalize4(thread float *q) {
    float norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < MJ_MINVAL) {
        q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
    } else {
        float inv = 1.0f / norm;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}
inline void mj_mulQuat(thread float *res, thread const float *a, thread const float *b) {
    float r0 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    float r1 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    float r2 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    float r3 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    res[0] = r0; res[1] = r1; res[2] = r2; res[3] = r3;
}
inline void mj_rotVecQuat(thread float *res, thread const float *vec, thread const float *q) {
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
inline void mj_axisAngle2Quat(thread float *res, thread const float *axis, float angle) {
    if (angle == 0.0f) {
        res[0] = 1.0f; res[1] = 0.0f; res[2] = 0.0f; res[3] = 0.0f;
    } else {
        float half_ = angle * 0.5f;
        float s = sin(half_);
        res[0] = cos(half_);
        res[1] = axis[0]*s; res[2] = axis[1]*s; res[3] = axis[2]*s;
    }
}
template<typename PR>
inline void mj_quat2Mat(PR res, thread const float *q) {
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
template<typename PR, typename PI, typename PV>
inline void mj_mulInertVec(PR res, PI i, PV v) {
    res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
    res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
    res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
    res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
    res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
    res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}
template<typename PR, typename PV, typename PM>
inline void mj_crossMotion(PR res, PV v, PM m) {
    res[0] = v[1]*m[2] - v[2]*m[1];
    res[1] = v[2]*m[0] - v[0]*m[2];
    res[2] = v[0]*m[1] - v[1]*m[0];
    res[3] = v[1]*m[5] - v[2]*m[4] + v[4]*m[2] - v[5]*m[1];
    res[4] = v[2]*m[3] - v[0]*m[5] + v[5]*m[0] - v[3]*m[2];
    res[5] = v[0]*m[4] - v[1]*m[3] + v[3]*m[1] - v[4]*m[0];
}
template<typename PR, typename PV, typename PF>
inline void mj_crossForce(PR res, PV v, PF f) {
    res[0] = v[1]*f[2] - v[2]*f[1] + v[4]*f[5] - v[5]*f[4];
    res[1] = v[2]*f[0] - v[0]*f[2] + v[5]*f[3] - v[3]*f[5];
    res[2] = v[0]*f[1] - v[1]*f[0] + v[3]*f[4] - v[4]*f[3];
    res[3] = v[1]*f[5] - v[2]*f[4];
    res[4] = v[2]*f[3] - v[0]*f[5];
    res[5] = v[0]*f[4] - v[1]*f[3];
}
template<typename PR, typename PM>
inline void mj_inertCom(PR res, thread const float *inert, PM mat,
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
template<typename PV3>
inline float mj_normalize3r(PV3 v) {
    float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm < MJ_MINVAL) {
        v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f;
    } else {
        float inv = 1.0f / norm;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
    return norm;
}

inline void mj_quatIntegrate(thread float *quat, thread const float *vel, float scale) {
    float tmp[3] = { vel[0], vel[1], vel[2] };
    float angle = scale * mj_normalize3r(tmp);
    float qrot[4];
    mj_axisAngle2Quat(qrot, tmp, angle);
    mj_normalize4(quat);
    float r[4];
    mj_mulQuat(r, quat, qrot);
    quat[0] = r[0]; quat[1] = r[1]; quat[2] = r[2]; quat[3] = r[3];
}

struct MjPreCon {
    float dist;
    float pos[3];
    float normal[3];
    float tangent[3];
};

template<typename PC>
inline int mj_rawPlaneSphere(PC pc, float margin,
                             thread const float *ppos, thread const float *pnormal,
                             thread const float *spos, float r) {
    pc->normal[0] = pnormal[0]; pc->normal[1] = pnormal[1]; pc->normal[2] = pnormal[2];
    float tmp[3] = { spos[0] - ppos[0], spos[1] - ppos[1], spos[2] - ppos[2] };
    float cdist = mj_dot3(tmp, pc->normal);
    if (cdist > margin + r) return 0;
    pc->dist = cdist - r;
    float scl = -pc->dist*0.5f - r;
    pc->pos[0] = spos[0] + pc->normal[0]*scl;
    pc->pos[1] = spos[1] + pc->normal[1]*scl;
    pc->pos[2] = spos[2] + pc->normal[2]*scl;
    pc->tangent[0] = 0.0f; pc->tangent[1] = 0.0f; pc->tangent[2] = 0.0f;
    return 1;
}

template<typename PC>
inline int mj_rawSphereSphere(PC pc, float margin,
                              thread const float *pos1, float r1, thread const float *axis1,
                              thread const float *pos2, float r2, thread const float *axis2) {
    float dif[3] = { pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2] };
    float cdistSqr = mj_dot3(dif, dif);
    float minDist = margin + r1 + r2;
    if (cdistSqr > minDist*minDist) return 0;
    pc->dist = sqrt(cdistSqr) - r1 - r2;
    pc->normal[0] = pos2[0] - pos1[0];
    pc->normal[1] = pos2[1] - pos1[1];
    pc->normal[2] = pos2[2] - pos1[2];
    float len = mj_normalize3r(pc->normal);
    if (len < MJ_MINVAL) {
        mj_cross(pc->normal, axis1, axis2);
        mj_normalize3(pc->normal);
    }
    float scl = r1 + pc->dist*0.5f;
    pc->pos[0] = pos1[0] + pc->normal[0]*scl;
    pc->pos[1] = pos1[1] + pc->normal[1]*scl;
    pc->pos[2] = pos1[2] + pc->normal[2]*scl;
    pc->tangent[0] = 0.0f; pc->tangent[1] = 0.0f; pc->tangent[2] = 0.0f;
    return 1;
}

template<typename PC>
inline int mj_rawPlaneCapsule(PC pc, float margin,
                              thread const float *ppos, thread const float *pnormal,
                              thread const float *cpos, thread const float *caxis,
                              float r, float hl) {
    float seg[3] = { hl*caxis[0], hl*caxis[1], hl*caxis[2] };
    float ep[3] = { cpos[0] + seg[0], cpos[1] + seg[1], cpos[2] + seg[2] };
    int n1 = mj_rawPlaneSphere(pc, margin, ppos, pnormal, ep, r);
    ep[0] = cpos[0] - seg[0]; ep[1] = cpos[1] - seg[1]; ep[2] = cpos[2] - seg[2];
    int n2 = mj_rawPlaneSphere(pc + n1, margin, ppos, pnormal, ep, r);
    if (n1) { pc[0].tangent[0] = caxis[0]; pc[0].tangent[1] = caxis[1]; pc[0].tangent[2] = caxis[2]; }
    if (n2) { pc[n1].tangent[0] = caxis[0]; pc[n1].tangent[1] = caxis[1]; pc[n1].tangent[2] = caxis[2]; }
    return n1 + n2;
}

template<typename PC>
inline int mj_rawPlaneCylinder(PC pc, float margin,
                               thread const float *ppos, thread const float *pnormal,
                               thread const float *cpos, thread const float *cmat,
                               float r, float hl) {
    float normal[3] = { pnormal[0], pnormal[1], pnormal[2] };
    float axis[3] = { cmat[2], cmat[5], cmat[8] };
    float prjaxis = mj_dot3(normal, axis);
    if (prjaxis > 0.0f) {
        axis[0] = -axis[0]; axis[1] = -axis[1]; axis[2] = -axis[2];
        prjaxis = -prjaxis;
    }
    float vec[3] = { cpos[0] - ppos[0], cpos[1] - ppos[1], cpos[2] - ppos[2] };
    float dist0 = mj_dot3(vec, normal);
    vec[0] = axis[0]*prjaxis - normal[0];
    vec[1] = axis[1]*prjaxis - normal[1];
    vec[2] = axis[2]*prjaxis - normal[2];
    float lenSqr = mj_dot3(vec, vec);
    if (lenSqr >= MJ_MINVAL*MJ_MINVAL) {
        float scl = r / sqrt(lenSqr);
        vec[0] *= scl; vec[1] *= scl; vec[2] *= scl;
    } else {
        vec[0] = cmat[0]*r; vec[1] = cmat[3]*r; vec[2] = cmat[6]*r;
    }
    float prjvec = mj_dot3(vec, normal);
    float saxis[3] = { axis[0]*hl, axis[1]*hl, axis[2]*hl };
    float prjaxisH = prjaxis*hl;
    int cnt = 0;
    if (dist0 + prjaxisH + prjvec <= margin) {
        pc[cnt].dist = dist0 + prjaxisH + prjvec;
        for (int k = 0; k < 3; k++)
            pc[cnt].pos[k] = cpos[k] + vec[k] + saxis[k] - normal[k]*pc[cnt].dist*0.5f;
        for (int k = 0; k < 3; k++) pc[cnt].normal[k] = normal[k];
        for (int k = 0; k < 3; k++) pc[cnt].tangent[k] = 0.0f;
        cnt++;
    } else {
        return 0;
    }
    if (dist0 - prjaxisH + prjvec <= margin) {
        pc[cnt].dist = dist0 - prjaxisH + prjvec;
        for (int k = 0; k < 3; k++)
            pc[cnt].pos[k] = cpos[k] + vec[k] - saxis[k] - normal[k]*pc[cnt].dist*0.5f;
        for (int k = 0; k < 3; k++) pc[cnt].normal[k] = normal[k];
        for (int k = 0; k < 3; k++) pc[cnt].tangent[k] = 0.0f;
        cnt++;
    }
    float prjvec1 = -prjvec*0.5f;
    if (dist0 + prjaxisH + prjvec1 <= margin) {
        float vec1[3];
        mj_cross(vec1, vec, axis);
        mj_normalize3(vec1);
        float scl = r * sqrt(3.0f) * 0.5f;
        vec1[0] *= scl; vec1[1] *= scl; vec1[2] *= scl;
        for (int e = 0; e < 2; e++) {
            float sgn = (e == 0) ? 1.0f : -1.0f;
            pc[cnt].dist = dist0 + prjaxisH + prjvec1;
            for (int k = 0; k < 3; k++)
                pc[cnt].pos[k] = cpos[k] + sgn*vec1[k] + saxis[k] - vec[k]*0.5f - normal[k]*pc[cnt].dist*0.5f;
            for (int k = 0; k < 3; k++) pc[cnt].normal[k] = normal[k];
            for (int k = 0; k < 3; k++) pc[cnt].tangent[k] = 0.0f;
            cnt++;
        }
    }
    return cnt;
}

template<typename PC>
inline int mj_rawSphereCapsule(PC pc, float margin,
                               thread const float *pos1, float r1, thread const float *axis1,
                               thread const float *pos2, thread const float *caxis,
                               float r2, float hl) {
    float vec[3] = { pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2] };
    float x = clamp(mj_dot3(caxis, vec), -hl, hl);
    float np[3] = { pos2[0] + caxis[0]*x, pos2[1] + caxis[1]*x, pos2[2] + caxis[2]*x };
    return mj_rawSphereSphere(pc, margin, pos1, r1, axis1, np, r2, caxis);
}

template<typename PC>
inline int mj_rawCapsuleCapsule(PC pc, float margin,
                                thread const float *pos1, thread const float *axis1u, float r1, float hl1,
                                thread const float *pos2, thread const float *axis2u, float r2, float hl2) {
    float axis1[3] = { axis1u[0]*hl1, axis1u[1]*hl1, axis1u[2]*hl1 };
    float axis2[3] = { axis2u[0]*hl2, axis2u[1]*hl2, axis2u[2]*hl2 };
    float dif[3] = { pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2] };
    float ma = mj_dot3(axis1, axis1);
    float mb = -mj_dot3(axis1, axis2);
    float mc = mj_dot3(axis2, axis2);
    float u = -mj_dot3(axis1, dif);
    float v = mj_dot3(axis2, dif);
    float det = ma*mc - mb*mb;
    if (fabs(det) >= MJ_MINVAL) {
        float x1 = (mc*u - mb*v) / det;
        float x2 = (ma*v - mb*u) / det;
        if (x1 > 1.0f) {
            x1 = 1.0f;
            x2 = (v - mb) / mc;
        } else if (x1 < -1.0f) {
            x1 = -1.0f;
            x2 = (v + mb) / mc;
        }
        if (x2 > 1.0f) {
            x2 = 1.0f;
            x1 = clamp((u - mb) / ma, -1.0f, 1.0f);
        } else if (x2 < -1.0f) {
            x2 = -1.0f;
            x1 = clamp((u + mb) / ma, -1.0f, 1.0f);
        }
        float vec1[3] = { pos1[0] + axis1[0]*x1, pos1[1] + axis1[1]*x1, pos1[2] + axis1[2]*x1 };
        float vec2[3] = { pos2[0] + axis2[0]*x2, pos2[1] + axis2[1]*x2, pos2[2] + axis2[2]*x2 };
        return mj_rawSphereSphere(pc, margin, vec1, r1, axis1u, vec2, r2, axis2u);
    }
    float vec1[3] = { pos1[0] + axis1[0], pos1[1] + axis1[1], pos1[2] + axis1[2] };
    float x2 = clamp((v - mb) / mc, -1.0f, 1.0f);
    float vec2[3] = { pos2[0] + axis2[0]*x2, pos2[1] + axis2[1]*x2, pos2[2] + axis2[2]*x2 };
    int n1 = mj_rawSphereSphere(pc, margin, vec1, r1, axis1u, vec2, r2, axis2u);
    vec1[0] = pos1[0] - axis1[0]; vec1[1] = pos1[1] - axis1[1]; vec1[2] = pos1[2] - axis1[2];
    x2 = clamp((v + mb) / mc, -1.0f, 1.0f);
    vec2[0] = pos2[0] + axis2[0]*x2; vec2[1] = pos2[1] + axis2[1]*x2; vec2[2] = pos2[2] + axis2[2]*x2;
    int n2 = mj_rawSphereSphere(pc + n1, margin, vec1, r1, axis1u, vec2, r2, axis2u);
    if (n1 + n2 >= 2) return n1 + n2;
    vec2[0] = pos2[0] + axis2[0]; vec2[1] = pos2[1] + axis2[1]; vec2[2] = pos2[2] + axis2[2];
    float x1 = clamp((u - mb) / ma, -1.0f, 1.0f);
    vec1[0] = pos1[0] + axis1[0]*x1; vec1[1] = pos1[1] + axis1[1]*x1; vec1[2] = pos1[2] + axis1[2]*x1;
    int n3 = mj_rawSphereSphere(pc + n1 + n2, margin, vec1, r1, axis1u, vec2, r2, axis2u);
    if (n1 + n2 + n3 >= 2) return n1 + n2 + n3;
    vec2[0] = pos2[0] - axis2[0]; vec2[1] = pos2[1] - axis2[1]; vec2[2] = pos2[2] - axis2[2];
    x1 = clamp((u + mb) / ma, -1.0f, 1.0f);
    vec1[0] = pos1[0] + axis1[0]*x1; vec1[1] = pos1[1] + axis1[1]*x1; vec1[2] = pos1[2] + axis1[2]*x1;
    int n4 = mj_rawSphereSphere(pc + n1 + n2 + n3, margin, vec1, r1, axis1u, vec2, r2, axis2u);
    return n1 + n2 + n3 + n4;
}

inline void mj_makeFrame(thread float *f) {
    mj_normalize3(f);
    float ynorm = sqrt(f[3]*f[3] + f[4]*f[4] + f[5]*f[5]);
    if (ynorm < 0.5f) {
        f[3] = 0.0f; f[4] = 0.0f; f[5] = 0.0f;
        if (fabs(f[1]) < 0.9f) f[4] = 1.0f;
        else f[5] = 1.0f;
    }
    float d = mj_dot3(f, f + 3);
    float tmp[3] = { f[3] - f[0]*d, f[4] - f[1]*d, f[5] - f[2]*d };
    mj_normalize3(tmp);
    f[3] = tmp[0]; f[4] = tmp[1]; f[5] = tmp[2];
    mj_cross(f + 6, f, f + 3);
}

inline float mj_getimpedance(float pos, float d0, float d1, float width, float mid) {
    if (d0 == d1 || width <= MJ_MINVAL) return d0;
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

struct MjLS { float alpha, cost, deriv0, deriv1; };

inline MjLS mj_lsEval(float alpha, int nefc, thread const float *jarOwn, thread const float *jvOwn,
                      thread const float *quadOwn, thread const float *quadGauss,
                      thread int &lsiter, uint slane) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    int k = 0;
    for (int i = int(slane); i < nefc; i += 32, k++) {
        float x = jarOwn[k] + alpha*jvOwn[k];
        if (x < 0.0f) {
            a0 += quadOwn[3*k];
            a1 += quadOwn[3*k + 1];
            a2 += quadOwn[3*k + 2];
        }
    }
    float q0 = quadGauss[0] + simd_sum(a0);
    float q1 = quadGauss[1] + simd_sum(a1);
    float q2 = quadGauss[2] + simd_sum(a2);
    MjLS p;
    p.alpha = alpha;
    p.cost = q2*alpha*alpha + q1*alpha + q0;
    p.deriv0 = 2.0f*q2*alpha + q1;
    p.deriv1 = 2.0f*q2;
    if (p.deriv1 <= 0.0f) p.deriv1 = MJ_MINVAL;
    lsiter++;
    return p;
}

inline bool mj_updateBracket(thread MjLS &p, thread const MjLS *cand, thread MjLS &pnext,
                             int nefc, thread const float *Jaref, thread const float *Jv,
                             thread const float *quadOwn, thread const float *quadGauss,
                             thread int &lsiter, uint slane) {
    bool flag = false;
    for (int k = 0; k < 3; k++) {
        if (p.deriv0 < 0.0f && cand[k].deriv0 < 0.0f && cand[k].deriv0 > p.deriv0) {
            p = cand[k]; flag = true;
        } else if (p.deriv0 > 0.0f && cand[k].deriv0 > 0.0f && cand[k].deriv0 < p.deriv0) {
            p = cand[k]; flag = true;
        }
    }
    if (flag)
        pnext = mj_lsEval(p.alpha - p.deriv0/p.deriv1, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
    return flag;
}

inline float mj_primalSearch(int nefc, thread const float *Jaref, thread const float *Jv,
                             thread const float *quadOwn, thread const float *quadGauss,
                             float snorm, float scale, uint slane) {
    if (snorm < MJ_MINVAL) return 0.0f;
    float gtol = MJ_SOLVER_TOL * MJ_LS_TOL * snorm / scale;
    int lsiter = 0;
    MjLS p0 = mj_lsEval(0.0f, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
    MjLS p1 = mj_lsEval(p0.alpha - p0.deriv0/p0.deriv1, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
    if (fabs(p1.deriv0) < gtol) return p1.alpha;
    float dir = (p1.deriv0 < 0.0f) ? 1.0f : -1.0f;
    MjLS p2 = p0;
    while (p1.deriv0*dir <= -gtol && lsiter < MJ_LS_ITER) {
        p2 = p1;
        p1 = mj_lsEval(p1.alpha - p1.deriv0/p1.deriv1, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
        if (fabs(p1.deriv0) < gtol) return p1.alpha;
    }
    if (lsiter >= MJ_LS_ITER) return p1.alpha;
    MjLS p2next = p1;
    MjLS p1next = mj_lsEval(p1.alpha - p1.deriv0/p1.deriv1, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
    while (lsiter < MJ_LS_ITER) {
        MjLS pmid = mj_lsEval(0.5f*(p1.alpha + p2.alpha), nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
        MjLS cand[3] = { p1next, p2next, pmid };
        bool any = false;
        MjLS best = pmid;
        for (int k = 0; k < 3; k++) {
            if (fabs(cand[k].deriv0) < gtol && (!any || cand[k].cost < best.cost)) {
                best = cand[k];
                any = true;
            }
        }
        if (any) return best.alpha;
        bool b1 = mj_updateBracket(p1, cand, p1next, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
        bool b2 = mj_updateBracket(p2, cand, p2next, nefc, Jaref, Jv, quadOwn, quadGauss, lsiter, slane);
        if (!b1 && !b2) return pmid.alpha;
    }
    if (p1.cost <= p2.cost && p1.cost < p0.cost) return p1.alpha;
    if (p2.cost <= p1.cost && p2.cost < p0.cost) return p2.alpha;
    return 0.0f;
}

template<uint NV, typename PS, typename PAS, typename PM, typename PQ>
inline float mj_gaussCost(PS qfrc_smooth, PAS qacc_smooth, PM Ma, PQ qacc) {
    float g = 0.0f;
    for (int i = 0; i < int(NV); i++)
        g = fma(0.5f*(Ma[i] - qfrc_smooth[i]), qacc[i] - qacc_smooth[i], g);
    return g;
}

template<uint NQ, uint NV, uint NU, uint NBODY, uint NGEOM, uint NJNT, uint NPAIR, uint NCON, uint NEFC, uint NM>
struct MjEngine {

    struct Model {
        constant int *body_parentid;
        constant int *body_weldid;
        constant int *body_dofadr;
        constant int *body_dofnum;
        constant int *body_lastdof;
        constant float *body_pos;
        constant float *body_quat;
        constant float *body_ipos;
        constant float *body_iquat;
        constant float *body_mass;
        constant float *body_inertia;
        constant float *body_subtreemass;
        constant float *body_invweight0;
        constant int *jnt_type;
        constant int *jnt_bodyid;
        constant int *jnt_qposadr;
        constant int *jnt_dofadr;
        constant int *jnt_limited;
        constant float *jnt_pos;
        constant float *jnt_axis;
        constant float *jnt_range;
        constant float *jnt_solref;
        constant float *jnt_solimp;
        constant float *jnt_margin;
        constant float *jnt_stiffness;
        constant int *dof_bodyid;
        constant int *dof_jntid;
        constant int *dof_parentid;
        constant float *dof_armature;
        constant float *dof_damping;
        constant float *dof_invweight0;
        constant float *qpos0;
        constant float *qpos_spring;
        constant int *geom_bodyid;
        constant int *geom_type;
        constant int *geom_condim;
        constant float *geom_pos;
        constant float *geom_quat;
        constant float *geom_size;
        constant float *geom_friction;
        constant float *geom_solref;
        constant float *geom_solimp;
        constant float *geom_margin;
        constant int *actuator_trnid;
        constant float *actuator_gear;
        constant float *actuator_ctrlrange;
        constant int *pair_geom1;
        constant int *pair_geom2;
        constant int *M_rownnz;
        constant int *M_rowadr;
        constant int *M_colind;
        constant float *opt;
        constant float *spec;
        float timestep;
        float gravityZ;
        int maxDepth;
        int body_rootid[NBODY];
    };

    struct State {
        device float *qpos;
        device float *qvel;
        device float *ctrl;
        device float *qacc_warmstart;
    };

    template<typename PF>
    struct WorkT {
        PF xipos;
        PF ximat;
        PF xanchor;
        PF xaxis;
        PF geom_xpos;
        PF geom_xaxis;
        PF cinert;
        PF cdof;
        PF cvel;
        PF M;
        PF qLD;
        PF qLDiagInv;
        PF efc_J;
        PF efc_pos;
        PF efc_D;
        PF efc_aref;
        PF qfrc_bias;
        PF qfrc_smooth;
        PF qacc_smooth;
        PF qacc;
        PF qfrc_constraint;
        PF stageXpos;
        PF stageXquat;
        PF stageXmat;
        PF subCom;
        PF rneCfrc;
        PF pcbuf;
        PF conPos;
        PF conFrame;
        PF conMu;
        PF conSolref;
        PF conSolimp;
        PF conMargin;
        PF conDist;
        device int *conB1;
        device int *conB2;
        device int *conDim;
        device int *conRows;
        float scale;
        int ncon;
        int nefc;
    };
    using Work = WorkT<device float *>;



    static Model loadModel(constant uint *blob, uint entry) {
        constant uint *e = blob + 2u + 12u*entry;
        constant uint *base = blob + (e[1] >> 2u);
        Model m;
        uint o = 0u;
        m.body_parentid = (constant int *)(base + o); o += NBODY;
        m.body_weldid = (constant int *)(base + o); o += NBODY;
        m.body_dofadr = (constant int *)(base + o); o += NBODY;
        m.body_dofnum = (constant int *)(base + o); o += NBODY;
        m.body_lastdof = (constant int *)(base + o); o += NBODY;
        m.body_pos = (constant float *)(base + o); o += 3u*NBODY;
        m.body_quat = (constant float *)(base + o); o += 4u*NBODY;
        m.body_ipos = (constant float *)(base + o); o += 3u*NBODY;
        m.body_iquat = (constant float *)(base + o); o += 4u*NBODY;
        m.body_mass = (constant float *)(base + o); o += NBODY;
        m.body_inertia = (constant float *)(base + o); o += 3u*NBODY;
        m.body_subtreemass = (constant float *)(base + o); o += NBODY;
        m.body_invweight0 = (constant float *)(base + o); o += 2u*NBODY;
        m.jnt_type = (constant int *)(base + o); o += NJNT;
        m.jnt_bodyid = (constant int *)(base + o); o += NJNT;
        m.jnt_qposadr = (constant int *)(base + o); o += NJNT;
        m.jnt_dofadr = (constant int *)(base + o); o += NJNT;
        m.jnt_limited = (constant int *)(base + o); o += NJNT;
        m.jnt_pos = (constant float *)(base + o); o += 3u*NJNT;
        m.jnt_axis = (constant float *)(base + o); o += 3u*NJNT;
        m.jnt_range = (constant float *)(base + o); o += 2u*NJNT;
        m.jnt_solref = (constant float *)(base + o); o += 2u*NJNT;
        m.jnt_solimp = (constant float *)(base + o); o += 5u*NJNT;
        m.jnt_margin = (constant float *)(base + o); o += NJNT;
        m.jnt_stiffness = (constant float *)(base + o); o += NJNT;
        m.dof_bodyid = (constant int *)(base + o); o += NV;
        m.dof_jntid = (constant int *)(base + o); o += NV;
        m.dof_parentid = (constant int *)(base + o); o += NV;
        m.dof_armature = (constant float *)(base + o); o += NV;
        m.dof_damping = (constant float *)(base + o); o += NV;
        m.dof_invweight0 = (constant float *)(base + o); o += NV;
        m.qpos0 = (constant float *)(base + o); o += NQ;
        m.qpos_spring = (constant float *)(base + o); o += NQ;
        m.geom_bodyid = (constant int *)(base + o); o += NGEOM;
        m.geom_type = (constant int *)(base + o); o += NGEOM;
        m.geom_condim = (constant int *)(base + o); o += NGEOM;
        m.geom_pos = (constant float *)(base + o); o += 3u*NGEOM;
        m.geom_quat = (constant float *)(base + o); o += 4u*NGEOM;
        m.geom_size = (constant float *)(base + o); o += 3u*NGEOM;
        m.geom_friction = (constant float *)(base + o); o += 3u*NGEOM;
        m.geom_solref = (constant float *)(base + o); o += 2u*NGEOM;
        m.geom_solimp = (constant float *)(base + o); o += 5u*NGEOM;
        m.geom_margin = (constant float *)(base + o); o += NGEOM;
        m.actuator_trnid = (constant int *)(base + o); o += NU;
        m.actuator_gear = (constant float *)(base + o); o += NU;
        m.actuator_ctrlrange = (constant float *)(base + o); o += 2u*NU;
        m.pair_geom1 = (constant int *)(base + o); o += NPAIR;
        m.pair_geom2 = (constant int *)(base + o); o += NPAIR;
        m.M_rownnz = (constant int *)(base + o); o += NV;
        m.M_rowadr = (constant int *)(base + o); o += NV;
        m.M_colind = (constant int *)(base + o); o += NM;
        m.opt = (constant float *)(base + o); o += 7u;
        m.spec = (constant float *)(base + o); o += 23u;
        m.timestep = m.opt[0];
        m.gravityZ = m.opt[1];
        m.body_rootid[0] = 0;
        m.maxDepth = 0;
        for (int i = 1; i < int(NBODY); i++) {
            int r = i;
            int d = 1;
            while (m.body_parentid[r] != 0) { r = m.body_parentid[r]; d++; }
            m.body_rootid[i] = r;
            m.maxDepth = max(m.maxDepth, d);
        }
        return m;
    }

    static State bindState(device uint *envState) {
        State s;
        s.qpos = (device float *)envState;
        s.qvel = s.qpos + NQ;
        s.ctrl = s.qvel + NV;
        s.qacc_warmstart = s.ctrl + NU;
        return s;
    }

    static Work bindWork(device float *scratch) {
        Work w;
        device float *p = scratch;
        w.xipos = p; p += 3u*NBODY;
        w.ximat = p; p += 9u*NBODY;
        w.xanchor = p; p += 3u*NV;
        w.xaxis = p; p += 3u*NV;
        w.geom_xpos = p; p += 3u*NGEOM;
        w.geom_xaxis = p; p += 3u*NGEOM;
        w.cinert = p; p += 10u*NBODY;
        w.cdof = p; p += 6u*NV;
        w.cvel = p; p += 6u*NBODY;
        w.M = p; p += NM;
        w.qLD = p; p += NM;
        w.qLDiagInv = p; p += NV;
        w.efc_J = p; p += NEFC*NV;
        w.efc_pos = p; p += NEFC;
        w.efc_D = p; p += NEFC;
        w.efc_aref = p; p += NEFC;
        w.qfrc_bias = p; p += NV;
        w.qfrc_smooth = p; p += NV;
        w.qacc_smooth = p; p += NV;
        w.qacc = p; p += NV;
        w.qfrc_constraint = p; p += NV;
        w.stageXpos = p; p += 3u*NBODY;
        w.stageXquat = p; p += 4u*NBODY;
        w.stageXmat = p; p += 9u*NBODY;
        w.subCom = p; p += 3u*NBODY;
        w.rneCfrc = p; p += 6u*NBODY;
        w.pcbuf = p; p += 40u;
        w.conPos = p; p += 3u*(NCON == 0u ? 1u : NCON);
        w.conFrame = p; p += 9u*(NCON == 0u ? 1u : NCON);
        w.conMu = p; p += (NCON == 0u ? 1u : NCON);
        w.conSolref = p; p += 2u*(NCON == 0u ? 1u : NCON);
        w.conSolimp = p; p += 5u*(NCON == 0u ? 1u : NCON);
        w.conMargin = p; p += (NCON == 0u ? 1u : NCON);
        w.conDist = p; p += (NCON == 0u ? 1u : NCON);
        w.conB1 = (device int *)p; p += (NCON == 0u ? 1u : NCON);
        w.conB2 = (device int *)p; p += (NCON == 0u ? 1u : NCON);
        w.conDim = (device int *)p; p += (NCON == 0u ? 1u : NCON);
        w.conRows = (device int *)p; p += (NCON == 0u ? 1u : NCON);
        w.scale = 0.0f;
        w.ncon = 0;
        w.nefc = 0;
        return w;
    }

    template<typename PL, typename PD>
    static void factorI(thread const Model &m, PL LD, PD diagInv) {
        for (int j = int(NV) - 1; j >= 0; j--) {
            int start = m.M_rowadr[j];
            int diagidx = m.M_rownnz[j] - 1;
            int end = start + diagidx;
            float invD = 1.0f / LD[end];
            diagInv[j] = invD;
            for (int adr = end - 1; adr >= start; adr--) {
                int i = m.M_colind[adr];
                float a = -LD[adr] * invD;
                int ri = m.M_rowadr[i];
                int ni = m.M_rownnz[i];
                for (int k = 0; k < ni; k++) LD[ri + k] += LD[start + k] * a;
            }
            for (int adr = start; adr < end; adr++) LD[adr] *= invD;
        }
    }

    static int bodyDepth(thread const Model &m, int i) {
        int d = 0;
        while (i != 0) { i = m.body_parentid[i]; d++; }
        return d;
    }

    static int bodyAncestor(thread const Model &m, int i, int depth, int level) {
        for (int u = depth; u > level; u--) i = m.body_parentid[i];
        return i;
    }

    static bool bodyUnder(thread const Model &m, int root, int i) {
        for (int a = m.body_parentid[i]; a != 0; a = m.body_parentid[a])
            if (a == root) return true;
        return false;
    }

    template<typename W>
    static void kinematics(thread const Model &m, State s, thread W &w,
                           uint slane, uint snum) {
        auto xpos = w.stageXpos;
        auto xquat = w.stageXquat;
        auto xmat = w.stageXmat;
        if (slane == 0u) {
            for (int k = 0; k < 3; k++) { xpos[k] = 0.0f; w.xipos[k] = 0.0f; }
            for (int k = 0; k < 4; k++) xquat[k] = (k == 0) ? 1.0f : 0.0f;
            for (int k = 0; k < 9; k++) {
                xmat[k] = (k % 4 == 0) ? 1.0f : 0.0f;
                w.ximat[k] = (k % 4 == 0) ? 1.0f : 0.0f;
            }
        }
        for (int b0 = 0; b0 < int(NBODY); b0 += int(snum)) {
            int i = b0 + int(slane);
            if (i < 1 || i >= int(NBODY)) continue;
            int depth = bodyDepth(m, i);
            float pos[3] = { 0.0f, 0.0f, 0.0f };
            float quat[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
            float mat[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
            for (int level = 1; level <= depth; level++) {
                int b = bodyAncestor(m, i, depth, level);
                bool owner = (b == i);
                int firstJ = (m.body_dofnum[b] > 0) ? m.dof_jntid[m.body_dofadr[b]] : -1;
                if (firstJ >= 0 && m.jnt_type[firstJ] == 0) {
                    int qadr = m.jnt_qposadr[firstJ];
                    pos[0] = s.qpos[qadr]; pos[1] = s.qpos[qadr + 1]; pos[2] = s.qpos[qadr + 2];
                    quat[0] = s.qpos[qadr + 3]; quat[1] = s.qpos[qadr + 4];
                    quat[2] = s.qpos[qadr + 5]; quat[3] = s.qpos[qadr + 6];
                    mj_normalize4(quat);
                    if (owner) {
                        w.xanchor[3*firstJ] = pos[0]; w.xanchor[3*firstJ + 1] = pos[1]; w.xanchor[3*firstJ + 2] = pos[2];
                        w.xaxis[3*firstJ] = m.jnt_axis[3*firstJ];
                        w.xaxis[3*firstJ + 1] = m.jnt_axis[3*firstJ + 1];
                        w.xaxis[3*firstJ + 2] = m.jnt_axis[3*firstJ + 2];
                    }
                } else {
                    float bpos[3] = { m.body_pos[3*b], m.body_pos[3*b + 1], m.body_pos[3*b + 2] };
                    float bquat[4] = { m.body_quat[4*b], m.body_quat[4*b + 1], m.body_quat[4*b + 2], m.body_quat[4*b + 3] };
                    if (m.body_parentid[b] == 0) {
                        pos[0] = bpos[0]; pos[1] = bpos[1]; pos[2] = bpos[2];
                        quat[0] = bquat[0]; quat[1] = bquat[1]; quat[2] = bquat[2]; quat[3] = bquat[3];
                    } else {
                        float px = mat[0]*bpos[0] + mat[1]*bpos[1] + mat[2]*bpos[2] + pos[0];
                        float py = mat[3]*bpos[0] + mat[4]*bpos[1] + mat[5]*bpos[2] + pos[1];
                        float pz = mat[6]*bpos[0] + mat[7]*bpos[1] + mat[8]*bpos[2] + pos[2];
                        float pq[4] = { quat[0], quat[1], quat[2], quat[3] };
                        mj_mulQuat(quat, pq, bquat);
                        pos[0] = px; pos[1] = py; pos[2] = pz;
                    }
                    for (int j = firstJ; j >= 0 && j < int(NJNT) && m.jnt_bodyid[j] == b; j++) {
                        float axisLocal[3] = { m.jnt_axis[3*j], m.jnt_axis[3*j + 1], m.jnt_axis[3*j + 2] };
                        float jp[3] = { m.jnt_pos[3*j], m.jnt_pos[3*j + 1], m.jnt_pos[3*j + 2] };
                        float xax[3], anc[3];
                        mj_rotVecQuat(xax, axisLocal, quat);
                        mj_rotVecQuat(anc, jp, quat);
                        anc[0] += pos[0]; anc[1] += pos[1]; anc[2] += pos[2];
                        int qadr = m.jnt_qposadr[j];
                        if (m.jnt_type[j] == 2) {
                            float dq = s.qpos[qadr] - m.qpos0[qadr];
                            pos[0] += xax[0]*dq; pos[1] += xax[1]*dq; pos[2] += xax[2]*dq;
                        } else {
                            float qloc[4], qn[4];
                            mj_axisAngle2Quat(qloc, axisLocal, s.qpos[qadr] - m.qpos0[qadr]);
                            mj_mulQuat(qn, quat, qloc);
                            quat[0] = qn[0]; quat[1] = qn[1]; quat[2] = qn[2]; quat[3] = qn[3];
                            float off[3];
                            mj_rotVecQuat(off, jp, quat);
                            pos[0] = anc[0] - off[0]; pos[1] = anc[1] - off[1]; pos[2] = anc[2] - off[2];
                        }
                        if (owner) {
                            w.xanchor[3*j] = anc[0]; w.xanchor[3*j + 1] = anc[1]; w.xanchor[3*j + 2] = anc[2];
                            w.xaxis[3*j] = xax[0]; w.xaxis[3*j + 1] = xax[1]; w.xaxis[3*j + 2] = xax[2];
                        }
                    }
                    mj_normalize4(quat);
                }
                mj_quat2Mat(mat, quat);
            }
            for (int k = 0; k < 3; k++) xpos[3*i + k] = pos[k];
            for (int k = 0; k < 4; k++) xquat[4*i + k] = quat[k];
            for (int k = 0; k < 9; k++) xmat[9*i + k] = mat[k];
            float ip[3] = { m.body_ipos[3*i], m.body_ipos[3*i + 1], m.body_ipos[3*i + 2] };
            w.xipos[3*i + 0] = mat[0]*ip[0] + mat[1]*ip[1] + mat[2]*ip[2] + pos[0];
            w.xipos[3*i + 1] = mat[3]*ip[0] + mat[4]*ip[1] + mat[5]*ip[2] + pos[1];
            w.xipos[3*i + 2] = mat[6]*ip[0] + mat[7]*ip[1] + mat[8]*ip[2] + pos[2];
            float iq[4] = { m.body_iquat[4*i], m.body_iquat[4*i + 1], m.body_iquat[4*i + 2], m.body_iquat[4*i + 3] };
            float q[4];
            mj_mulQuat(q, quat, iq);
            mj_quat2Mat(w.ximat + 9*i, q);
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
        for (int g0 = 0; g0 < int(NGEOM); g0 += int(snum)) {
            int g = g0 + int(slane);
            if (g >= int(NGEOM)) continue;
            int b = m.geom_bodyid[g];
            auto bm = xmat + 9*b;
            float gp[3] = { m.geom_pos[3*g], m.geom_pos[3*g + 1], m.geom_pos[3*g + 2] };
            w.geom_xpos[3*g + 0] = bm[0]*gp[0] + bm[1]*gp[1] + bm[2]*gp[2] + xpos[3*b + 0];
            w.geom_xpos[3*g + 1] = bm[3]*gp[0] + bm[4]*gp[1] + bm[5]*gp[2] + xpos[3*b + 1];
            w.geom_xpos[3*g + 2] = bm[6]*gp[0] + bm[7]*gp[1] + bm[8]*gp[2] + xpos[3*b + 2];
            float gq[4] = { m.geom_quat[4*g], m.geom_quat[4*g + 1], m.geom_quat[4*g + 2], m.geom_quat[4*g + 3] };
            float bq[4] = { xquat[4*b], xquat[4*b + 1], xquat[4*b + 2], xquat[4*b + 3] };
            float q[4], gm[9];
            mj_mulQuat(q, bq, gq);
            mj_quat2Mat(gm, q);
            w.geom_xaxis[3*g + 0] = gm[2];
            w.geom_xaxis[3*g + 1] = gm[5];
            w.geom_xaxis[3*g + 2] = gm[8];
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
    }

    template<typename W>
    static void comPos(thread const Model &m, State s, thread W &w,
                       uint slane, uint snum) {
        auto subCom = w.subCom;
        for (int r = 1; r < int(NBODY); r++) {
            if (m.body_parentid[r] != 0) continue;
            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            for (int b0 = 0; b0 < int(NBODY); b0 += int(snum)) {
                int j = b0 + int(slane);
                bool inSub = (j > 0 && j < int(NBODY)) && (m.body_rootid[j] == r);
                int src = inSub ? j : 0;
                float mass = inSub ? m.body_mass[src] : 0.0f;
                cx += w.xipos[3*src + 0]*mass;
                cy += w.xipos[3*src + 1]*mass;
                cz += w.xipos[3*src + 2]*mass;
            }
            cx = mj_peerSum(subCom, cx);
            cy = mj_peerSum(subCom, cy);
            cz = mj_peerSum(subCom, cz);
            if (m.body_subtreemass[r] < MJ_MINVAL) {
                cx = w.xipos[3*r + 0]; cy = w.xipos[3*r + 1]; cz = w.xipos[3*r + 2];
            } else {
                float inv = 1.0f / m.body_subtreemass[r];
                cx *= inv; cy *= inv; cz *= inv;
            }
            if (slane == 0u) { subCom[3*r + 0] = cx; subCom[3*r + 1] = cy; subCom[3*r + 2] = cz; }
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
        for (int k = 0; k < 10; k++) w.cinert[k] = 0.0f;
        for (int i = 1 + int(slane); i < int(NBODY); i += int(snum)) {
            int r = m.body_rootid[i];
            float inert[3] = { m.body_inertia[3*i], m.body_inertia[3*i + 1], m.body_inertia[3*i + 2] };
            float off[3] = { w.xipos[3*i] - subCom[3*r], w.xipos[3*i + 1] - subCom[3*r + 1], w.xipos[3*i + 2] - subCom[3*r + 2] };
            mj_inertCom(w.cinert + 10*i, inert, w.ximat + 9*i, off, m.body_mass[i]);
        }
        for (int j = int(slane); j < int(NJNT); j += int(snum)) {
            int bi = m.jnt_bodyid[j];
            int r = m.body_rootid[bi];
            int d = m.jnt_dofadr[j];
            float off[3] = { subCom[3*r] - w.xanchor[3*j], subCom[3*r + 1] - w.xanchor[3*j + 1], subCom[3*r + 2] - w.xanchor[3*j + 2] };
            int jt = m.jnt_type[j];
            if (jt == 0) {
                for (int k = 0; k < 18; k++) w.cdof[6*d + k] = 0.0f;
                w.cdof[6*d + 3] = 1.0f;
                w.cdof[6*d + 10] = 1.0f;
                w.cdof[6*d + 17] = 1.0f;
                int qadr = m.jnt_qposadr[j];
                float bq[4] = { s.qpos[qadr + 3], s.qpos[qadr + 4], s.qpos[qadr + 5], s.qpos[qadr + 6] };
                mj_normalize4(bq);
                float bm[9];
                mj_quat2Mat(bm, bq);
                for (int k = 0; k < 3; k++) {
                    float axis[3] = { bm[k], bm[k + 3], bm[k + 6] };
                    int dd = d + 3 + k;
                    w.cdof[6*dd + 0] = axis[0]; w.cdof[6*dd + 1] = axis[1]; w.cdof[6*dd + 2] = axis[2];
                    mj_cross(w.cdof + 6*dd + 3, axis, off);
                }
            } else if (jt == 2) {
                auto ax = w.xaxis + 3*j;
                w.cdof[6*d + 0] = 0.0f; w.cdof[6*d + 1] = 0.0f; w.cdof[6*d + 2] = 0.0f;
                w.cdof[6*d + 3] = ax[0]; w.cdof[6*d + 4] = ax[1]; w.cdof[6*d + 5] = ax[2];
            } else {
                auto ax = w.xaxis + 3*j;
                w.cdof[6*d + 0] = ax[0]; w.cdof[6*d + 1] = ax[1]; w.cdof[6*d + 2] = ax[2];
                mj_cross(w.cdof + 6*d + 3, w.xaxis + 3*j, off);
            }
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
    }

    static void makeM(thread const Model &m, thread Work &w, thread float *mrow, uint slane) {
        int i = int(slane);
        for (int c = 0; c < int(NV); c++) mrow[c] = 0.0f;
        if (i < int(NV)) {
            int b = m.dof_bodyid[i];
            float crbBody[10];
            for (int k = 0; k < 10; k++) crbBody[k] = w.cinert[10*b + k];
            for (int j = b + 1; j < int(NBODY); j++) {
                if (!bodyUnder(m, b, j)) continue;
                for (int k = 0; k < 10; k++) crbBody[k] += w.cinert[10*j + k];
            }
            float buf[6];
            mj_mulInertVec(buf, crbBody, w.cdof + 6*i);
            mrow[i] = m.dof_armature[i] + mj_dot6(w.cdof + 6*i, buf);
            int j = m.dof_parentid[i];
            while (j >= 0) {
                mrow[j] = mj_dot6(w.cdof + 6*j, buf);
                j = m.dof_parentid[j];
            }
            int adr = m.M_rowadr[i];
            int nnz = m.M_rownnz[i];
            for (int k = 0; k < nnz; k++) w.M[adr + k] = mrow[m.M_colind[adr + k]];
        }
        simdgroup_barrier(mem_flags::mem_device);
    }

    static void mulMVecDist(thread const float *mrow, thread const float *x, thread float *y, uint slane) {
        float contrib[NV];
        for (int i = 0; i < int(NV); i++) contrib[i] = 0.0f;
        int r = int(slane);
        if (r < int(NV)) {
            float xr = x[r];
            float sacc = 0.0f;
            for (int j = 0; j < r; j++) {
                contrib[j] = mrow[j]*xr;
                sacc = fma(mrow[j], x[j], sacc);
            }
            sacc = fma(mrow[r], xr, sacc);
            contrib[r] = sacc;
        }
        for (int i = 0; i < int(NV); i++) y[i] = simd_sum(contrib[i]);
    }

    static void cholFactorRowsDist(thread float *row, uint slane) {
        int i = int(slane);
        for (int j = 0; j < int(NV); j++) {
            float diagRaw = simd_shuffle(row[j], ushort(j));
            float d = sqrt(max(diagRaw, MJ_MINVAL));
            if (i == j) row[j] = d;
            else if (i > j && i < int(NV)) row[j] *= 1.0f/d;
            for (int c = j + 1; c < int(NV); c++) {
                float Lcj = simd_shuffle(row[j], ushort(c));
                if (i >= c && i < int(NV)) row[c] = fma(-row[j], Lcj, row[c]);
            }
        }
    }

    static void cholSolveRowsDist(thread const float *row, thread const float *rhs, thread float *res,
                                  uint slane) {
        int i = int(slane);
        float yv[NV];
        float yOwn = (i < int(NV)) ? rhs[i] : 0.0f;
        for (int j = 0; j < int(NV); j++) {
            float djj = simd_shuffle(row[j], ushort(j));
            float yj = simd_shuffle(yOwn, ushort(j))/djj;
            yv[j] = yj;
            if (i > j && i < int(NV)) yOwn = fma(-row[j], yj, yOwn);
        }
        float xOwn = 0.0f;
        for (int k = int(NV) - 1; k >= 0; k--) {
            float part = (i > k && i < int(NV)) ? row[k]*xOwn : 0.0f;
            float ssum = simd_sum(part);
            float diag = simd_shuffle(row[k], ushort(k));
            float xk = (yv[k] - ssum)/diag;
            if (i == k) xOwn = xk;
            res[k] = xk;
        }
    }

    template<typename W>
    static void collision(thread const Model &m, State s, thread W &w, uint slane) {
        auto xquat = w.stageXquat;
        w.ncon = 0;
        uint cand[4] = { 0u, 0u, 0u, 0u };
        for (int p = int(slane); p < int(NPAIR); p += 32) {
            int g1 = m.pair_geom1[p];
            int g2 = m.pair_geom2[p];
            if (m.geom_type[g1] > m.geom_type[g2]) { int t = g1; g1 = g2; g2 = t; }
            int t1 = m.geom_type[g1];
            int t2 = m.geom_type[g2];
            float margin = m.geom_margin[g1] + m.geom_margin[g2];
            float reach2 = m.geom_size[3*g2] + ((t2 == 3 || t2 == 5) ? m.geom_size[3*g2 + 1] : 0.0f);
            float d1[3] = { w.geom_xpos[3*g2] - w.geom_xpos[3*g1],
                           w.geom_xpos[3*g2 + 1] - w.geom_xpos[3*g1 + 1],
                           w.geom_xpos[3*g2 + 2] - w.geom_xpos[3*g1 + 2] };
            bool hit;
            if (t1 == 0) {
                float n[3] = { w.geom_xaxis[3*g1], w.geom_xaxis[3*g1 + 1], w.geom_xaxis[3*g1 + 2] };
                hit = mj_dot3(n, d1) <= margin + reach2;
            } else {
                float reach1 = m.geom_size[3*g1] + ((t1 == 3 || t1 == 5) ? m.geom_size[3*g1 + 1] : 0.0f);
                float b = margin + reach1 + reach2;
                hit = mj_dot3(d1, d1) <= b*b;
            }
            if (hit) cand[p >> 5] |= (1u << uint(p & 31));
        }
        for (int k = 0; k < 4; k++) cand[k] = simd_or(cand[k]);
        for (int p = 0; p < int(NPAIR); p++) {
            if ((cand[p >> 5] & (1u << uint(p & 31))) == 0u) continue;
            int g1 = m.pair_geom1[p];
            int g2 = m.pair_geom2[p];
            if (m.geom_type[g1] > m.geom_type[g2]) { int t = g1; g1 = g2; g2 = t; }
            int t1 = m.geom_type[g1];
            int t2 = m.geom_type[g2];
            float margin = m.geom_margin[g1] + m.geom_margin[g2];
            float pos1[3] = { w.geom_xpos[3*g1], w.geom_xpos[3*g1 + 1], w.geom_xpos[3*g1 + 2] };
            float pos2[3] = { w.geom_xpos[3*g2], w.geom_xpos[3*g2 + 1], w.geom_xpos[3*g2 + 2] };
            float ax1[3] = { w.geom_xaxis[3*g1], w.geom_xaxis[3*g1 + 1], w.geom_xaxis[3*g1 + 2] };
            float ax2[3] = { w.geom_xaxis[3*g2], w.geom_xaxis[3*g2 + 1], w.geom_xaxis[3*g2 + 2] };
            float r1 = m.geom_size[3*g1];
            float hl1 = m.geom_size[3*g1 + 1];
            float r2 = m.geom_size[3*g2];
            float hl2 = m.geom_size[3*g2 + 1];
            device MjPreCon *pc = (device MjPreCon *)w.pcbuf;
            int n = 0;
            if (t1 == 0 && t2 == 2) {
                n = mj_rawPlaneSphere(pc, margin, pos1, ax1, pos2, r2);
            } else if (t1 == 0 && t2 == 3) {
                n = mj_rawPlaneCapsule(pc, margin, pos1, ax1, pos2, ax2, r2, hl2);
            } else if (t1 == 0 && t2 == 5) {
                int b2 = m.geom_bodyid[g2];
                float gq[4] = { m.geom_quat[4*g2], m.geom_quat[4*g2 + 1], m.geom_quat[4*g2 + 2], m.geom_quat[4*g2 + 3] };
                float bq[4] = { xquat[4*b2], xquat[4*b2 + 1], xquat[4*b2 + 2], xquat[4*b2 + 3] };
                float q[4], gm[9];
                mj_mulQuat(q, bq, gq);
                mj_quat2Mat(gm, q);
                n = mj_rawPlaneCylinder(pc, margin, pos1, ax1, pos2, gm, r2, hl2);
            } else if (t1 == 2 && t2 == 2) {
                n = mj_rawSphereSphere(pc, margin, pos1, r1, ax1, pos2, r2, ax2);
            } else if (t1 == 2 && t2 == 3) {
                n = mj_rawSphereCapsule(pc, margin, pos1, r1, ax1, pos2, ax2, r2, hl2);
            } else if (t1 == 3 && t2 == 3) {
                n = mj_rawCapsuleCapsule(pc, margin, pos1, ax1, r1, hl1, pos2, ax2, r2, hl2);
            } else if (t1 == 3 && t2 == 5) {
                n = mj_rawCapsuleCapsule(pc, margin, pos1, ax1, r1, hl1, pos2, ax2, r2, hl2);
            }
            if (n == 0) continue;
            int condim = max(m.geom_condim[g1], m.geom_condim[g2]);
            float solref[2], solimp[5], fri[3];
            if (m.geom_solref[2*g1] > 0.0f && m.geom_solref[2*g2] > 0.0f) {
                for (int k = 0; k < 2; k++) solref[k] = 0.5f*m.geom_solref[2*g1 + k] + 0.5f*m.geom_solref[2*g2 + k];
            } else {
                for (int k = 0; k < 2; k++) solref[k] = min(m.geom_solref[2*g1 + k], m.geom_solref[2*g2 + k]);
            }
            for (int k = 0; k < 5; k++) solimp[k] = 0.5f*m.geom_solimp[5*g1 + k] + 0.5f*m.geom_solimp[5*g2 + k];
            for (int k = 0; k < 3; k++) fri[k] = max(m.geom_friction[3*g1 + k], m.geom_friction[3*g2 + k]);
            float mu = max(MJ_MINMU, fri[0]);
            for (int k = 0; k < n; k++) {
                int c = w.ncon;
                if (c >= int(NCON)) break;
                bool excl = (pc[k].dist >= margin);
                w.conRows[c] = excl ? 0 : ((condim == 1) ? 1 : 4);
                if (slane == 0u) {
                    w.conDist[c] = pc[k].dist;
                    w.conPos[3*c + 0] = pc[k].pos[0];
                    w.conPos[3*c + 1] = pc[k].pos[1];
                    w.conPos[3*c + 2] = pc[k].pos[2];
                    float fr[9] = { pc[k].normal[0], pc[k].normal[1], pc[k].normal[2],
                                    pc[k].tangent[0], pc[k].tangent[1], pc[k].tangent[2],
                                    0.0f, 0.0f, 0.0f };
                    mj_makeFrame(fr);
                    for (int t = 0; t < 9; t++) w.conFrame[9*c + t] = fr[t];
                    w.conB1[c] = m.geom_bodyid[g1];
                    w.conB2[c] = m.geom_bodyid[g2];
                    w.conDim[c] = condim;
                    w.conMu[c] = mu;
                    w.conSolref[2*c + 0] = solref[0];
                    w.conSolref[2*c + 1] = solref[1];
                    for (int t = 0; t < 5; t++) w.conSolimp[5*c + t] = solimp[t];
                    w.conMargin[c] = margin;
                }
                w.ncon = c + 1;
            }
        }
    }

    static void makeConstraint(thread const Model &m, State s, thread Work &w, uint slane) {
        float qv[NV];
        for (int i = 0; i < int(NV); i++) qv[i] = s.qvel[i];
        int nefc = 0;
        for (int j = 0; j < int(NJNT); j++) {
            if (!m.jnt_limited[j]) continue;
            int qadr = m.jnt_qposadr[j];
            int dadr = m.jnt_dofadr[j];
            float v = s.qpos[qadr];
            for (int side = -1; side <= 1; side += 2) {
                float dist = (side == -1) ? (v - m.jnt_range[2*j + 0])
                                          : (m.jnt_range[2*j + 1] - v);
                if (dist < m.jnt_margin[j]) {
                    if (nefc >= int(NEFC)) break;
                    for (int k = 0; k < int(NV); k++) w.efc_J[int(NV)*nefc + k] = 0.0f;
                    w.efc_J[int(NV)*nefc + dadr] = (float)(-side);
                    float posm = dist - m.jnt_margin[j];
                    w.efc_pos[nefc] = posm;
                    float dA = m.dof_invweight0[dadr];
                    float solref0 = m.jnt_solref[2*j + 0];
                    float solref1 = m.jnt_solref[2*j + 1];
                    if (solref0 > 0.0f) solref0 = max(solref0, 2.0f*m.timestep);
                    float d0 = clamp(m.jnt_solimp[5*j + 0], MJ_MINIMP, MJ_MAXIMP);
                    float d1 = clamp(m.jnt_solimp[5*j + 1], MJ_MINIMP, MJ_MAXIMP);
                    float width = m.jnt_solimp[5*j + 2];
                    float mid = clamp(m.jnt_solimp[5*j + 3], MJ_MINIMP, MJ_MAXIMP);
                    float imp = mj_getimpedance(posm, d0, d1, width, mid);
                    float R = max(MJ_MINVAL, (1.0f - imp)*dA/imp);
                    float K;
                    if (solref0 > 0.0f) {
                        K = 1.0f / max(MJ_MINVAL, d1*d1*solref0*solref0*solref1*solref1);
                    } else {
                        K = -solref0 / max(MJ_MINVAL, d1*d1);
                    }
                    float B;
                    if (solref1 > 0.0f) {
                        B = 2.0f / max(MJ_MINVAL, d1*solref0);
                    } else {
                        B = -solref1 / max(MJ_MINVAL, d1);
                    }
                    w.efc_D[nefc] = 1.0f / R;
                    float vel = (float)(-side)*qv[dadr];
                    w.efc_aref[nefc] = -B*vel - K*imp*posm;
                    nefc++;
                }
            }
        }
        int nlimit = nefc;
        for (int c = 0; c < w.ncon; c++) {
            int rows = w.conRows[c];
            if (rows == 0) continue;
            if (nefc + rows > int(NEFC)) { if (slane == 0u) w.conRows[c] = 0; rows = 0; continue; }
            nefc += rows;
        }
        w.nefc = nefc;
        simdgroup_barrier(mem_flags::mem_device);
        {
            int adr = nlimit;
            for (int c = 0; c < w.ncon; c++) {
                int rows = w.conRows[c];
                if (rows == 0) continue;
                int myadr = adr;
                adr += rows;
                if (uint(c & 31) != slane) continue;
                int cadr = myadr;
                int dim = w.conDim[c];
                int b1 = w.conB1[c];
                int b2 = w.conB2[c];
                float fr[9];
                for (int t = 0; t < 9; t++) fr[t] = w.conFrame[9*c + t];
                float cpos[3] = { w.conPos[3*c], w.conPos[3*c + 1], w.conPos[3*c + 2] };
                float mu = w.conMu[c];
                int nrows = (dim == 1) ? 1 : 4;
                for (int rr = 0; rr < nrows; rr++)
                    for (int col = 0; col < int(NV); col++)
                        w.efc_J[int(NV)*(cadr + rr) + col] = 0.0f;
                float vel[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                for (int side = 0; side < 2; side++) {
                    int b = (side == 0) ? b1 : b2;
                    float sg = (side == 0) ? -1.0f : 1.0f;
                    if (b == 0) continue;
                    int r = m.body_rootid[b];
                    float off[3] = { cpos[0] - w.subCom[3*r],
                                     cpos[1] - w.subCom[3*r + 1],
                                     cpos[2] - w.subCom[3*r + 2] };
                    int i = m.body_lastdof[m.body_weldid[b]];
                    while (i >= 0) {
                        device const float *cd = w.cdof + 6*i;
                        float jp0 = sg*(cd[3] + (cd[1]*off[2] - cd[2]*off[1]));
                        float jp1 = sg*(cd[4] + (cd[2]*off[0] - cd[0]*off[2]));
                        float jp2 = sg*(cd[5] + (cd[0]*off[1] - cd[1]*off[0]));
                        float jc0 = fr[0]*jp0 + fr[1]*jp1 + fr[2]*jp2;
                        float qvi = qv[i];
                        if (dim == 1) {
                            float nw = w.efc_J[int(NV)*cadr + i] + jc0;
                            w.efc_J[int(NV)*cadr + i] = nw;
                            vel[0] += jc0*qvi;
                        } else {
                            float jc1 = fr[3]*jp0 + fr[4]*jp1 + fr[5]*jp2;
                            float jc2 = fr[6]*jp0 + fr[7]*jp1 + fr[8]*jp2;
                            float a0 = jc0 + mu*jc1;
                            float a1 = jc0 - mu*jc1;
                            float a2 = jc0 + mu*jc2;
                            float a3 = jc0 - mu*jc2;
                            w.efc_J[int(NV)*(cadr + 0) + i] += a0;
                            w.efc_J[int(NV)*(cadr + 1) + i] += a1;
                            w.efc_J[int(NV)*(cadr + 2) + i] += a2;
                            w.efc_J[int(NV)*(cadr + 3) + i] += a3;
                            vel[0] += a0*qvi;
                            vel[1] += a1*qvi;
                            vel[2] += a2*qvi;
                            vel[3] += a3*qvi;
                        }
                        i = m.dof_parentid[i];
                    }
                }
                float solref0 = w.conSolref[2*c + 0];
                float solref1 = w.conSolref[2*c + 1];
                if (solref0 > 0.0f) solref0 = max(solref0, 2.0f*m.timestep);
                float d0 = clamp(w.conSolimp[5*c + 0], MJ_MINIMP, MJ_MAXIMP);
                float d1 = clamp(w.conSolimp[5*c + 1], MJ_MINIMP, MJ_MAXIMP);
                float width = w.conSolimp[5*c + 2];
                float mid = clamp(w.conSolimp[5*c + 3], MJ_MINIMP, MJ_MAXIMP);
                float posm = w.conDist[c] - w.conMargin[c];
                float imp = mj_getimpedance(posm, d0, d1, width, mid);
                float K;
                if (solref0 > 0.0f) {
                    K = 1.0f / max(MJ_MINVAL, d1*d1*solref0*solref0*solref1*solref1);
                } else {
                    K = -solref0 / max(MJ_MINVAL, d1*d1);
                }
                float B;
                if (solref1 > 0.0f) {
                    B = 2.0f / max(MJ_MINVAL, d1*solref0);
                } else {
                    B = -solref1 / max(MJ_MINVAL, d1);
                }
                float tran = m.body_invweight0[2*b1 + 0] + m.body_invweight0[2*b2 + 0];
                if (dim == 1) {
                    w.efc_pos[cadr] = posm;
                    float R0 = max(MJ_MINVAL, (1.0f - imp)*tran/imp);
                    w.efc_D[cadr] = 1.0f / R0;
                    w.efc_aref[cadr] = -B*vel[0] - K*imp*posm;
                } else {
                    for (int rr = 0; rr < 4; rr++) {
                        w.efc_pos[cadr + rr] = posm;
                        w.efc_aref[cadr + rr] = -B*vel[rr] - K*imp*posm;
                    }
                    float dA0 = tran + mu*mu*tran;
                    float R0 = max(MJ_MINVAL, (1.0f - imp)*dA0/imp);
                    float Rpy = 2.0f*mu*mu*R0;
                    for (int r2 = cadr; r2 < cadr + 4; r2++) w.efc_D[r2] = 1.0f / Rpy;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_device);
    }

    template<typename W>
    static void comVelRne(thread const Model &m, State s, thread W &w, thread float *qfrcBiasT,
                          uint slane, uint snum) {
        auto cfrc = w.rneCfrc;
        auto cvel = w.cvel;
        if (slane == 0u) {
            for (int k = 0; k < 6; k++) { cvel[k] = 0.0f; cfrc[k] = 0.0f; }
        }
        for (int b0 = 0; b0 < int(NBODY); b0 += int(snum)) {
            int i = b0 + int(slane);
            if (i < 1 || i >= int(NBODY)) continue;
            int depth = bodyDepth(m, i);
            float v6[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            float acc[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -m.gravityZ };
            for (int level = 1; level <= depth; level++) {
                int b = bodyAncestor(m, i, depth, level);
                int bda = m.body_dofadr[b];
                int nn = m.body_dofnum[b];
                int k = 0;
                while (k < nn) {
                    int j = bda + k;
                    if (m.jnt_type[m.dof_jntid[j]] == 0) {
                        for (int t = 0; t < 3; t++)
                            for (int c = 0; c < 6; c++) v6[c] += w.cdof[6*(j + t) + c] * s.qvel[j + t];
                        for (int t = 3; t < 6; t++) {
                            float dd[6];
                            mj_crossMotion(dd, v6, w.cdof + 6*(j + t));
                            for (int c = 0; c < 6; c++) acc[c] += dd[c] * s.qvel[j + t];
                        }
                        float tmp[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        for (int t = 3; t < 6; t++)
                            for (int c = 0; c < 6; c++) tmp[c] += w.cdof[6*(j + t) + c] * s.qvel[j + t];
                        for (int c = 0; c < 6; c++) v6[c] += tmp[c];
                        k += 6;
                    } else {
                        float dd[6];
                        mj_crossMotion(dd, v6, w.cdof + 6*j);
                        for (int c = 0; c < 6; c++) acc[c] += dd[c] * s.qvel[j];
                        for (int t = 0; t < 6; t++) v6[t] += w.cdof[6*j + t] * s.qvel[j];
                        k++;
                    }
                }
            }
            float ci[10];
            for (int t = 0; t < 10; t++) ci[t] = w.cinert[10*i + t];
            float tmp[6], tmp1[6], cf[6];
            mj_mulInertVec(tmp, ci, acc);
            mj_mulInertVec(tmp1, ci, v6);
            mj_crossForce(cf, v6, tmp1);
            for (int c = 0; c < 6; c++) {
                cvel[6*i + c] = v6[c];
                cfrc[6*i + c] = tmp[c] + cf[c];
            }
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
        for (int d0 = 0; d0 < int(NV); d0 += int(snum)) {
            int v = d0 + int(slane);
            if (v >= int(NV)) continue;
            int b = m.dof_bodyid[v];
            float fb[6];
            for (int c = 0; c < 6; c++) fb[c] = cfrc[6*b + c];
            for (int j = b + 1; j < int(NBODY); j++) {
                if (!bodyUnder(m, b, j)) continue;
                for (int c = 0; c < 6; c++) fb[c] += cfrc[6*j + c];
            }
            w.qfrc_bias[v] = mj_dot6(w.cdof + 6*v, fb);
        }
        if (snum == 32u) simdgroup_barrier(mem_flags::mem_device);
        for (int v = 0; v < int(NV); v++) qfrcBiasT[v] = w.qfrc_bias[v];
    }

    template<typename PX, typename PL, typename PD>
    static void solveLDx(thread const Model &m, PX x, PL LD, PD diagInv) {
        for (int i = int(NV) - 1; i >= 0; i--) {
            if (m.M_rownnz[i] == 1) continue;
            float xi = x[i];
            if (xi != 0.0f) {
                int start = m.M_rowadr[i];
                int nn = m.M_rownnz[i] - 1;
                for (int k = 0; k < nn; k++) x[m.M_colind[start + k]] -= LD[start + k] * xi;
            }
        }
        for (int i = 0; i < int(NV); i++) x[i] *= diagInv[i];
        for (int i = 0; i < int(NV); i++) {
            if (m.M_rownnz[i] == 1) continue;
            int start = m.M_rowadr[i];
            int nn = m.M_rownnz[i] - 1;
            float acc = 0.0f;
            for (int k = 0; k < nn; k++) acc += LD[start + k] * x[m.M_colind[start + k]];
            x[i] -= acc;
        }
    }

    static float rowCostOwn(thread const Work &w, int nefc, thread const float *jarOwn, uint slane) {
        float c = 0.0f;
        int k = 0;
        for (int i = int(slane); i < nefc; i += 32, k++) {
            float j = jarOwn[k];
            if (j < 0.0f) c = fma(0.5f*w.efc_D[i]*j, j, c);
        }
        return simd_sum(c);
    }

    static void jtForce(thread const Model &m, thread Work &w, device const float *Jar,
                        thread float *qfc, uint slane) {
        int nefc = w.nefc;
        float part[NV];
        for (int k = 0; k < int(NV); k++) part[k] = 0.0f;
        for (int r = int(slane); r < nefc; r += 32) {
            float jar = Jar[r];
            if (jar >= 0.0f) continue;
            float t = -w.efc_D[r]*jar;
            device const float *row = w.efc_J + int(NV)*r;
            for (int k = 0; k < int(NV); k++) part[k] = fma(row[k], t, part[k]);
        }
        for (int k = 0; k < int(NV); k++) qfc[k] = simd_sum(part[k]);
        if (int(slane) < int(NV)) w.qfrc_constraint[int(slane)] = qfc[int(slane)];
    }

    static void hessFactorRows(thread const Model &m, thread Work &w, device const float *Jar,
                               thread const float *mrow, thread float *hrow, uint slane) {
        int i = int(slane);
        int nefc = w.nefc;
        for (int c = 0; c < int(NV); c++) hrow[c] = (i < int(NV) && c <= i) ? mrow[c] : 0.0f;
        if (i < int(NV)) {
            for (int r = 0; r < nefc; r++) {
                float jar = Jar[r];
                if (jar >= 0.0f) continue;
                float t = w.efc_J[int(NV)*r + i]*w.efc_D[r];
                if (t == 0.0f) continue;
                device const packed_float4 *j4 = (device const packed_float4 *)(w.efc_J + int(NV)*r);
                constexpr int NB4 = int(NV)/4;
                int lim4 = (i + 4)/4;
                if (lim4 > NB4) lim4 = NB4;
                for (int k = 0; k < lim4; k++) {
                    float4 jv = float4(j4[k]);
                    hrow[4*k + 0] = fma(jv.x, t, hrow[4*k + 0]);
                    hrow[4*k + 1] = fma(jv.y, t, hrow[4*k + 1]);
                    hrow[4*k + 2] = fma(jv.z, t, hrow[4*k + 2]);
                    hrow[4*k + 3] = fma(jv.w, t, hrow[4*k + 3]);
                }
                for (int c = 4*NB4; c <= i; c++) hrow[c] = fma(w.efc_J[int(NV)*r + c], t, hrow[c]);
            }
        }
        cholFactorRowsDist(hrow, slane);
    }

    static void newton(thread const Model &m, thread Work &w, thread float *qacc,
                       thread const float *mrow, thread const float *qfrcSmooth,
                       thread const float *qaccSmooth, uint slane) {
        int nefc = w.nefc;
        device float *Jar = w.efc_pos;
        float Ma[NV];
        mulMVecDist(mrow, (thread const float *)qacc, Ma, slane);
        constexpr int NOWN = (int(NEFC) + 31) / 32;
        float jarOwn[NOWN], jvOwn[NOWN], quadOwn[3*NOWN];
        {
            int k = 0;
            for (int i = int(slane); i < nefc; i += 32, k++) {
                float j = mj_dotDevRow<NV>(w.efc_J + int(NV)*i, (thread const float *)qacc, -w.efc_aref[i]);
                jarOwn[k] = j;
                Jar[i] = j;
            }
        }
        simdgroup_barrier(mem_flags::mem_device);
        float costC = rowCostOwn(w, nefc, jarOwn, slane);
        float qfc[NV];
        jtForce(m, w, Jar, qfc, slane);
        float gauss = mj_gaussCost<NV>(qfrcSmooth, qaccSmooth, Ma, qacc);
        float cost = costC + gauss;
        float hrow[NV];
        hessFactorRows(m, w, Jar, mrow, hrow, slane);
        float grad[NV], Mgrad[NV], search[NV];
        for (int v = 0; v < int(NV); v++) grad[v] = Ma[v] - qfrcSmooth[v] - qfc[v];
        cholSolveRowsDist(hrow, (thread const float *)grad, Mgrad, slane);
        for (int v = 0; v < int(NV); v++) search[v] = -Mgrad[v];

        int iter = 0;
        while (iter < MJ_SOLVER_ITER) {
            float snorm = sqrt(mj_dotN<NV>(search, search));
            float Mv[NV];
            mulMVecDist(mrow, (thread const float *)search, Mv, slane);
            {
                int k = 0;
                for (int i = int(slane); i < nefc; i += 32, k++) {
                    float jv = mj_dotDevRow<NV>(w.efc_J + int(NV)*i, (thread const float *)search, 0.0f);
                    jvOwn[k] = jv;
                    float Dj = w.efc_D[i];
                    quadOwn[3*k + 0] = 0.5f*Dj*jarOwn[k]*jarOwn[k];
                    quadOwn[3*k + 1] = Dj*jarOwn[k]*jv;
                    quadOwn[3*k + 2] = 0.5f*Dj*jv*jv;
                }
            }
            float quadGauss[3];
            quadGauss[0] = gauss;
            {
                float aq = 0.0f;
                for (int v = 0; v < int(NV); v++) aq = fma(search[v], Ma[v], aq);
                for (int v = 0; v < int(NV); v++) aq = fma(-qfrcSmooth[v], search[v], aq);
                quadGauss[1] = aq;
            }
            quadGauss[2] = 0.5f*mj_dotNf<NV>((thread const float *)search, (thread const float *)Mv, 0.0f);
            float alpha = mj_primalSearch(nefc, jarOwn, jvOwn, quadOwn, quadGauss, snorm, w.scale, slane);
            if (alpha == 0.0f) break;
            for (int v = 0; v < int(NV); v++) {
                qacc[v] += alpha*search[v];
                Ma[v] += alpha*Mv[v];
            }
            int nflip = 0;
            {
                int k = 0;
                for (int i = int(slane); i < nefc; i += 32, k++) {
                    bool oldneg = jarOwn[k] < 0.0f;
                    jarOwn[k] += alpha*jvOwn[k];
                    Jar[i] = jarOwn[k];
                    if (oldneg != (jarOwn[k] < 0.0f)) nflip++;
                }
            }
            simdgroup_barrier(mem_flags::mem_device);
            int totalFlips = simd_sum(nflip);
            float oldcost = cost;
            costC = rowCostOwn(w, nefc, jarOwn, slane);
            jtForce(m, w, Jar, qfc, slane);
            gauss = mj_gaussCost<NV>(qfrcSmooth, qaccSmooth, Ma, qacc);
            cost = costC + gauss;
            if (totalFlips > 0) hessFactorRows(m, w, Jar, mrow, hrow, slane);
            for (int v = 0; v < int(NV); v++) grad[v] = Ma[v] - qfrcSmooth[v] - qfc[v];
            cholSolveRowsDist(hrow, (thread const float *)grad, Mgrad, slane);
            float improvement = w.scale*(oldcost - cost);
            float gradient = w.scale*sqrt(mj_dotN<NV>(grad, grad));
            iter++;
            if (improvement < MJ_SOLVER_TOL || gradient < MJ_SOLVER_TOL) break;
            for (int v = 0; v < int(NV); v++) search[v] = -Mgrad[v];
        }
    }

    static float warmCost(thread const Model &m, thread Work &w, thread const float *cand,
                          thread const float *mrow, thread const float *qfrcSmooth,
                          thread const float *qaccSmooth, uint slane) {
        int nefc = w.nefc;
        float c = 0.0f;
        for (int i = int(slane); i < nefc; i += 32) {
            float jar = mj_dotDevRow<NV>(w.efc_J + int(NV)*i, cand, -w.efc_aref[i]);
            if (jar < 0.0f) c = fma(0.5f*w.efc_D[i]*jar, jar, c);
        }
        float cost = simd_sum(c);
        float Mc[NV];
        mulMVecDist(mrow, cand, Mc, slane);
        float g = cost;
        for (int v = 0; v < int(NV); v++)
            g = fma(0.5f*(Mc[v] - qfrcSmooth[v]), cand[v] - qaccSmooth[v], g);
        return g;
    }

    static void fwdConstraint(thread const Model &m, State s, thread Work &w,
                              thread const float *mrow, thread const float *qfrcSmooth,
                              thread const float *qaccSmooth, uint slane) {
        int nefc = w.nefc;
        if (nefc == 0) {
            if (slane == 0u) {
                for (int v = 0; v < int(NV); v++) {
                    w.qfrc_constraint[v] = 0.0f;
                    w.qacc[v] = qaccSmooth[v];
                }
            }
            simdgroup_barrier(mem_flags::mem_device);
            return;
        }
        float ws[NV];
        for (int v = 0; v < int(NV); v++) ws[v] = s.qacc_warmstart[v];
        float costWs = warmCost(m, w, (thread const float *)ws, mrow, qfrcSmooth, qaccSmooth, slane);
        float costSmooth = warmCost(m, w, qaccSmooth, mrow, qfrcSmooth, qaccSmooth, slane);
        float qacc[NV];
        if (costWs > costSmooth) {
            for (int v = 0; v < int(NV); v++) qacc[v] = qaccSmooth[v];
        } else {
            for (int v = 0; v < int(NV); v++) qacc[v] = ws[v];
        }
        newton(m, w, qacc, mrow, qfrcSmooth, qaccSmooth, slane);
        for (int v = int(slane); v < int(NV); v += 32) w.qacc[v] = qacc[v];
        simdgroup_barrier(mem_flags::mem_device);
    }

    static void addFluid(thread const Model &m, thread Work &w, thread float *qf) {
        for (int v = 0; v < int(NV); v++) qf[v] = 0.0f;
        float dens = m.opt[2];
        float visc = m.opt[3];
        if (dens == 0.0f && visc == 0.0f) return;
        for (int i = 1; i < int(NBODY); i++) {
            if (m.body_mass[i] < MJ_MINVAL) continue;
            int r = m.body_rootid[i];
            float in0 = m.body_inertia[3*i], in1 = m.body_inertia[3*i + 1], in2 = m.body_inertia[3*i + 2];
            float invMass = 1.0f / m.body_mass[i];
            float box[3];
            box[0] = sqrt(max(MJ_MINVAL, (in1 + in2 - in0)) * invMass * 6.0f);
            box[1] = sqrt(max(MJ_MINVAL, (in0 + in2 - in1)) * invMass * 6.0f);
            box[2] = sqrt(max(MJ_MINVAL, (in0 + in1 - in2)) * invMass * 6.0f);
            float dif[3] = { w.xipos[3*i] - w.subCom[3*r], w.xipos[3*i + 1] - w.subCom[3*r + 1], w.xipos[3*i + 2] - w.subCom[3*r + 2] };
            float cv[6];
            for (int k = 0; k < 6; k++) cv[k] = w.cvel[6*i + k];
            float crossv[3];
            float rotv[3] = { cv[0], cv[1], cv[2] };
            mj_cross(crossv, dif, rotv);
            float tran[6] = { cv[0], cv[1], cv[2], cv[3] - crossv[0], cv[4] - crossv[1], cv[5] - crossv[2] };
            float lvel[6];
            device const float *R = w.ximat + 9*i;
            for (int k = 0; k < 3; k++) {
                lvel[k] = R[0*3 + k]*tran[0] + R[1*3 + k]*tran[1] + R[2*3 + k]*tran[2];
                lvel[3 + k] = R[0*3 + k]*tran[3] + R[1*3 + k]*tran[4] + R[2*3 + k]*tran[5];
            }
            float windW[3] = { m.opt[4], m.opt[5], m.opt[6] };
            float lwind[3];
            for (int k = 0; k < 3; k++)
                lwind[k] = R[0*3 + k]*windW[0] + R[1*3 + k]*windW[1] + R[2*3 + k]*windW[2];
            lvel[3] -= lwind[0]; lvel[4] -= lwind[1]; lvel[5] -= lwind[2];
            float lfrc[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            if (visc > 0.0f) {
                float diam = (box[0] + box[1] + box[2]) / 3.0f;
                float aScl = -M_PI_F*diam*diam*diam*visc;
                lfrc[0] = lvel[0]*aScl; lfrc[1] = lvel[1]*aScl; lfrc[2] = lvel[2]*aScl;
                float lScl = -3.0f*M_PI_F*diam*visc;
                lfrc[3] = lvel[3]*lScl; lfrc[4] = lvel[4]*lScl; lfrc[5] = lvel[5]*lScl;
            }
            if (dens > 0.0f) {
                lfrc[3] -= 0.5f*dens*box[1]*box[2]*fabs(lvel[3])*lvel[3];
                lfrc[4] -= 0.5f*dens*box[0]*box[2]*fabs(lvel[4])*lvel[4];
                lfrc[5] -= 0.5f*dens*box[0]*box[1]*fabs(lvel[5])*lvel[5];
                lfrc[0] -= dens*box[0]*(box[1]*box[1]*box[1]*box[1] + box[2]*box[2]*box[2]*box[2])*
                           fabs(lvel[0])*lvel[0]/64.0f;
                lfrc[1] -= dens*box[1]*(box[0]*box[0]*box[0]*box[0] + box[2]*box[2]*box[2]*box[2])*
                           fabs(lvel[1])*lvel[1]/64.0f;
                lfrc[2] -= dens*box[2]*(box[0]*box[0]*box[0]*box[0] + box[1]*box[1]*box[1]*box[1])*
                           fabs(lvel[2])*lvel[2]/64.0f;
            }
            float btorque[3], bforce[3];
            for (int k = 0; k < 3; k++) {
                btorque[k] = R[3*k + 0]*lfrc[0] + R[3*k + 1]*lfrc[1] + R[3*k + 2]*lfrc[2];
                bforce[k] = R[3*k + 0]*lfrc[3] + R[3*k + 1]*lfrc[4] + R[3*k + 2]*lfrc[5];
            }
            float off[3] = { w.xipos[3*i] - w.subCom[3*r], w.xipos[3*i + 1] - w.subCom[3*r + 1], w.xipos[3*i + 2] - w.subCom[3*r + 2] };
            int d = m.body_lastdof[m.body_weldid[i]];
            while (d >= 0) {
                device const float *cd = w.cdof + 6*d;
                float jacp[3] = { cd[3] + (cd[1]*off[2] - cd[2]*off[1]),
                                  cd[4] + (cd[2]*off[0] - cd[0]*off[2]),
                                  cd[5] + (cd[0]*off[1] - cd[1]*off[0]) };
                qf[d] += jacp[0]*bforce[0] + jacp[1]*bforce[1] + jacp[2]*bforce[2]
                       + cd[0]*btorque[0] + cd[1]*btorque[1] + cd[2]*btorque[2];
                d = m.dof_parentid[d];
            }
        }
    }

    static void forward(thread const Model &m, State s, thread Work &w, uint slane) {
        kinematics(m, s, w, slane, 32u);
        comPos(m, s, w, slane, 32u);
        collision(m, s, w, slane);
        float qfrcBiasT[NV];
        comVelRne(m, s, w, qfrcBiasT, slane, 32u);
        float mrow[NV];
        makeM(m, w, mrow, slane);
        w.scale = 1.0f / simd_sum((int(slane) < int(NV)) ? mrow[int(slane)] : 0.0f);
        float qf[NV];
        addFluid(m, w, qf);
        float qfrcAct[NV];
        for (int v = 0; v < int(NV); v++) qfrcAct[v] = 0.0f;
        {
            float cl[NU];
            bool bad = false;
            for (int u = 0; u < int(NU); u++) {
                cl[u] = max(m.actuator_ctrlrange[2*u + 0], min(m.actuator_ctrlrange[2*u + 1], s.ctrl[u]));
                if (mj_isBad(cl[u])) bad = true;
            }
            if (bad)
                for (int u = 0; u < int(NU); u++) cl[u] = 0.0f;
            for (int u = 0; u < int(NU); u++)
                qfrcAct[m.jnt_dofadr[m.actuator_trnid[u]]] += m.actuator_gear[u]*cl[u];
        }
        for (int d0 = 0; d0 < int(NV); d0 += 32) {
            int v = d0 + int(slane);
            if (v >= int(NV)) continue;
            int j = m.dof_jntid[v];
            int jt = m.jnt_type[j];
            float spring = 0.0f;
            if (jt == 2 || jt == 3) {
                int qadr = m.jnt_qposadr[j];
                spring = -m.jnt_stiffness[j]*(s.qpos[qadr] - m.qpos_spring[qadr]);
            }
            float damper = -m.dof_damping[v]*s.qvel[v];
            w.qfrc_smooth[v] = spring + damper + qf[v] - qfrcBiasT[v] + qfrcAct[v];
        }
        makeConstraint(m, s, w, slane);
        float rhs[NV];
        for (int v = 0; v < int(NV); v++) rhs[v] = w.qfrc_smooth[v];
        float lrow[NV];
        for (int c = 0; c < int(NV); c++) lrow[c] = (int(slane) < int(NV) && c <= int(slane)) ? mrow[c] : 0.0f;
        cholFactorRowsDist(lrow, slane);
        float qsm[NV];
        cholSolveRowsDist((thread const float *)lrow, (thread const float *)rhs, qsm, slane);
        for (int v = int(slane); v < int(NV); v += 32) w.qacc_smooth[v] = qsm[v];
        simdgroup_barrier(mem_flags::mem_device);

        fwdConstraint(m, s, w, (thread const float *)mrow, (thread const float *)rhs,
                      (thread const float *)qsm, slane);
    }

    template<typename PV>
    static void integratePos(thread const Model &m, State s, PV vel, float h, uint slane, uint snum) {
        for (int j = int(slane); j < int(NJNT); j += int(snum)) {
            int padr = m.jnt_qposadr[j];
            int vadr = m.jnt_dofadr[j];
            int jt = m.jnt_type[j];
            if (jt == 0) {
                for (int k = 0; k < 3; k++) s.qpos[padr + k] += h*vel[vadr + k];
                float quat[4] = { s.qpos[padr + 3], s.qpos[padr + 4], s.qpos[padr + 5], s.qpos[padr + 6] };
                float w3[3] = { vel[vadr + 3], vel[vadr + 4], vel[vadr + 5] };
                mj_quatIntegrate(quat, w3, h);
                for (int k = 0; k < 4; k++) s.qpos[padr + 3 + k] = quat[k];
            } else {
                s.qpos[padr] += h*vel[vadr];
            }
        }
    }

    static void clearState(thread const Model &m, State s) {
        for (int v = 0; v < int(NQ); v++) s.qpos[v] = m.qpos0[v];
        for (int v = 0; v < int(NV); v++) {
            s.qvel[v] = 0.0f;
            s.qacc_warmstart[v] = 0.0f;
        }
        for (int u = 0; u < int(NU); u++) s.ctrl[u] = 0.0f;
    }

    static bool isBadVecQ(device const float *x) {
        bool bad = false;
        for (int i = 0; i < int(NQ); i++) bad = bad || mj_isBad(x[i]);
        return bad;
    }
    static bool isBadVecV(device const float *x) {
        bool bad = false;
        for (int i = 0; i < int(NV); i++) bad = bad || mj_isBad(x[i]);
        return bad;
    }

    static void stepPhysics(thread const Model &m, State s, thread Work &w, uint slane) {
        bool bad = isBadVecQ(s.qpos) || isBadVecV(s.qvel);
        if (bad) {
            if (slane == 0u) clearState(m, s);
            simdgroup_barrier(mem_flags::mem_device);
        }
        forward(m, s, w, slane);
        if (isBadVecV(w.qacc)) {
            if (slane == 0u) clearState(m, s);
            simdgroup_barrier(mem_flags::mem_device);
            forward(m, s, w, slane);
        }
        if (slane == 0u) {
            for (int k = 0; k < int(NM); k++) w.qLD[k] = w.M[k];
            for (int i = 0; i < int(NV); i++)
                w.qLD[m.M_rowadr[i] + m.M_rownnz[i] - 1] += m.timestep*m.dof_damping[i];
            factorI(m, w.qLD, w.qLDiagInv);
            float qaccInt[NV];
            for (int v = 0; v < int(NV); v++) qaccInt[v] = w.qfrc_smooth[v] + w.qfrc_constraint[v];
            solveLDx(m, (thread float *)qaccInt, w.qLD, w.qLDiagInv);
            float vv[NV];
            for (int v = 0; v < int(NV); v++) {
                vv[v] = s.qvel[v] + m.timestep*qaccInt[v];
                s.qvel[v] = vv[v];
            }
            integratePos(m, s, (thread const float *)vv, m.timestep, 0u, 1u);
            for (int v = 0; v < int(NV); v++) s.qacc_warmstart[v] = w.qacc[v];
        }
        simdgroup_barrier(mem_flags::mem_device);
    }
};

#pragma METAL fp math_mode(safe)


template<uint N>
inline void mj_rngUAdd(uint seed, uint resetIdx, uint ctrBase, float lo, float hi,
                       constant float *base, device float *dst) {
    uint2 key = uint2(seed, RNG_STREAM_RESET);
    constexpr uint NB4 = (N + 3u) / 4u;
    for (uint b = 0u; b < NB4; b++) {
        uint4 r = philox4x32(key, uint4(resetIdx, ctrBase + b, 0u, 0u));
        uint raw[4] = { r.x, r.y, r.z, r.w };
        for (uint t = 0u; t < 4u; t++) {
            uint idx = 4u*b + t;
            if (idx < N) dst[idx] = base[idx] + uniformRange(raw[t], lo, hi);
        }
    }
}

template<uint N>
inline void mj_rngNSet(uint seed, uint resetIdx, uint ctrBase, float scale, device float *dst) {
    uint2 key = uint2(seed, RNG_STREAM_RESET);
    constexpr uint NP = (N + 1u) / 2u;
    constexpr uint NB4 = (2u*NP + 3u) / 4u;
    for (uint b = 0u; b < NB4; b++) {
        uint4 r = philox4x32(key, uint4(resetIdx, ctrBase + b, 0u, 0u));
        float u1 = u01open(r.x);
        float u2 = u01(r.y);
        float rad = sqrt(-2.0f * log(u1));
        float ang = 2.0f * M_PI_F * u2;
        uint idx = 4u*b;
        if (idx < N) dst[idx] = scale * (rad * cos(ang));
        if (idx + 1u < N) dst[idx + 1u] = scale * (rad * sin(ang));
        u1 = u01open(r.z);
        u2 = u01(r.w);
        rad = sqrt(-2.0f * log(u1));
        ang = 2.0f * M_PI_F * u2;
        if (idx + 2u < N) dst[idx + 2u] = scale * (rad * cos(ang));
        if (idx + 3u < N) dst[idx + 3u] = scale * (rad * sin(ang));
    }
}

using E0 = MjEngine<9, 9, 6, 8, 9, 9, 8, 16, 70, 36>;

inline void mjR_cheetah(thread const E0::Model &m, E0::State s, uint seed,
                        thread uint &elapsed, thread uint &resets) {
    E0::clearState(m, s);
    mj_rngUAdd<9>(seed, resets, 0u, -m.spec[21], m.spec[21], m.qpos0, s.qpos);
    mj_rngNSet<9>(seed, resets, 8u, m.spec[21], s.qvel);
    resets++;
    elapsed = 0u;
}

inline void mjObs_cheetah(E0::State s, device float *obs) {
    for (int i = 1; i < 9; i++) obs[i - 1] = s.qpos[i];
    for (int i = 0; i < 9; i++) obs[8 + i] = s.qvel[i];
}

inline float mjStep_cheetah(thread const E0::Model &m, E0::State s, thread E0::Work &w,
                        thread const float *ctrl, device float *obs, thread uint &term, uint slane) {
    float xb = s.qpos[0];
    for (int u = 0; u < 6; u++) s.ctrl[u] = ctrl[u];
    int fs = int(m.spec[7]);
    for (int k = 0; k < fs; k++) E0::stepPhysics(m, s, w, slane);
    float dt = m.spec[7] * m.timestep;
    float xv = (s.qpos[0] - xb) / dt;
    float ctrlCost = 0.0f;
    for (int u = 0; u < 6; u++) ctrlCost += m.spec[4]*ctrl[u]*ctrl[u];
    mjObs_cheetah(s, obs);
    term = 0u;
    return m.spec[6]*xv - ctrlCost;
}

inline void mj_reset(uint envKind, constant uint *mjModels, device uint *envState,
                     uint seed, thread uint &elapsed, thread uint &resets, uint slane) {
    switch (envKind) {
    case 0u: {
        E0::Model m = E0::loadModel(mjModels, 0u);
        E0::State s = E0::bindState(envState);
        mjR_cheetah(m, s, seed, elapsed, resets);
        break;
    }
    default:
        envState[0] = as_type<uint>(NAN);
        break;
    }
}

inline void mj_observe(uint envKind, constant uint *mjModels, device const uint *envState,
                       device float *obs, uint slane) {
    switch (envKind) {
    case 0u: {
        E0::State s = E0::bindState((device uint *)envState);
        mjObs_cheetah(s, obs);
        break;
    }
    default:
        obs[0] = NAN;
        break;
    }
}

inline float mj_step_env(uint envKind, constant uint *mjModels, device uint *envState,
                         device float *scratch, uint seed, thread const float *ctrl,
                         thread uint &elapsed, thread uint &resets,
                         device float *obs, thread uint &done, uint slane) {
    done = 0u;
    switch (envKind) {
    case 0u: {
        E0::Model m = E0::loadModel(mjModels, 0u);
        E0::State s = E0::bindState(envState);
        E0::Work w = E0::bindWork(scratch);
        float reward;
        uint term = 0u;
        reward = mjStep_cheetah(m, s, w, ctrl, obs, term, slane);
        elapsed++;
        done = term != 0u ? 1u : 0u;
        reward = simd_broadcast_first(reward);
        done = simd_broadcast_first(done);
        elapsed = simd_broadcast_first(elapsed);
        resets = simd_broadcast_first(resets);
        return reward;
    }
    default:
        obs[0] = NAN;
        return NAN;
    }
}

inline void mj_poseQpos(device const uint *envState, uint nq, thread float *pose) {
    device const float *qpos = (device const float *)envState;
    uint n = min(nq, OAK_POSE_LEN);
    for (uint i = 0u; i < n; i++) pose[i] = qpos[i];
    for (uint i = n; i < OAK_POSE_LEN; i++) pose[i] = 0.0f;
}

inline void mj_pose(uint envKind, constant uint *mjModels, device const uint *envState,
                    thread const float *ctrl, thread float *pose, uint slane) {
    switch (envKind) {
    case 0u: mj_poseQpos(envState, 9u, pose); break;
    default:
        for (uint i = 0u; i < OAK_POSE_LEN; i++) pose[i] = NAN;
        break;
    }
}
