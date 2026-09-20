#include "policy.h"

#define PPO_H 64u
#define PPO_T 128u
#define PPO_EPOCHS 4u
#define PPO_MB 32u
#define PPO_OMAX OAK_ENV_OBS_MAX
#define PPO_AMAX OAK_ENV_ACT_MAX
#define PPO_PARAM_STRIDE (((2u * PPO_OMAX * PPO_H + 2u * PPO_H + PPO_H * PPO_AMAX + 2u * PPO_AMAX + PPO_H + 1u) + 3u) & ~3u)
#define PPO_LR 3e-4f
#define PPO_GAMMA 0.99f
#define PPO_LAMBDA 0.95f
#define PPO_CLIP 0.2f
#define PPO_VF_COEF 0.5f
#define PPO_ENT_COEF_DISCRETE 0.01f
#define PPO_MAX_GRAD_NORM 0.5f
#define PPO_ADAM_B1 0.9f
#define PPO_ADAM_B2 0.999f
#define PPO_ADAM_EPS 1e-5f
#define PPO_NORM_EPS 1e-8f
#define PPO_NORM_CLIP 10.0f

#define PPO_ROLL_OBS 0u
#define PPO_ROLL_ACT ((PPO_T + 1u) * PPO_OMAX)
#define PPO_ROLL_LOGP (PPO_ROLL_ACT + PPO_T * PPO_AMAX)
#define PPO_ROLL_VAL (PPO_ROLL_LOGP + PPO_T)
#define PPO_ROLL_REW (PPO_ROLL_VAL + PPO_T)
#define PPO_ROLL_DONE (PPO_ROLL_REW + PPO_T)
#define PPO_ROLL_ADV (PPO_ROLL_DONE + PPO_T)
#define PPO_ROLL_RET (PPO_ROLL_ADV + PPO_T)
#define PPO_ROLL_STRIDE (((PPO_ROLL_RET + PPO_T) + 15u) & ~15u)

#define PPO_SCR_OBSN 0u
#define PPO_SCR_HA (PPO_MB * PPO_OMAX)
#define PPO_SCR_HC (PPO_SCR_HA + PPO_MB * PPO_H)
#define PPO_SCR_DHA (PPO_SCR_HC + PPO_MB * PPO_H)
#define PPO_SCR_DHC (PPO_SCR_DHA + PPO_MB * PPO_H)
#define PPO_SCR_DLG (PPO_SCR_DHC + PPO_MB * PPO_H)
#define PPO_SCR_DLS (PPO_SCR_DLG + PPO_MB * PPO_AMAX)
#define PPO_SCR_DV (PPO_SCR_DLS + PPO_MB * PPO_AMAX)
#define PPO_SCR_IDX (PPO_SCR_DV + PPO_MB)
#define PPO_SCR_GRAD (PPO_SCR_IDX + PPO_T)
#define PPO_SCRATCH_STRIDE ((((PPO_SCR_GRAD + PPO_PARAM_STRIDE) + 15u) & ~15u) + 16u)

#define PPO_ST_OBSMEAN 0u
#define PPO_ST_OBSM2 PPO_OMAX
#define PPO_ST_OBSCOUNT (2u * PPO_OMAX)
#define PPO_ST_RETTRACE (2u * PPO_OMAX + 1u)
#define PPO_ST_RETMEAN (2u * PPO_OMAX + 2u)
#define PPO_ST_RETM2 (2u * PPO_OMAX + 3u)
#define PPO_ST_RETCOUNT (2u * PPO_OMAX + 4u)
#define PPO_ST_ADAMT (2u * PPO_OMAX + 5u)
#define PPO_STATE_STRIDE (((2u * PPO_OMAX + 6u) + 15u) & ~15u)

#define AT(i) ((i) * OAK_LANES + lane)

typedef struct {
    uint32_t O, A, box;
    uint32_t w1a, b1a, w2a, b2a, logstd, w1c, b1c, w2c, b2c, count;
} PpoLayout;

typedef struct {
    float P[PPO_PARAM_STRIDE * OAK_LANES];
    float M[PPO_PARAM_STRIDE * OAK_LANES];
    float V[PPO_PARAM_STRIDE * OAK_LANES];
    float R[PPO_ROLL_STRIDE * OAK_LANES];
    float W[PPO_SCRATCH_STRIDE * OAK_LANES];
    float S[PPO_STATE_STRIDE * OAK_LANES];
    PpoLayout L[OAK_LANES];
} PpoAgent;

static PpoLayout ppoLayout(uint32_t env) {
    PpoLayout L;
    L.O = env_obs(env);
    L.A = env_actions(env);
    L.box = env_box(env);
    L.w1a = 0u;
    L.b1a = L.w1a + L.O * PPO_H;
    L.w2a = L.b1a + PPO_H;
    L.b2a = L.w2a + PPO_H * L.A;
    L.logstd = L.b2a + L.A;
    L.w1c = L.logstd + (L.box ? L.A : 0u);
    L.b1c = L.w1c + L.O * PPO_H;
    L.w2c = L.b1c + PPO_H;
    L.b2c = L.w2c + PPO_H;
    L.count = L.b2c + 1u;
    return L;
}

static float ppoGauss(uint32_t seed, uint32_t stream, uint32_t c0, uint32_t c1) {
    u4 r = philox4x32(seed, stream, c0, c1, 0u, 0u);
    float u1 = u01open(r.x);
    float u2 = u01(r.y);
    return sqrtf(-2.0f * oak_log(u1)) * oak_cos(2.0f * OAK_PI * u2);
}

static void ppoNormObs(uint32_t lane, device const float *S, uint32_t O, device const float *raw, device float *outN) {
    float count = fmaxf(S[AT(PPO_ST_OBSCOUNT)], 1.0f);
    for (uint32_t d = 0u; d < O; d++) {
        float variance = S[AT(PPO_ST_OBSM2 + d)] / count;
        float n = (raw[AT(d)] - S[AT(PPO_ST_OBSMEAN + d)]) / sqrtf(variance + PPO_NORM_EPS);
        outN[AT(d)] = oak_clamp(n, -PPO_NORM_CLIP, PPO_NORM_CLIP);
    }
}

static void ppoWelfordObs(uint32_t lane, device float *S, uint32_t O, device const float *raw) {
    float count = S[AT(PPO_ST_OBSCOUNT)] + 1.0f;
    S[AT(PPO_ST_OBSCOUNT)] = count;
    for (uint32_t d = 0u; d < O; d++) {
        float delta = raw[d] - S[AT(PPO_ST_OBSMEAN + d)];
        S[AT(PPO_ST_OBSMEAN + d)] += delta / count;
        S[AT(PPO_ST_OBSM2 + d)] += delta * (raw[d] - S[AT(PPO_ST_OBSMEAN + d)]);
    }
}

static float ppoScaleReward(uint32_t lane, device float *S, float reward, uint32_t done) {
    float trace = S[AT(PPO_ST_RETTRACE)] * PPO_GAMMA * (done != 0u ? 0.0f : 1.0f) + reward;
    S[AT(PPO_ST_RETTRACE)] = trace;
    float count = S[AT(PPO_ST_RETCOUNT)] + 1.0f;
    S[AT(PPO_ST_RETCOUNT)] = count;
    float delta = trace - S[AT(PPO_ST_RETMEAN)];
    S[AT(PPO_ST_RETMEAN)] += delta / count;
    S[AT(PPO_ST_RETM2)] += delta * (trace - S[AT(PPO_ST_RETMEAN)]);
    float variance = S[AT(PPO_ST_RETM2)] / count;
    return oak_clamp(reward / sqrtf(variance + PPO_NORM_EPS), -PPO_NORM_CLIP, PPO_NORM_CLIP);
}

static float ppoForward(uint32_t lane, device const float *P, thread const PpoLayout *L, device const float *obsn, device float *haOut, device float *hcOut, thread float *logits) {
    float actor[PPO_H], critic[PPO_H];
    for (uint32_t hu = 0u; hu < PPO_H; hu++) { actor[hu] = P[AT(L->b1a + hu)]; critic[hu] = P[AT(L->b1c + hu)]; }
    for (uint32_t o = 0u; o < L->O; o++) {
        device const float *wa = P + (L->w1a + o * PPO_H) * OAK_LANES, *wc = P + (L->w1c + o * PPO_H) * OAK_LANES;
        float x = obsn[AT(o)];
        for (uint32_t hu = 0u; hu < PPO_H; hu++) { actor[hu] += wa[AT(hu)] * x; critic[hu] += wc[AT(hu)] * x; }
    }
    for (uint32_t hu = 0u; hu < PPO_H; hu++) { haOut[AT(hu)] = oak_tanh(actor[hu]); hcOut[AT(hu)] = oak_tanh(critic[hu]); }
    for (uint32_t a = 0u; a < L->A; a++) logits[a] = P[AT(L->b2a + a)];
    for (uint32_t hu = 0u; hu < PPO_H; hu++) {
        device const float *w = P + (L->w2a + hu * L->A) * OAK_LANES;
        float h = haOut[AT(hu)];
        for (uint32_t a = 0u; a < L->A; a++) logits[a] += w[AT(a)] * h;
    }
    float v = P[AT(L->b2c)];
    for (uint32_t hu = 0u; hu < PPO_H; hu++) v += P[AT(L->w2c + hu)] * hcOut[AT(hu)];
    return v;
}

uint32_t ppo_stateWords(void) { return (uint32_t)(sizeof(PpoAgent) / sizeof(uint32_t) / OAK_LANES); }

void ppo_init(device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, device const float *obs) {
    device PpoAgent *g = (device PpoAgent *)state;
    for (uint32_t w = 0u; w < ppo_stateWords(); w++) state[AT(w)] = 0u;
    PpoLayout layout = ppoLayout(env);
    g->L[lane] = layout;
    thread const PpoLayout *L = &layout;
    float stdIn = sqrtf(2.0f / (float)(L->O + PPO_H));
    float stdOutA = 0.01f * sqrtf(2.0f / (float)(PPO_H + L->A));
    float stdOutC = sqrtf(2.0f / (float)(PPO_H + 1u));
    for (uint32_t p = 0u; p < L->count; p++) {
        float std = 0.0f;
        if (p < L->b1a) std = stdIn;
        else if (p >= L->w2a && p < L->b2a) std = stdOutA;
        else if (p >= L->w1c && p < L->b1c) std = stdIn;
        else if (p >= L->w2c && p < L->b2c) std = stdOutC;
        g->P[AT(p)] = std == 0.0f ? 0.0f : std * ppoGauss(seed, RNG_STREAM_PPO_INIT, p, 0u);
    }
    for (uint32_t d = 0u; d < L->O; d++) g->R[AT(PPO_ROLL_OBS + d)] = obs[d];
    ppoWelfordObs(lane, g->S, L->O, obs);
}

static void ppoUpdate(device PpoAgent *g, uint32_t lane, uint32_t seed, uint32_t updateIdx) {
    PpoLayout layout = g->L[lane];
    thread const PpoLayout *L = &layout;
    device float *P = g->P, *M = g->M, *V = g->V, *R = g->R, *W = g->W, *S = g->S;
    float bootLogits[PPO_AMAX];
    ppoNormObs(lane, S, L->O, R + (PPO_ROLL_OBS + PPO_T * PPO_OMAX) * OAK_LANES, W + PPO_SCR_OBSN * OAK_LANES);
    float bootV = ppoForward(lane, P, L, W + PPO_SCR_OBSN * OAK_LANES, W + PPO_SCR_HA * OAK_LANES, W + PPO_SCR_HC * OAK_LANES, bootLogits);
    float lastGae = 0.0f;
    for (int ti = (int)PPO_T - 1; ti >= 0; ti--) {
        uint32_t t = (uint32_t)ti;
        float nextV = (ti == (int)PPO_T - 1) ? bootV : R[AT(PPO_ROLL_VAL + t + 1u)];
        float nonTerm = 1.0f - R[AT(PPO_ROLL_DONE + t)];
        float delta = R[AT(PPO_ROLL_REW + t)] + PPO_GAMMA * nextV * nonTerm - R[AT(PPO_ROLL_VAL + t)];
        lastGae = delta + PPO_GAMMA * PPO_LAMBDA * nonTerm * lastGae;
        R[AT(PPO_ROLL_ADV + t)] = lastGae;
        R[AT(PPO_ROLL_RET + t)] = lastGae + R[AT(PPO_ROLL_VAL + t)];
    }
    for (uint32_t epoch = 0u; epoch < PPO_EPOCHS; epoch++) {
        for (uint32_t i = 0u; i < PPO_T; i++) W[AT(PPO_SCR_IDX + i)] = (float)i;
        uint32_t block = 0u, wordIdx = 4u;
        u4 rb = { 0u, 0u, 0u, 0u };
        for (uint32_t i = PPO_T - 1u; i > 0u; i--) {
            if (wordIdx == 4u) {
                rb = philox4x32(seed, RNG_STREAM_PPO_SHUFFLE, updateIdx, epoch, block, 0u);
                block += 1u;
                wordIdx = 0u;
            }
            uint32_t words[4] = { rb.x, rb.y, rb.z, rb.w };
            uint32_t j = words[wordIdx] % (i + 1u);
            wordIdx += 1u;
            float tmp = W[AT(PPO_SCR_IDX + i)];
            W[AT(PPO_SCR_IDX + i)] = W[AT(PPO_SCR_IDX + j)];
            W[AT(PPO_SCR_IDX + j)] = tmp;
        }
        for (uint32_t mb = 0u; mb < PPO_T / PPO_MB; mb++) {
            float advSum = 0.0f;
            for (uint32_t ss = 0u; ss < PPO_MB; ss++) advSum += R[AT(PPO_ROLL_ADV + (uint32_t)W[AT(PPO_SCR_IDX + mb * PPO_MB + ss)])];
            float mbMean = advSum / (float)PPO_MB;
            float devSum = 0.0f;
            for (uint32_t ss = 0u; ss < PPO_MB; ss++) {
                float dev = R[AT(PPO_ROLL_ADV + (uint32_t)W[AT(PPO_SCR_IDX + mb * PPO_MB + ss)])] - mbMean;
                devSum += dev * dev;
            }
            float mbStd = sqrtf(devSum / (float)PPO_MB);
            for (uint32_t ss = 0u; ss < PPO_MB; ss++) {
                uint32_t s = (uint32_t)W[AT(PPO_SCR_IDX + mb * PPO_MB + ss)];
                float an = (R[AT(PPO_ROLL_ADV + s)] - mbMean) / (mbStd + PPO_NORM_EPS);
                device float *obsnRow = W + (PPO_SCR_OBSN + ss * PPO_OMAX) * OAK_LANES;
                ppoNormObs(lane, S, L->O, R + (PPO_ROLL_OBS + s * PPO_OMAX) * OAK_LANES, obsnRow);
                device float *haRow = W + (PPO_SCR_HA + ss * PPO_H) * OAK_LANES;
                device float *hcRow = W + (PPO_SCR_HC + ss * PPO_H) * OAK_LANES;
                float sLogits[PPO_AMAX];
                float v = ppoForward(lane, P, L, obsnRow, haRow, hcRow, sLogits);
                float oldLogp = R[AT(PPO_ROLL_LOGP + s)];
                float newLogp;
                float dlg[PPO_AMAX], dls[PPO_AMAX];
                if (L->box) {
                    newLogp = 0.0f;
                    for (uint32_t d = 0u; d < L->A; d++) {
                        float ls = P[AT(L->logstd + d)];
                        float sigma = oak_exp(ls);
                        float x = R[AT(PPO_ROLL_ACT + s * PPO_AMAX + d)];
                        float zd = (x - sLogits[d]) / sigma;
                        newLogp += -0.5f * zd * zd - ls - 0.9189385332f;
                        dlg[d] = zd / sigma;
                        dls[d] = zd * zd - 1.0f;
                    }
                    float ratio = oak_exp(newLogp - oldLogp);
                    int clippedOut = (an >= 0.0f && ratio > 1.0f + PPO_CLIP) || (an <= 0.0f && ratio < 1.0f - PPO_CLIP);
                    float dNewLogp = clippedOut ? 0.0f : -an * ratio;
                    for (uint32_t d = 0u; d < L->A; d++) {
                        dlg[d] *= dNewLogp;
                        dls[d] *= dNewLogp;
                    }
                } else {
                    float mx = sLogits[0];
                    for (uint32_t a = 1u; a < L->A; a++) mx = fmaxf(mx, sLogits[a]);
                    float denom = 0.0f;
                    for (uint32_t a = 0u; a < L->A; a++) denom += oak_exp(sLogits[a] - mx);
                    uint32_t act = (uint32_t)R[AT(PPO_ROLL_ACT + s * PPO_AMAX)];
                    newLogp = sLogits[act] - mx - oak_log(denom);
                    float entropy = 0.0f;
                    for (uint32_t a = 0u; a < L->A; a++) {
                        float pa = oak_exp(sLogits[a] - mx) / denom;
                        if (pa > 0.0f) entropy -= pa * oak_log(pa);
                    }
                    float ratio = oak_exp(newLogp - oldLogp);
                    int clippedOut = (an >= 0.0f && ratio > 1.0f + PPO_CLIP) || (an <= 0.0f && ratio < 1.0f - PPO_CLIP);
                    float dNewLogp = clippedOut ? 0.0f : -an * ratio;
                    for (uint32_t a = 0u; a < L->A; a++) {
                        float pa = oak_exp(sLogits[a] - mx) / denom;
                        float dEnt = PPO_ENT_COEF_DISCRETE * pa * (oak_log(fmaxf(pa, 1e-30f)) + entropy);
                        dlg[a] = dNewLogp * ((a == act ? 1.0f : 0.0f) - pa) + dEnt;
                        dls[a] = 0.0f;
                    }
                }
                float retS = R[AT(PPO_ROLL_RET + s)];
                float oldV = R[AT(PPO_ROLL_VAL + s)];
                float vClipped = oldV + oak_clamp(v - oldV, -PPO_CLIP, PPO_CLIP);
                float l1 = (v - retS) * (v - retS);
                float l2 = (vClipped - retS) * (vClipped - retS);
                float dV;
                if (l1 >= l2) dV = PPO_VF_COEF * (v - retS);
                else dV = fabsf(v - oldV) < PPO_CLIP ? PPO_VF_COEF * (vClipped - retS) : 0.0f;
                device float *dlgRow = W + (PPO_SCR_DLG + ss * PPO_AMAX) * OAK_LANES;
                device float *dlsRow = W + (PPO_SCR_DLS + ss * PPO_AMAX) * OAK_LANES;
                for (uint32_t a = 0u; a < L->A; a++) {
                    dlgRow[AT(a)] = dlg[a];
                    dlsRow[AT(a)] = dls[a];
                }
                W[AT(PPO_SCR_DV + ss)] = dV;
                device float *dhaRow = W + (PPO_SCR_DHA + ss * PPO_H) * OAK_LANES;
                device float *dhcRow = W + (PPO_SCR_DHC + ss * PPO_H) * OAK_LANES;
                for (uint32_t hu = 0u; hu < PPO_H; hu++) {
                    float accA = 0.0f;
                    for (uint32_t a = 0u; a < L->A; a++) accA += dlg[a] * P[AT(L->w2a + hu * L->A + a)];
                    float h = haRow[AT(hu)];
                    dhaRow[AT(hu)] = accA * (1.0f - h * h);
                    float hcv = hcRow[AT(hu)];
                    dhcRow[AT(hu)] = dV * P[AT(L->w2c + hu)] * (1.0f - hcv * hcv);
                }
            }
            device float *G = W + PPO_SCR_GRAD * OAK_LANES;
            device const float *dha = W + PPO_SCR_DHA * OAK_LANES, *dhc = W + PPO_SCR_DHC * OAK_LANES, *obsn = W + PPO_SCR_OBSN * OAK_LANES;
            device const float *ha = W + PPO_SCR_HA * OAK_LANES, *hc = W + PPO_SCR_HC * OAK_LANES, *dlg = W + PPO_SCR_DLG * OAK_LANES;
            device const float *dls = W + PPO_SCR_DLS * OAK_LANES, *dv = W + PPO_SCR_DV * OAK_LANES;
            for (uint32_t o = 0u; o < L->O; o++) {
                device const float *column = obsn + o * OAK_LANES;
                for (uint32_t hu = 0u; hu < PPO_H; hu += 16u) {
                    float a[16], c[16];
                    for (uint32_t k = 0u; k < 16u; k++) { a[k] = 0.0f; c[k] = 0.0f; }
                    for (uint32_t ss = 0u; ss < PPO_MB; ss++) {
                        float x = column[AT(ss * PPO_OMAX)];
                        device const float *da = dha + (ss * PPO_H + hu) * OAK_LANES, *dc = dhc + (ss * PPO_H + hu) * OAK_LANES;
                        for (uint32_t k = 0u; k < 16u; k++) { a[k] += da[AT(k)] * x; c[k] += dc[AT(k)] * x; }
                    }
                    device float *ga = G + (L->w1a + o * PPO_H + hu) * OAK_LANES, *gc = G + (L->w1c + o * PPO_H + hu) * OAK_LANES;
                    for (uint32_t k = 0u; k < 16u; k++) { ga[AT(k)] = a[k]; gc[AT(k)] = c[k]; }
                }
            }
            for (uint32_t hu = 0u; hu < PPO_H; hu += 16u) {
                float b[16], d[16], v[16];
                for (uint32_t k = 0u; k < 16u; k++) { b[k] = 0.0f; d[k] = 0.0f; v[k] = 0.0f; }
                for (uint32_t ss = 0u; ss < PPO_MB; ss++) {
                    device const float *da = dha + (ss * PPO_H + hu) * OAK_LANES, *dc = dhc + (ss * PPO_H + hu) * OAK_LANES, *h = hc + (ss * PPO_H + hu) * OAK_LANES;
                    float scale = dv[AT(ss)];
                    for (uint32_t k = 0u; k < 16u; k++) { b[k] += da[AT(k)]; d[k] += dc[AT(k)]; v[k] += scale * h[AT(k)]; }
                }
                for (uint32_t k = 0u; k < 16u; k++) { G[AT(L->b1a + hu + k)] = b[k]; G[AT(L->b1c + hu + k)] = d[k]; G[AT(L->w2c + hu + k)] = v[k]; }
            }
            for (uint32_t a = 0u; a < L->A; a++) {
                device const float *column = dlg + a * OAK_LANES;
                for (uint32_t hu = 0u; hu < PPO_H; hu += 16u) {
                    float w[16];
                    for (uint32_t k = 0u; k < 16u; k++) w[k] = 0.0f;
                    for (uint32_t ss = 0u; ss < PPO_MB; ss++) {
                        float g = column[AT(ss * PPO_AMAX)];
                        device const float *hAct = ha + (ss * PPO_H + hu) * OAK_LANES;
                        for (uint32_t k = 0u; k < 16u; k++) w[k] += g * hAct[AT(k)];
                    }
                    for (uint32_t k = 0u; k < 16u; k++) G[AT(L->w2a + (hu + k) * L->A + a)] = w[k];
                }
            }
            for (uint32_t a = 0u; a < L->A; a++) {
                float accG = 0.0f, accS = 0.0f;
                for (uint32_t ss = 0u; ss < PPO_MB; ss++) { accG += dlg[AT(ss * PPO_AMAX + a)]; accS += dls[AT(ss * PPO_AMAX + a)]; }
                G[AT(L->b2a + a)] = accG;
                if (L->box) G[AT(L->logstd + a)] = accS;
            }
            {
                float acc = 0.0f;
                for (uint32_t ss = 0u; ss < PPO_MB; ss++) acc += dv[AT(ss)];
                G[AT(L->b2c)] = acc;
            }
            float sumSq = 0.0f;
            for (uint32_t p = 0u; p < L->count; p++) {
                G[AT(p)] /= (float)PPO_MB;
                sumSq += G[AT(p)] * G[AT(p)];
            }
            float gNorm = sqrtf(sumSq);
            float scale = gNorm > PPO_MAX_GRAD_NORM ? PPO_MAX_GRAD_NORM / gNorm : 1.0f;
            float adamT = S[AT(PPO_ST_ADAMT)] + 1.0f;
            float bc1 = 1.0f - oak_exp(adamT * oak_log(PPO_ADAM_B1));
            float bc2 = 1.0f - oak_exp(adamT * oak_log(PPO_ADAM_B2));
            for (uint32_t p = 0u; p < L->count; p++) {
                float gp = W[AT(PPO_SCR_GRAD + p)] * scale;
                float m = PPO_ADAM_B1 * M[AT(p)] + (1.0f - PPO_ADAM_B1) * gp;
                float vv = PPO_ADAM_B2 * V[AT(p)] + (1.0f - PPO_ADAM_B2) * gp * gp;
                M[AT(p)] = m;
                V[AT(p)] = vv;
                P[AT(p)] -= PPO_LR * (m / bc1) / (sqrtf(vv / bc2) + PPO_ADAM_EPS);
            }
            S[AT(PPO_ST_ADAMT)] = adamT;
        }
    }
    for (uint32_t d = 0u; d < L->O; d++) R[AT(PPO_ROLL_OBS + d)] = R[AT(PPO_ROLL_OBS + PPO_T * PPO_OMAX + d)];
}

void ppo_act(device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, device const float *obs, float reward, uint32_t done, thread float *ctrl) {
    (void)env;
    device PpoAgent *g = (device PpoAgent *)state;
    PpoLayout layout = g->L[lane];
    thread const PpoLayout *L = &layout;
    device float *P = g->P, *R = g->R, *S = g->S;
    if (step > 0u) {
        uint32_t tPrev = (step - 1u) % PPO_T;
        float safeReward = isfinite(reward) ? reward : 0.0f;
        R[AT(PPO_ROLL_REW + tPrev)] = ppoScaleReward(lane, S, safeReward, done);
        R[AT(PPO_ROLL_DONE + tPrev)] = (float)done;
        for (uint32_t d = 0u; d < L->O; d++) R[AT(PPO_ROLL_OBS + (tPrev + 1u) * PPO_OMAX + d)] = obs[d];
        ppoWelfordObs(lane, S, L->O, obs);
        if (step % PPO_T == 0u) ppoUpdate(g, lane, seed, step / PPO_T);
    }
    uint32_t t = step % PPO_T;
    float logits[PPO_AMAX];
    device float *W = g->W;
    ppoNormObs(lane, S, L->O, R + (PPO_ROLL_OBS + t * PPO_OMAX) * OAK_LANES, W + PPO_SCR_OBSN * OAK_LANES);
    float value = ppoForward(lane, P, L, W + PPO_SCR_OBSN * OAK_LANES, W + PPO_SCR_HA * OAK_LANES, W + PPO_SCR_HC * OAK_LANES, logits);
    float logp;
    if (L->box) {
        logp = 0.0f;
        for (uint32_t d = 0u; d < L->A; d++) {
            float u1 = u01open(baselineWord(seed, RNG_STREAM_PPO_ACT, step, 2u * d));
            float u2 = u01(baselineWord(seed, RNG_STREAM_PPO_ACT, step, 2u * d + 1u));
            float z = sqrtf(-2.0f * oak_log(u1)) * oak_cos(2.0f * OAK_PI * u2);
            float ls = P[AT(L->logstd + d)];
            float sample = logits[d] + oak_exp(ls) * z;
            R[AT(PPO_ROLL_ACT + t * PPO_AMAX + d)] = sample;
            ctrl[d] = oak_clamp(sample, -1.0f, 1.0f);
            logp += -0.5f * z * z - ls - 0.9189385332f;
        }
    } else {
        float mx = logits[0];
        for (uint32_t a = 1u; a < L->A; a++) mx = fmaxf(mx, logits[a]);
        float denom = 0.0f;
        for (uint32_t a = 0u; a < L->A; a++) denom += oak_exp(logits[a] - mx);
        float u = u01(baselineWord(seed, RNG_STREAM_PPO_ACT, step, 0u));
        float cdf = 0.0f;
        uint32_t act = L->A - 1u;
        for (uint32_t a = 0u; a < L->A; a++) {
            cdf += oak_exp(logits[a] - mx) / denom;
            if (u < cdf) { act = a; break; }
        }
        logp = logits[act] - mx - oak_log(denom);
        R[AT(PPO_ROLL_ACT + t * PPO_AMAX)] = (float)act;
        ctrl[0] = (float)act;
        for (uint32_t d = 1u; d < L->A; d++) ctrl[d] = 0.0f;
    }
    R[AT(PPO_ROLL_LOGP + t)] = logp;
    R[AT(PPO_ROLL_VAL + t)] = value;
}
