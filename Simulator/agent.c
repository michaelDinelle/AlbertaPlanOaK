#include <float.h>
#include "agent.h"
static float keepFinite(float o, float n){ return isfinite(n) ? n : o; }
static float g_symbolValue[OAK_OBSERVATION_CAPACITY][OAK_SYMBOLS];
static uint32_t g_symbolCount[OAK_OBSERVATION_CAPACITY], g_symbol[OAK_OBSERVATION_CAPACITY];
typedef struct { float reward, effect; uint32_t seen; } Predict;
static Predict g_model[OAK_SYMBOLS][OAK_POLICY_MAX][OAK_OBSERVATION_CAPACITY][OAK_SYMBOLS];
static uint32_t g_mem;
static float g_actMean[OAK_POLICY_MAX][OAK_OBSERVATION_CAPACITY], g_chMean[OAK_OBSERVATION_CAPACITY], g_chSq[OAK_OBSERVATION_CAPACITY];
static float g_context = 1.0f, g_canonical[OAK_SYMBOLS], g_moveRate;
static uint32_t g_stale, g_staleRecord, g_gate, g_banks, g_ind;
static uint32_t encodeSymbol(uint32_t o, float x){
    uint32_t n = g_symbolCount[o];
    if (n == OAK_SYMBOLS) return OAK_SYMBOLS;
    for (uint32_t c = 0u; c < n; ++c) if (x == g_symbolValue[o][c]) return c;
    g_symbolValue[o][n] = x; g_symbolCount[o] = n + 1u; return n;
}
static uint32_t convertSymbol(uint32_t from, uint32_t c, uint32_t to){
    if (c >= g_symbolCount[from]) return OAK_SYMBOLS;
    float x = g_symbolValue[from][c];
    for (uint32_t k = 0u; k < g_symbolCount[to]; ++k) if (g_symbolValue[to][k] == x) return k;
    return OAK_SYMBOLS;
}
static uint32_t g_source[OAK_POLICY_MAX];
static float g_align[OAK_POLICY_MAX][OAK_OBSERVATION_CAPACITY];
static uint32_t g_prevSymbol[OAK_OBSERVATION_CAPACITY], g_alignAct;
static int g_haveAlign;
static uint32_t g_xlat[OAK_OBSERVATION_CAPACITY][OAK_OBSERVATION_CAPACITY][OAK_SYMBOLS+1u], g_dictSize;
static uint32_t g_prevAct;
static int   g_havePrevAct;
void constructFeatures(Agent a){
    for (uint32_t o = 0u; o < a.observationCount; ++o) g_symbol[o] = encodeSymbol(o, a.memory->observation[o]);
    { uint32_t n = 0u; for (uint32_t o = 0u; o < a.observationCount; ++o) n += g_symbolCount[o];
      if (n != g_dictSize) for (g_dictSize = n, n = 0u; n < a.observationCount; ++n)
          for (uint32_t o = 0u; o < a.observationCount; ++o)
              for (uint32_t c = 0u; c <= OAK_SYMBOLS; ++c) g_xlat[n][o][c] = convertSymbol(n, c, o); }
}
uint32_t oakSampleDiscrete(Agent a, uint32_t discreteActionCount, const uint32_t *words){
    uint32_t sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint32_t d=0u; d<discreteActionCount; ++d){
        float u = ((float)(words[d] >> 8) + 1.0f) * 0x1p-24f;
        if (u > 0x1.fffffep-1f) u = 0x1.fffffep-1f;
        float g = a.memory->policyMeanRaw[d] - logf(-logf(u));
        if (g > bestGumbelScore){ bestGumbelScore = g; sampledAction = d; }
    }
    return sampledAction;
}
void oakActionModelLearn(Agent a, float reward){
    g_stale++;
    if (reward != 0.0f){
        float bestGate = 0.0f;
        for (uint32_t o = 0u; o < a.observationCount; ++o){
            if (g_symbolCount[o] == OAK_SYMBOLS) continue;
            float spread = 0.0f;
            for (uint32_t d = 0u; d < a.envActionCount; ++d)
                for (uint32_t c = 0u; c < g_symbolCount[o]; ++c)
                    for (uint32_t m = 0u; m < g_banks; ++m) spread += g_model[m][d][o][c].reward*g_model[m][d][o][c].reward;
            if (spread > bestGate){ bestGate = spread; g_gate = o; }
        }
        if (g_symbol[g_gate] < OAK_SYMBOLS){
            g_staleRecord = g_stale > g_staleRecord ? g_stale : g_staleRecord; g_stale = 0u;
            uint32_t s = g_symbol[g_gate];
            float canonical = g_canonical[s];
            if (canonical != 0.0f) g_context = keepFinite(g_context, reward / canonical);
            else g_canonical[s] = keepFinite(0.0f, reward / g_context);
        }
    }
    a.memory->reward = keepFinite(0.0f, reward / g_context);
    g_moveRate += OAK_TRACKING_RATE*(reward - g_moveRate);
    if (g_haveAlign && g_symbol[g_gate] < OAK_SYMBOLS){
        uint32_t so = g_symbol[g_gate], arg = g_gate; float *row = g_align[g_alignAct], best = -1.0f;
        for (uint32_t p = 0u; p < a.observationCount; ++p){
            row[p] = row[p]*(1.0f - OAK_TRACKING_RATE) + (so == g_prevSymbol[p] ? OAK_TRACKING_RATE : 0.0f);
            if (row[p] > best){ best = row[p]; arg = p; }
        }
        g_source[g_alignAct] = arg;
    }
    uint32_t ind = 0u; float bestInd = -1.0f;
    for (uint32_t o = 0u; o < a.observationCount; ++o){
        float x = a.memory->observation[o], m = 0.0f, between = 0.0f;
        if (g_haveAlign) g_actMean[g_alignAct][o] += OAK_TRACKING_RATE*(x - g_actMean[g_alignAct][o]);
        g_chMean[o] += OAK_TRACKING_RATE*(x - g_chMean[o]);
        g_chSq[o]   += OAK_TRACKING_RATE*(x*x - g_chSq[o]);
        for (uint32_t d = 0u; d < a.envActionCount; ++d){ float v = g_actMean[d][o]; m += v; between += v*v; }
        float total = g_chSq[o] - g_chMean[o]*g_chMean[o];
        float score = (between - m*m/(float)a.envActionCount)/(total > 0.0f ? total : 1.0f);
        if (score > bestInd && g_symbol[o] < OAK_SYMBOLS){ bestInd = score; ind = o; }
    }
    if (g_haveAlign)
        for (uint32_t o = 0u; o < a.observationCount; ++o) if (g_prevSymbol[o] < OAK_SYMBOLS){
            Predict *e = &g_model[g_mem][g_alignAct][o][g_prevSymbol[o]];
            e->effect = keepFinite(e->effect, e->effect + (e->seen < OAK_FEATURE_TIMESCALE ? 1.0f/(float)++e->seen : OAK_TRACKING_RATE)*(a.memory->observation[ind] - e->effect));
        }
    g_mem = (g_prevSymbol[ind] < OAK_SYMBOLS) ? g_prevSymbol[ind] : 0u; g_banks = g_symbolCount[ind]; g_ind = ind;
    memcpy(g_prevSymbol, g_symbol, sizeof g_prevSymbol);
    g_alignAct = g_prevAct; g_haveAlign = g_havePrevAct;
    if (!g_havePrevAct) return;
    float pred = 0.0f, strongest = -1.0f; uint32_t credit = OAK_OBSERVATION_CAPACITY;
    for (uint32_t o = 0u; o < a.observationCount; ++o) if (g_symbol[o] < OAK_SYMBOLS){
        float w = g_model[g_mem][g_prevAct][o][g_symbol[o]].reward; pred += w;
        if (fabsf(w) > strongest){ strongest = fabsf(w); credit = o; } }
    if (credit == OAK_OBSERVATION_CAPACITY) return;
    Predict *e = &g_model[g_mem][g_prevAct][credit][g_symbol[credit]]; e->reward = keepFinite(e->reward, e->reward + OAK_TRACKING_RATE*(a.memory->reward - pred));
}
void oakActionModelRemember(Agent a, uint32_t action){ g_prevAct=action; g_havePrevAct=1; }
void oakActionModelBiasScaleFree(Agent a){
    const uint32_t actionCount = a.envActionCount;
    float q[OAK_POLICY_MAX]; float qm=0.0f; uint32_t mover = 0u;
    for (uint32_t d=1u; d<actionCount; ++d) if (g_actMean[d][g_ind] > g_actMean[mover][g_ind]) mover = d;
    if (g_stale > g_staleRecord){ for (uint32_t d=0;d<OAK_POLICY_MAX;++d) a.memory->policyMeanRaw[d] = 0.0f; return; }
    float trust = 1.0f/(1.0f + (float)g_stale*OAK_TRACKING_RATE);
    for (uint32_t d=0;d<actionCount;++d){
        float effect = FLT_MAX;
        uint32_t p = g_source[d], t = g_xlat[p][g_gate][g_symbol[p]];
        float ahead = (t < OAK_SYMBOLS && d != mover) ? g_canonical[t] : 0.0f;
        float reward = g_symbol[g_gate] < OAK_SYMBOLS ? g_canonical[g_symbol[g_gate]] : 0.0f;
        for (uint32_t o = 0u; o < a.observationCount; ++o)
            if (g_symbol[o] < OAK_SYMBOLS){ float ef = g_model[g_mem][d][o][g_symbol[o]].effect;
                effect = effect < ef ? effect : ef; }
        float changes = (float)(d == mover ? effect > 0.0f : g_source[d] != g_gate);
        q[d] = keepFinite(0.0f, effect*(g_context*reward + g_moveRate) + fmaxf(g_context*ahead, 0.0f) + changes*g_moveRate);
        qm += q[d]/(float)actionCount;
    }
    float qs=0.0f;
    for (uint32_t d=0;d<actionCount;++d){ float dq=q[d]-qm; qs += dq*dq; }
    float sd = sqrtf(qs);
    float gain = logf((float)(actionCount-1u)*(float)OAK_FEATURE_TIMESCALE) / (sd > 0.0f ? sd : 1.0f);
    for (uint32_t d=actionCount;d<OAK_POLICY_MAX;++d) a.memory->policyMeanRaw[d] = 0.0f;
    for (uint32_t d=0;d<actionCount;++d)
        a.memory->policyMeanRaw[d] = keepFinite(0.0f, gain*(q[d]-qm)*trust);
}
