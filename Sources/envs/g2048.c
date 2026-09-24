#include "../env.h"

enum { G2048_ROWS = 65536, G2048_FORESIGHT = 3 };

static const uint8_t G2048_ORIENT[4][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 },
    { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 },
    { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 }
};

static const uint8_t G2048_RESTORE[4][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 },
    { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 },
    { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 }
};

static const uint8_t G2048_SQUEEZE[64] = {
    128, 128, 128, 128, 0, 128, 128, 128, 1, 128, 128, 128, 0, 1, 128, 128,
    2, 128, 128, 128, 0, 2, 128, 128, 1, 2, 128, 128, 0, 1, 2, 128,
    3, 128, 128, 128, 0, 3, 128, 128, 1, 3, 128, 128, 0, 1, 3, 128,
    2, 3, 128, 128, 0, 2, 3, 128, 1, 2, 3, 128, 0, 1, 2, 3
};

static inline __attribute__((always_inline)) uint8x16_t g2048_squeeze(uint8x16_t line, uint8x16x4_t table) {
    uint8x16_t live = vandq_u8(vtstq_u8(line, line), (uint8x16_t){ 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8 });
    uint8x16_t mask = vpaddq_u8(live, live);
    mask = vpaddq_u8(mask, mask);
    uint8x16_t position = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
    uint8x16_t index = vsliq_n_u8(position, vqtbl1q_u8(mask, (uint8x16_t){ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 }), 2);
    return vqtbl1q_u8(line, vorrq_u8(vqtbl4q_u8(table, index), (uint8x16_t){ 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12 }));
}

static inline __attribute__((always_inline)) uint32_t g2048_score(uint8x16_t grown, uint8x16_t first) {
    uint8x16_t index = vornq_u8(grown, first);
    uint32_t low = vaddlvq_u8(vqtbl1q_u8((uint8x16_t){ 1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0 }, index));
    uint32_t middle = vaddlvq_u8(vqtbl1q_u8((uint8x16_t){ 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, 128 }, index));
    uint32_t high = vaddlvq_u8(vqtbl1q_u8((uint8x16_t){ 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, vsubq_u8(index, vdupq_n_u8(16))));
    return low + (middle << 8) + (high << 16);
}

static inline __attribute__((always_inline)) uint8x16_t g2048_move(uint8x16_t board, uint32_t action, uint32_t *score, uint8x16x4_t table) {
    uint8x16_t line = g2048_squeeze(vqtbl1q_u8(board, vld1q_u8(G2048_ORIENT[action])), table);
    uint32x4_t pair = vreinterpretq_u32_u8(vandq_u8(vceqq_u8(line, vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(line), 8))), vtstq_u8(line, line)));
    uint32x4_t first = vbicq_u32(pair, vshlq_n_u32(vbicq_u32(pair, vshlq_n_u32(pair, 8)), 8));
    uint8x16_t grown = vsubq_u8(line, vreinterpretq_u8_u32(first));
    *score = g2048_score(grown, vreinterpretq_u8_u32(first));
    uint8x16_t kept = vbicq_u8(grown, vreinterpretq_u8_u32(vshlq_n_u32(first, 8)));
    return vqtbl1q_u8(g2048_squeeze(kept, table), vld1q_u8(G2048_RESTORE[action]));
}

static inline __attribute__((always_inline)) uint8x16_t g2048_spawn(uint8x16_t board, uint32_t draw, uint32_t cell, uint32_t live) {
    uint8x16_t hole = vceqzq_u8(board), rank = vandq_u8(hole, vdupq_n_u8(1)), zero = vdupq_n_u8(0);
    uint32_t open = vaddvq_u8(rank);
    rank = vaddq_u8(rank, vextq_u8(zero, rank, 15));
    rank = vaddq_u8(rank, vextq_u8(zero, rank, 14));
    rank = vaddq_u8(rank, vextq_u8(zero, rank, 12));
    rank = vaddq_u8(rank, vextq_u8(zero, rank, 8));
    uint32_t pick = live ? (uint32_t)(((uint64_t)cell * open) >> 32) + 1u : 0u;
    uint8x16_t target = vandq_u8(vceqq_u8(rank, vdupq_n_u8((uint8_t)pick)), hole);
    return vorrq_u8(board, vandq_u8(target, vdupq_n_u8(draw < u01Bound(0.9f) ? 1u : 2u)));
}

static inline __attribute__((always_inline)) uint32_t g2048_alive(uint8x16_t board) {
    uint8x16_t across = vceqq_u8(board, vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(board), 8)));
    uint8x16_t down = vceqq_u8(board, vextq_u8(board, vdupq_n_u8(0), 4));
    return vmaxvq_u8(vorrq_u8(vorrq_u8(across, down), vceqzq_u8(board))) != 0u;
}

static void g2048_obs(uint8x16_t board, float *obs) {
    uint8x16_t rows = vqtbl1q_u8(board, vld1q_u8(G2048_ORIENT[2]));
    uint16x8_t low = vmovl_u8(vget_low_u8(rows)), high = vmovl_high_u8(rows);
    float32x4_t eleven = vdupq_n_f32(11.0f);
    vst1q_f32(obs, vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))), eleven));
    vst1q_f32(obs + 4, vdivq_f32(vcvtq_f32_u32(vmovl_high_u16(low)), eleven));
    vst1q_f32(obs + 8, vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))), eleven));
    vst1q_f32(obs + 12, vdivq_f32(vcvtq_f32_u32(vmovl_high_u16(high)), eleven));
    obs[16] = 1.0f;
}

static uint32_t g2048_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 4.0f ? (uint32_t)c : 0u;
}

static uint8x16_t g2048_start(uint32_t seed, uint32_t step) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u);
    return g2048_spawn(g2048_spawn(vdupq_n_u8(0), r.x, r.y, 1u), r.z, r.w, 1u);
}

static float g2048_heuristic(uint32_t row) {
    float rank[4], power[4], sum = 0.0f, left = 0.0f, right = 0.0f;
    uint32_t empty = 0u, merges = 0u, previous = 0u, counter = 0u;
    for (uint32_t i = 0u; i < 4u; i++) {
        uint32_t tile = (row >> (4u * i)) & 15u;
        rank[i] = (float)tile;
        power[i] = rank[i] * rank[i] * rank[i] * rank[i];
        sum += rank[i] * rank[i] * rank[i] * sqrtf(rank[i]);
        if (tile == 0u) { empty++; continue; }
        if (previous == tile) counter++;
        else if (counter > 0u) { merges += 1u + counter; counter = 0u; }
        previous = tile;
    }
    if (counter > 0u) merges += 1u + counter;
    for (uint32_t i = 1u; i < 4u; i++) {
        if (rank[i - 1u] > rank[i]) left += power[i - 1u] - power[i];
        else right += power[i] - power[i - 1u];
    }
    return 200000.0f + 270.0f * (float)empty + 700.0f * (float)merges - 47.0f * fminf(left, right) - 11.0f * sum;
}

uint64_t g2048_shared(float *table) {
    if (table) {
        table[G2048_ROWS] = 0.0f;
        for (uint32_t row = 0u; row < G2048_ROWS; row++) {
            table[row] = g2048_heuristic(row);
            table[G2048_ROWS] = fminf(table[G2048_ROWS], 8.0f * table[row]);
        }
    }
    return G2048_ROWS + 1u;
}

void g2048_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)extra; (void)steps;
    vst1q_u8((uint8_t *)words, g2048_start(seed, 0u));
}

static __attribute__((noinline)) float g2048_over(uint32_t seed, uint32_t step, uint32_t *words, float *obs, uint32_t *done, uint32_t score) {
    uint8x16_t board = g2048_start(seed, step);
    vst1q_u8((uint8_t *)words, board);
    g2048_obs(board, obs);
    *done = 1u;
    return (float)score;
}

float g2048_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)instance; (void)extra; (void)shared;
    uint8x16_t board = vld1q_u8((const uint8_t *)words);
    uint32_t score;
    uint8x16_t next = g2048_move(board, g2048_action(ctrl), &score, vld1q_u8_x4(G2048_SQUEEZE));
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    uint32_t moved = vmaxvq_u8(veorq_u8(next, board)) != 0u;
    next = g2048_spawn(next, r.x, r.y, moved);
    if (moved & !g2048_alive(next)) return g2048_over(seed, step, words, obs, done, score);
    *done = 0u;
    vst1q_u8((uint8_t *)words, next);
    g2048_obs(next, obs);
    return (float)score;
}

void g2048_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    g2048_obs(vld1q_u8((const uint8_t *)words), obs);
}

static uint8x16_t g2048_place(uint8x16_t board, uint32_t cell, uint8_t tile) {
    uint8x16_t spot = vceqq_u8((uint8x16_t){ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, vdupq_n_u8((uint8_t)cell));
    return vorrq_u8(board, vandq_u8(spot, vdupq_n_u8(tile)));
}

static float g2048_best(uint8x16_t board, uint32_t depth, const float *shared, uint8x16x4_t table);

static float g2048_expect(uint8x16_t board, uint32_t depth, const float *shared, uint8x16x4_t table) {
    uint8_t key[16];
    vst1q_u8(key, vminq_u8(board, vdupq_n_u8(15)));
    uint32_t row[4], column[4];
    float base = 0.0f, total = 0.0f, lost = shared[G2048_ROWS];
    for (uint32_t i = 0u; i < 4u; i++) {
        column[i] = key[4u * i] | (uint32_t)key[4u * i + 1u] << 4 | (uint32_t)key[4u * i + 2u] << 8 | (uint32_t)key[4u * i + 3u] << 12;
        row[i] = key[i] | (uint32_t)key[4u + i] << 4 | (uint32_t)key[8u + i] << 8 | (uint32_t)key[12u + i] << 12;
        base += shared[row[i]] + shared[column[i]];
    }
    uint8x16_t hole = vceqzq_u8(board);
    uint32_t open = vaddvq_u8(vandq_u8(hole, vdupq_n_u8(1)));
    for (uint64_t holes = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(hole), 4)), 0) & 0x1111111111111111ull; holes; holes &= holes - 1u) {
        uint32_t i = (uint32_t)__builtin_ctzll(holes) >> 2, x = i >> 2, y = i & 3u;
        float two, four;
        if (depth) {
            two = g2048_best(g2048_place(board, i, 1u), depth - 1u, shared, table);
            four = g2048_best(g2048_place(board, i, 2u), depth - 1u, shared, table);
        } else {
            float rest = base - shared[row[y]] - shared[column[x]];
            two = rest + shared[row[y] | 1u << (4u * x)] + shared[column[x] | 1u << (4u * y)];
            four = rest + shared[row[y] | 2u << (4u * x)] + shared[column[x] | 2u << (4u * y)];
            if (open == 1u) {
                two = g2048_alive(g2048_place(board, i, 1u)) ? two : lost;
                four = g2048_alive(g2048_place(board, i, 2u)) ? four : lost;
            }
        }
        total += 0.9f * two + 0.1f * four;
    }
    return total / (float)open;
}

static float g2048_best(uint8x16_t board, uint32_t depth, const float *shared, uint8x16x4_t table) {
    float best = shared[G2048_ROWS];
    for (uint32_t action = 0u; action < 4u; action++) {
        uint32_t score;
        uint8x16_t next = g2048_move(board, action, &score, table);
        if (vmaxvq_u8(veorq_u8(next, board)) != 0u) best = fmaxf(best, g2048_expect(next, depth, shared, table));
    }
    return best;
}

void g2048_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    uint8x16x4_t table = vld1q_u8_x4(G2048_SQUEEZE);
    uint8x16_t board = vld1q_u8((const uint8_t *)words);
    float best = -INFINITY;
    uint32_t choice = 0u;
    for (uint32_t action = 0u; action < 4u; action++) {
        uint32_t score;
        uint8x16_t next = g2048_move(board, action, &score, table);
        if (vmaxvq_u8(veorq_u8(next, board)) == 0u) continue;
        float value = g2048_expect(next, vaddvq_u8(vandq_u8(vceqzq_u8(next), vdupq_n_u8(1))) <= G2048_FORESIGHT, shared, table);
        if (value > best) { best = value; choice = action; }
    }
    ctrl[0] = (float)choice;
}
