#pragma METAL fp math_mode(safe)

constant uint JBW_EMPTY = 0u;
constant uint JBW_BANANA = 1u;
constant uint JBW_ONION = 2u;
constant uint JBW_JELLY = 3u;
constant uint JBW_WALL = 4u;
constant uint JBW_TREE = 5u;
constant uint JBW_TRUFFLE = 6u;
constant uint JBW_RESPONSE_RADIUS = 33u;
constant uint JBW_RESPONSE_CELLS = 561u;
constant uint JBW_DESTRUCTION_LIFETIME = 2000u;

inline int jbw_floorDiv(int x, int d) {
    int q = x / d;
    return q - ((x < 0 && x % d != 0) ? 1 : 0);
}

inline uint4 jbw_field(uint seed, int x, int y, uint stream) {
    return philox4x32(uint2(seed, ENV_RNG_STREAM),
                     uint4(as_type<uint>(x), as_type<uint>(y), stream, 0u));
}

inline int2 jbw_candidate(uint seed, int bx, int by, uint item) {
    uint4 r = jbw_field(seed, bx, by, 16u);
    return int2(bx * 16 + int(r.x & 15u), by * 16 + int(r.y & 15u));
}

inline bool jbw_wallAt(uint seed, int x, int y) {
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    uint4 r = jbw_field(seed, bx, by, 80u);
    if ((r.z & 7u) != 0u) return false;
    return (r.w & 1u) == 0u ? y == by * 16 + int(r.y & 15u)
                            : x == bx * 16 + int(r.x & 15u);
}

inline uint jbw_itemAt(uint seed, int x, int y) {
    if (jbw_wallAt(seed, x, y)) return JBW_WALL;
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    int2 p = jbw_candidate(seed, bx, by, 0u);
    if (!all(p == int2(x, y))) return JBW_EMPTY;
    uint4 r = jbw_field(seed, bx, by, 16u);
    uint4 ecology = jbw_field(seed, jbw_floorDiv(x, 64), jbw_floorDiv(y, 64), 64u);
    if ((ecology.y & 3u) == 0u) return (r.w & 31u) == 0u ? JBW_TRUFFLE : JBW_TREE;
    if ((ecology.x & 3u) == 0u) return (r.z & 1u) == 0u ? JBW_BANANA : JBW_JELLY;
    return (r.z & 3u) == 0u ? JBW_ONION : JBW_EMPTY;
}

inline ulong jbw_key(int x, int y) {
    return (ulong(as_type<uint>(x)) << 32u) | ulong(as_type<uint>(y));
}

inline uint jbw_slot(ulong key, uint mask) {
    key ^= key >> 30u;
    key *= 0xbf58476d1ce4e5b9ul;
    key ^= key >> 27u;
    key *= 0x94d049bb133111ebul;
    key ^= key >> 31u;
    return uint(key) & mask;
}

inline uint jbw_destroyedAt(device const uint *history, uint mask, int x, int y) {
    ulong key = jbw_key(x, y);
    uint slot = jbw_slot(key, mask);
    for (;;) {
        device const uint *entry = history + 3u * slot;
        if (entry[2] == 0u) return 0u;
        if (entry[0] == uint(key >> 32u) && entry[1] == uint(key)) return entry[2];
        slot = (slot + 1u) & mask;
    }
}

inline void jbw_destroy(device uint *history, uint mask, int x, int y, uint step) {
    ulong key = jbw_key(x, y);
    uint slot = jbw_slot(key, mask);
    for (;;) {
        device uint *entry = history + 3u * slot;
        if (entry[2] == 0u) {
            entry[0] = uint(key >> 32u);
            entry[1] = uint(key);
            entry[2] = step;
            return;
        }
        if (entry[0] == uint(key >> 32u) && entry[1] == uint(key)) return;
        slot = (slot + 1u) & mask;
    }
}

inline float jbw_energy(uint item) {
    switch (item) {
        case JBW_BANANA: return 2.04f;
        case JBW_ONION: return 1.68f;
        case JBW_JELLY: return 1.29f;
        case JBW_WALL: return 0.87f;
        case JBW_TREE: return 0.53f;
        case JBW_TRUFFLE: return 0.79f;
        default: return 0.0f;
    }
}

inline float3 jbw_scent(uint item) {
    switch (item) {
        case JBW_BANANA: return float3(1.92f, 1.76f, 0.40f);
        case JBW_ONION: return float3(0.68f, 0.01f, 0.99f);
        case JBW_JELLY: return float3(1.64f, 0.54f, 0.40f);
        case JBW_TREE: return float3(0.00f, 0.47f, 0.06f);
        case JBW_TRUFFLE: return float3(8.40f, 4.80f, 2.60f);
        default: return float3(0.0f);
    }
}

inline float jbw_response(device const float *table, uint time, int dx, int dy) {
    uint x = uint(abs(dx)), y = uint(abs(dy));
    if (x < y) { uint t = x; x = y; y = t; }
    return table[time * JBW_RESPONSE_CELLS + x * (x + 1u) / 2u + y];
}

inline float3 jbw_scentAt(uint seed, int x, int y, uint step,
                          device const uint *history, uint historyMask,
                          device const float *response) {
    float3 scent = float3(0.0f);
    int bx0 = jbw_floorDiv(x - 32, 16), bx1 = jbw_floorDiv(x + 32, 16);
    int by0 = jbw_floorDiv(y - 32, 16), by1 = jbw_floorDiv(y + 32, 16);
    for (int by = by0; by <= by1; by++) {
        for (int bx = bx0; bx <= bx1; bx++) {
            int2 p = jbw_candidate(seed, bx, by, 0u);
            int dx = p.x - x, dy = p.y - y;
            uint item = jbw_itemAt(seed, p.x, p.y);
            if (item != JBW_EMPTY && item != JBW_WALL
                && abs(dx) < int(JBW_RESPONSE_RADIUS) && abs(dy) < int(JBW_RESPONSE_RADIUS)) {
                uint deletion = jbw_destroyedAt(history, historyMask, p.x, p.y);
                float value = jbw_response(response, JBW_DESTRUCTION_LIFETIME - 1u, dx, dy);
                if (deletion != 0u) {
                    uint age = step - deletion;
                    if (age >= JBW_DESTRUCTION_LIFETIME) value = 0.0f;
                    else value -= jbw_response(response, age, dx, dy);
                }
                scent += jbw_scent(item) * value;
            }
        }
    }
    return scent;
}

inline void jbw_masks(uint seed, int x, int y, device const uint *history, uint historyMask,
                      device uint &lo, device uint &hi) {
    lo = 0u; hi = 0u;
    uint bit = 0u;
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx == 0 && dy == 0) continue;
            uint item = jbw_itemAt(seed, x + dx, y + dy);
            bool collectible = item == JBW_BANANA || item == JBW_ONION
                || item == JBW_JELLY || item == JBW_TRUFFLE;
            if (collectible && jbw_destroyedAt(history, historyMask, x + dx, y + dy) != 0u) {
                if (bit < 24u) lo |= 1u << bit;
                else hi |= 1u << (bit - 24u);
            }
            bit++;
        }
    }
}

inline void jbw_observe(uint seed, device uint *words, device const uint *history,
                        uint historyMask, device const float *response, uint step,
                        device float *obs) {
    int x = as_type<int>(words[0]), y = as_type<int>(words[1]);
    uint heading = words[2] & 3u;
    int2 directions[8] = {
        int2(0, -1), int2(1, -1), int2(1, 0), int2(1, 1),
        int2(0, 1), int2(-1, 1), int2(-1, 0), int2(-1, -1)
    };
    for (uint i = 0u; i < 8u; i++) {
        int2 d = directions[(2u * heading + i) & 7u];
        uint item = jbw_itemAt(seed, x + d.x, y + d.y);
        bool absent = item != JBW_WALL && item != JBW_TREE
            && jbw_destroyedAt(history, historyMask, x + d.x, y + d.y) != 0u;
        obs[i] = absent ? 0.0f : jbw_energy(item);
    }
    float3 scent = jbw_scentAt(seed, x, y, step, history, historyMask, response);
    obs[8] = scent.x; obs[9] = scent.y; obs[10] = scent.z;
    obs[11] = float(words[3]);
    for (uint i = 0u; i < 4u; i++) obs[12u + i] = float(words[4u + i]);
    jbw_masks(seed, x, y, history, historyMask, words[8], words[9]);
}

inline float jbw_step(uint seed, uint action, device uint *words, device uint *history,
                      uint historyMask, device const float *response, uint step,
                      device float *obs) {
    words[3] = 0u;
    if (action == 1u) words[2] = (words[2] + 3u) & 3u;
    else if (action == 2u) words[2] = (words[2] + 1u) & 3u;
    else {
        int2 forward[4] = { int2(0, -1), int2(1, 0), int2(0, 1), int2(-1, 0) };
        int2 p = int2(as_type<int>(words[0]), as_type<int>(words[1])) + forward[words[2] & 3u];
        uint item = jbw_itemAt(seed, p.x, p.y);
        if (item != JBW_WALL) {
            words[0] = as_type<uint>(p.x); words[1] = as_type<uint>(p.y); words[3] = 1u;
            bool collectible = item == JBW_BANANA || item == JBW_ONION
                || item == JBW_JELLY || item == JBW_TRUFFLE;
            if (collectible && jbw_destroyedAt(history, historyMask, p.x, p.y) == 0u) {
                uint inventory = item == JBW_BANANA ? 4u : item == JBW_ONION ? 5u
                    : item == JBW_JELLY ? 6u : 7u;
                words[inventory]++;
                jbw_destroy(history, historyMask, p.x, p.y, step);
            }
        }
    }
    float reward = 0.0f;
    if (action == 0u && words[3] != 0u) {
        int x = as_type<int>(words[0]), y = as_type<int>(words[1]);
        uint item = jbw_itemAt(seed, x, y);
        if (jbw_destroyedAt(history, historyMask, x, y) == step) {
            float sign = (step / 100000u & 1u) == 0u ? 1.0f : -1.0f;
            if (item == JBW_JELLY) reward = sign;
            else if (item == JBW_ONION) reward = -sign;
        }
    }
    jbw_observe(seed, words, history, historyMask, response, step, obs);
    return reward;
}
