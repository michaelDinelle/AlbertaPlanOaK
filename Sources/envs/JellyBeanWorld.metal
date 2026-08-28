#pragma METAL fp math_mode(safe)

constant uint JBW_EMPTY = 0u;
constant uint JBW_BANANA = 1u;
constant uint JBW_ONION = 2u;
constant uint JBW_JELLY = 3u;
constant uint JBW_WALL = 4u;
constant uint JBW_TREE = 5u;
constant uint JBW_TRUFFLE = 6u;
constant uint JBW_RESPONSE_RADIUS = 17u;
constant uint JBW_RESPONSE_CELLS = 153u;
constant uint JBW_DESTRUCTION_LIFETIME = 2000u;
constant uint JBW_RESPONSE_WORDS = JBW_RESPONSE_CELLS * JBW_DESTRUCTION_LIFETIME;
constant uint JBW_BLOCK_ITEMS = 4u;

inline int jbw_floorDiv(int x, int d) {
    int q = x / d;
    return q - ((x < 0 && x % d != 0) ? 1 : 0);
}

inline uint4 jbw_field(uint seed, int x, int y, uint stream) {
    return philox4x32(uint2(seed, ENV_RNG_STREAM),
                     uint4(as_type<uint>(x), as_type<uint>(y), stream, 0u));
}

inline int2 jbw_candidate(uint4 block, int bx, int by, uint slot) {
    uint bits = (block.x >> (8u * slot)) & 255u;
    return int2(bx * 16 + int(bits & 15u), by * 16 + int((bits >> 4u) & 15u));
}

inline uint jbw_typeOf(uint4 block, uint4 ecology, uint slot) {
    uint roll = (block.y >> (8u * slot)) & 255u;
    if ((ecology.y & 3u) == 0u) return (roll & 31u) == 0u ? JBW_TRUFFLE : JBW_TREE;
    uint pick = roll % 3u;
    return pick == 0u ? JBW_BANANA : pick == 1u ? JBW_JELLY : JBW_ONION;
}

inline bool jbw_wallHit(uint4 w, int bx, int by, int x, int y) {
    if ((w.z & 7u) != 0u) return false;
    return (w.w & 1u) == 0u ? y == by * 16 + int(w.y & 15u)
                            : x == bx * 16 + int(w.x & 15u);
}

inline bool jbw_wallAt(uint seed, int x, int y) {
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    return jbw_wallHit(jbw_field(seed, bx, by, 80u), bx, by, x, y);
}

inline uint jbw_itemAt(uint seed, int x, int y) {
    if (jbw_wallAt(seed, x, y)) return JBW_WALL;
    int bx = jbw_floorDiv(x, 16), by = jbw_floorDiv(y, 16);
    uint4 block = jbw_field(seed, bx, by, 16u);
    for (uint slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
        if (!all(jbw_candidate(block, bx, by, slot) == int2(x, y))) continue;
        uint4 ecology = jbw_field(seed, jbw_floorDiv(x, 64), jbw_floorDiv(y, 64), 64u);
        return jbw_typeOf(block, ecology, slot);
    }
    return JBW_EMPTY;
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
            entry[2] = step + 1u;
            return;
        }
        if (entry[0] == uint(key >> 32u) && entry[1] == uint(key)) return;
        slot = (slot + 1u) & mask;
    }
}

inline bool jbw_blocks(uint item) {
    return item == JBW_WALL || item == JBW_TREE;
}

inline bool jbw_collectible(uint item) {
    return item == JBW_BANANA || item == JBW_ONION
        || item == JBW_JELLY || item == JBW_TRUFFLE;
}

inline float jbw_energy(uint item) {
    switch (item) {
        case JBW_BANANA: return 2.04f;
        case JBW_ONION: return 1.68f;
        case JBW_JELLY: return 1.29f;
        case JBW_WALL: return 1.34f;
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

inline void jbw_buildResponse(device float *response) {
    for (uint i = 0u; i < JBW_RESPONSE_WORDS; i++) response[i] = 0.0f;
    response[0] = 1.0f;
    for (uint time = 1u; time < JBW_DESTRUCTION_LIFETIME; time++) {
        uint previous = (time - 1u) * JBW_RESPONSE_CELLS;
        uint current = time * JBW_RESPONSE_CELLS;
        for (uint i = 0u; i < JBW_RESPONSE_CELLS; i++) {
            response[current + i] = 0.4f * response[previous + i];
        }
        response[current] += 1.0f;
        response[current + JBW_RESPONSE_CELLS - 1u] +=
            0.28f * response[previous + JBW_RESPONSE_CELLS - 2u];
        for (int y = 0; y < int(JBW_RESPONSE_RADIUS) - 1; y++) {
            response[current + JBW_RESPONSE_CELLS - JBW_RESPONSE_RADIUS + uint(y)] += 0.14f * (
                jbw_response(response, time - 1u, int(JBW_RESPONSE_RADIUS) - 2, y)
                + jbw_response(response, time - 1u, int(JBW_RESPONSE_RADIUS) - 1, y + 1)
                + jbw_response(response, time - 1u, int(JBW_RESPONSE_RADIUS) - 1, y - 1));
        }
        for (int x = 0; x < int(JBW_RESPONSE_RADIUS) - 1; x++) {
            for (int y = 0; y <= x; y++) {
                uint index = uint(x * (x + 1) / 2 + y);
                response[current + index] += 0.14f * (
                    jbw_response(response, time - 1u, x + 1, y)
                    + jbw_response(response, time - 1u, x - 1, y)
                    + jbw_response(response, time - 1u, x, y + 1)
                    + jbw_response(response, time - 1u, x, y - 1));
            }
        }
    }
}

inline float3 jbw_scentAt(uint seed, int x, int y, uint step,
                          device const uint *history, uint historyMask,
                          device const float *response, uint slane, uint width) {
    float3 scent = float3(0.0f);
    int reach = int(JBW_RESPONSE_RADIUS) - 1;
    int bx0 = jbw_floorDiv(x - reach, 16), bx1 = jbw_floorDiv(x + reach, 16);
    int by0 = jbw_floorDiv(y - reach, 16), by1 = jbw_floorDiv(y + reach, 16);
    uint span = uint(bx1 - bx0 + 1);
    uint blocks = span * uint(by1 - by0 + 1);
    for (uint base = 0u; base < blocks; base += width) {
        uint k = base + slane;
        float3 share = float3(0.0f);
        uint carries = 0u;
        if (k < blocks) {
            int by = by0 + int(k / span), bx = bx0 + int(k % span);
            uint4 block = jbw_field(seed, bx, by, 16u);
            uint4 wall = jbw_field(seed, bx, by, 80u);
            uint4 ecology = jbw_field(seed, jbw_floorDiv(bx * 16, 64), jbw_floorDiv(by * 16, 64), 64u);
            for (uint slot = 0u; slot < JBW_BLOCK_ITEMS; slot++) {
                int2 p = jbw_candidate(block, bx, by, slot);
                int dx = p.x - x, dy = p.y - y;
                if (abs(dx) >= int(JBW_RESPONSE_RADIUS) || abs(dy) >= int(JBW_RESPONSE_RADIUS)) continue;
                if (jbw_wallHit(wall, bx, by, p.x, p.y)) continue;
                uint item = jbw_typeOf(block, ecology, slot);
                uint deletion = jbw_collectible(item)
                    ? jbw_destroyedAt(history, historyMask, p.x, p.y) : 0u;
                float value = jbw_response(response, JBW_DESTRUCTION_LIFETIME - 1u, dx, dy);
                if (deletion != 0u) {
                    uint age = step - (deletion - 1u);
                    if (age >= JBW_DESTRUCTION_LIFETIME) value = 0.0f;
                    else value -= jbw_response(response, age, dx, dy);
                }
                share += jbw_scent(item) * value;
                carries = 1u;
            }
        }
        for (uint j = 0u; j < width && base + j < blocks; j++) {
            ushort src = ushort(j);
            bool add = width > 1u ? simd_shuffle(carries, src) != 0u : carries != 0u;
            float3 term = width > 1u
                ? float3(simd_shuffle(share.x, src), simd_shuffle(share.y, src),
                         simd_shuffle(share.z, src))
                : share;
            if (add) scent += term;
        }
    }
    return scent;
}

inline void jbw_observe(uint seed, device uint *words, device const uint *history,
                        uint historyMask, device const float *response, uint step,
                        uint slane, uint width, device float *obs) {
    int x = as_type<int>(words[0]), y = as_type<int>(words[1]);
    uint heading = words[2] & 3u;
    int2 directions[8] = {
        int2(0, -1), int2(1, -1), int2(1, 0), int2(1, 1),
        int2(0, 1), int2(-1, 1), int2(-1, 0), int2(-1, -1)
    };
    for (uint i = slane; i < 8u; i += width) {
        int2 d = directions[(2u * heading + i) & 7u];
        uint item = jbw_itemAt(seed, x + d.x, y + d.y);
        bool absent = !jbw_blocks(item)
            && jbw_destroyedAt(history, historyMask, x + d.x, y + d.y) != 0u;
        obs[i] = absent ? 0.0f : jbw_energy(item);
    }
    float3 scent = jbw_scentAt(seed, x, y, step, history, historyMask, response, slane, width);
    if (width > 1u) simdgroup_barrier(mem_flags::mem_device);
    if (slane != 0u) return;
    obs[8] = scent.x; obs[9] = scent.y; obs[10] = scent.z;
    obs[11] = float(words[3]);
    for (uint i = 0u; i < 4u; i++) obs[12u + i] = float(words[4u + i]);
}

inline float jbw_step(uint seed, uint action, device uint *words, device uint *history,
                      uint historyMask, device const float *response, uint step,
                      uint slane, uint width, device float *obs) {
    float reward = 0.0f;
    if (slane == 0u) {
        words[3] = 0u;
        if (action == 1u) words[2] = (words[2] + 3u) & 3u;
        else if (action == 2u) words[2] = (words[2] + 1u) & 3u;
        else {
            int2 forward[4] = { int2(0, -1), int2(1, 0), int2(0, 1), int2(-1, 0) };
            int2 p = int2(as_type<int>(words[0]), as_type<int>(words[1])) + forward[words[2] & 3u];
            uint item = jbw_itemAt(seed, p.x, p.y);
            if (!jbw_blocks(item)) {
                words[0] = as_type<uint>(p.x); words[1] = as_type<uint>(p.y); words[3] = 1u;
                if (jbw_collectible(item)
                    && jbw_destroyedAt(history, historyMask, p.x, p.y) == 0u) {
                    uint inventory = item == JBW_BANANA ? 4u : item == JBW_ONION ? 5u
                        : item == JBW_JELLY ? 6u : 7u;
                    words[inventory]++;
                    jbw_destroy(history, historyMask, p.x, p.y, step);
                }
            }
        }
        if (action == 0u && words[3] != 0u) {
            int x = as_type<int>(words[0]), y = as_type<int>(words[1]);
            uint item = jbw_itemAt(seed, x, y);
            if (jbw_destroyedAt(history, historyMask, x, y) == step + 1u) {
                float sign = (step / 100000u & 1u) == 0u ? 1.0f : -1.0f;
                if (item == JBW_JELLY) reward = sign;
                else if (item == JBW_ONION) reward = -sign;
            }
        }
    }
    if (width > 1u) {
        simdgroup_barrier(mem_flags::mem_device);
        reward = simd_broadcast_first(reward);
    }
    jbw_observe(seed, words, history, historyMask, response, step, slane, width, obs);
    return reward;
}
