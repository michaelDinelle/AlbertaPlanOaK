#pragma METAL fp math_mode(safe)

constant uint ENV_RNG_STREAM       = 3u;
constant uint RNG_STREAM_ACT = 2u;
constant uint RNG_STREAM_RANDOM_ACT = 4u;
constant uint RNG_STREAM_PPO_INIT = 5u;
constant uint RNG_STREAM_PPO_ACT = 6u;
constant uint RNG_STREAM_PPO_SHUFFLE = 7u;
constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;
constant uint RNG_STREAM_RESET = 0u;

inline uint4 philox4x32(uint2 key, uint4 ctr) {
    for (int r = 0; r < 10; r++) {
        if (r > 0) { key.x += PHILOX_W0; key.y += PHILOX_W1; }
        uint hi0 = mulhi(PHILOX_M0, ctr.x), lo0 = PHILOX_M0 * ctr.x;
        uint hi1 = mulhi(PHILOX_M1, ctr.z), lo1 = PHILOX_M1 * ctr.z;
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    }
    return ctr;
}

inline float u01(uint x) { return float(x >> 8) * 0x1p-24f; }
inline float u01open(uint x) { return (float(x >> 8) + 1.0f) * 0x1p-24f; }
inline float uniformRange(uint x, float lo, float hi) { return lo + (hi - lo)*u01(x); }

inline float wrapPi(float angle) {
    return angle - 2.0f * M_PI_F * floor((angle + M_PI_F) / (2.0f * M_PI_F));
}

constant uint OAK_POSE_LEN = 9u;
constant uint OAK_RECORD_FLOATS = 1u + OAK_POSE_LEN;

inline int2 gridDelta(uint direction) {
    switch (direction & 3u) {
        case 0u: return int2(0, -1);
        case 1u: return int2(0, 1);
        case 2u: return int2(-1, 0);
        default: return int2(1, 0);
    }
}

template<typename WallFn>
inline int2 sampleFreeCell(uint laneSeed, uint resampleIndex, uint sampleBlock,
                           thread const WallFn &isWall, int excludeX, int excludeY) {
    for (;;) {
        uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, sampleBlock, 0u, 0u));
        uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
        for (int wi = 0; wi < 4; wi++) {
            uint idx = uint(u01(philoxWords[wi]) * 81.0f);
            int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
            if (!isWall(candidateX, candidateY)
                && (candidateX != excludeX || candidateY != excludeY)) {
                return int2(candidateX, candidateY);
            }
        }
        sampleBlock += 2u;
    }
}
