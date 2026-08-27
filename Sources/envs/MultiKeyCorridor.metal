#pragma METAL fp math_mode(safe)

constant uint KEY1_HELD            = 1u;
constant uint KEY2_HELD            = 2u;
constant uint KEY3_HELD            = 4u;
constant uint DOOR1_UNLOCKED       = 8u;
constant uint DOOR2_UNLOCKED       = 16u;
constant uint DOOR3_UNLOCKED       = 32u;

inline bool multiKeyCorridor_isWall(int x, int y) {
    if (x <= 0 || x >= 16 || y <= 0 || y >= 6) return true;
    if ((x == 4 || x == 8 || x == 12) && y != 3) return true;
    return false;
}

inline void multiKeyCorridor_obs(uint x, uint y, uint key1Packed, uint key2Packed, uint key3Packed,
                       uint goalX, uint goalY, uint keyDoorBits, thread float *obs) {
    bool hasKey1 = (keyDoorBits & KEY1_HELD) != 0u;
    bool hasKey2 = (keyDoorBits & KEY2_HELD) != 0u;
    bool hasKey3 = (keyDoorBits & KEY3_HELD) != 0u;
    obs[0] = float(x) / 16.0f;
    obs[1] = float(y) / 6.0f;
    obs[2] = hasKey1 ? 1.0f : 0.0f;
    obs[3] = hasKey2 ? 1.0f : 0.0f;
    obs[4] = hasKey3 ? 1.0f : 0.0f;
    obs[5] = ((keyDoorBits & DOOR1_UNLOCKED) != 0u) ? 1.0f : 0.0f;
    obs[6] = ((keyDoorBits & DOOR2_UNLOCKED) != 0u) ? 1.0f : 0.0f;
    obs[7] = ((keyDoorBits & DOOR3_UNLOCKED) != 0u) ? 1.0f : 0.0f;
    obs[8]  = hasKey1 ? 0.0f : float(int(key1Packed & 15u) - int(x)) / 16.0f;
    obs[9]  = hasKey1 ? 0.0f : float(int(key1Packed >> 4u) - int(y)) / 6.0f;
    obs[10] = hasKey2 ? 0.0f : float(int(key2Packed & 15u) - int(x)) / 16.0f;
    obs[11] = hasKey2 ? 0.0f : float(int(key2Packed >> 4u) - int(y)) / 6.0f;
    obs[12] = hasKey3 ? 0.0f : float(int(key3Packed & 15u) - int(x)) / 16.0f;
    obs[13] = hasKey3 ? 0.0f : float(int(key3Packed >> 4u) - int(y)) / 6.0f;
    obs[14] = float(int(goalX) - int(x)) / 16.0f;
    obs[15] = float(int(goalY) - int(y)) / 6.0f;
    obs[16] = 1.0f;
}

inline float multiKeyCorridor_step(device uint *words, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], key1Packed = words[2], key2Packed = words[3], key3Packed = words[4];
    uint goalX = words[5], goalY = words[6], keyDoorBits = words[7], stepsSinceGoal = words[8], resampleIndex = words[9];
    int2 delta = gridDelta(action & 3u);
    int deltaX = delta.x, deltaY = delta.y;
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (multiKeyCorridor_isWall(nextX, nextY)) { nextX = int(x); nextY = int(y); }
    else if (nextY == 3 && (nextX == 4 || nextX == 8 || nextX == 12)) {
        uint doorIndex = uint(nextX >> 2);
        uint doorUnlockMask = DOOR1_UNLOCKED << (doorIndex - 1u);
        uint doorKeyMask = KEY1_HELD << (doorIndex - 1u);
        if ((keyDoorBits & doorUnlockMask) == 0u) {
            if ((keyDoorBits & doorKeyMask) != 0u) keyDoorBits |= doorUnlockMask;
            else { nextX = int(x); nextY = int(y); }
        }
    }
    if ((keyDoorBits & KEY1_HELD) == 0u && uint(nextX) == (key1Packed & 15u) && uint(nextY) == (key1Packed >> 4u))
        keyDoorBits |= KEY1_HELD;
    if ((keyDoorBits & KEY2_HELD) == 0u && uint(nextX) == (key2Packed & 15u) && uint(nextY) == (key2Packed >> 4u))
        keyDoorBits |= KEY2_HELD;
    if ((keyDoorBits & KEY3_HELD) == 0u && uint(nextX) == (key3Packed & 15u) && uint(nextY) == (key3Packed >> 4u))
        keyDoorBits |= KEY3_HELD;
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        uint4 raw2 = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                   uint4(resampleIndex, 1u, 0u, 0u));
        uint agentCellIndex = uint(u01(raw.x) * 15.0f);
        uint key1CellIndex = uint(u01(raw.y) * 15.0f);
        uint key2CellIndex = uint(u01(raw.z) * 15.0f);
        uint key3CellIndex = uint(u01(raw.w) * 15.0f);
        uint goalCellIndex = uint(u01(raw2.x) * 15.0f);
        nextX = int(1u + agentCellIndex % 3u); nextY = int(1u + agentCellIndex / 3u);
        key1Packed = (1u + key1CellIndex % 3u) | ((1u + key1CellIndex / 3u) << 4u);
        key2Packed = (5u + key2CellIndex % 3u) | ((1u + key2CellIndex / 3u) << 4u);
        key3Packed = (9u + key3CellIndex % 3u) | ((1u + key3CellIndex / 3u) << 4u);
        goalX = 13u + goalCellIndex % 3u; goalY = 1u + goalCellIndex / 3u;
        keyDoorBits = 0u;
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = key1Packed; words[3] = key2Packed; words[4] = key3Packed;
    words[5] = goalX; words[6] = goalY; words[7] = keyDoorBits; words[8] = nextStepsSinceGoal; words[9] = nextResampleIndex;
    multiKeyCorridor_obs(uint(nextX), uint(nextY), key1Packed, key2Packed, key3Packed, goalX, goalY, keyDoorBits, obs);
    *doneOut = done;
    return reward;
}
