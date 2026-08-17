#pragma METAL fp math_mode(safe)

constant uint KEY_HELD             = 1u;
constant uint DOOR_UNLOCKED        = 2u;

inline bool keyCorridor_isWall(int x, int y) {
    if (x <= 0 || x >= 8 || y <= 0 || y >= 6) return true;
    if (x == 4 && y != 3) return true;
    return false;
}

inline void keyCorridor_obs(uint x, uint y, uint keyX, uint keyY, uint goalX, uint goalY,
                      uint keyDoorBits, uint stepsSinceGoal, thread float *obs) {
    bool hasKey = (keyDoorBits & KEY_HELD) != 0u;
    obs[0] = float(x) / 8.0f;
    obs[1] = float(y) / 6.0f;
    obs[2] = hasKey ? 1.0f : 0.0f;
    obs[3] = ((keyDoorBits & DOOR_UNLOCKED) != 0u) ? 1.0f : 0.0f;
    obs[4] = hasKey ? 0.0f : 1.0f;
    obs[5] = hasKey ? 0.0f : float(int(keyX) - int(x)) / 8.0f;
    obs[6] = hasKey ? 0.0f : float(int(keyY) - int(y)) / 6.0f;
    obs[7] = float(4 - int(x)) / 8.0f;
    obs[8] = float(3 - int(y)) / 6.0f;
    obs[9] = float(int(goalX) - int(x)) / 8.0f;
    obs[10] = float(int(goalY) - int(y)) / 6.0f;
    obs[11] = (x == 4u && y == 3u) ? 1.0f : 0.0f;
    obs[12] = min(float(stepsSinceGoal) / 256.0f, 1.0f);
    obs[13] = 1.0f;
    for (int i = 14; i < 17; i++) obs[i] = 0.0f;
}

inline float keyCorridor_step(thread uint *words, uint laneSeed, uint action,
                        thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], keyX = words[2], keyY = words[3], goalX = words[4], goalY = words[5];
    uint keyDoorBits = words[6], stepsSinceGoal = words[7], resampleIndex = words[8];
    int deltaX = 0, deltaY = 0;
    switch (action & 3u) {
        case 0u: deltaY = -1; break;
        case 1u: deltaY = 1; break;
        case 2u: deltaX = -1; break;
        default: deltaX = 1; break;
    }
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (keyCorridor_isWall(nextX, nextY)) { nextX = int(x); nextY = int(y); }
    else if (nextX == 4 && nextY == 3 && (keyDoorBits & DOOR_UNLOCKED) == 0u) {
        if ((keyDoorBits & KEY_HELD) != 0u) keyDoorBits |= DOOR_UNLOCKED;
        else { nextX = int(x); nextY = int(y); }
    }
    if ((keyDoorBits & KEY_HELD) == 0u && uint(nextX) == keyX && uint(nextY) == keyY) keyDoorBits |= KEY_HELD;
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        uint agentCellIndex = uint(u01(raw.x) * 15.0f);
        uint keyCellIndex = uint(u01(raw.y) * 15.0f);
        uint goalCellIndex = uint(u01(raw.z) * 15.0f);
        nextX = int(1u + agentCellIndex % 3u); nextY = int(1u + agentCellIndex / 3u);
        keyX = 1u + keyCellIndex % 3u; keyY = 1u + keyCellIndex / 3u;
        goalX = 5u + goalCellIndex % 3u; goalY = 1u + goalCellIndex / 3u;
        keyDoorBits = 0u;
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = keyX; words[3] = keyY;
    words[4] = goalX; words[5] = goalY; words[6] = keyDoorBits; words[7] = nextStepsSinceGoal; words[8] = nextResampleIndex;
    keyCorridor_obs(uint(nextX), uint(nextY), keyX, keyY, goalX, goalY, keyDoorBits, nextStepsSinceGoal, obs);
    *doneOut = done;
    return reward;
}
