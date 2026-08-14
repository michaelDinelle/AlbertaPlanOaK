#ifndef OAK_GRIDWORLD_METAL_INCLUDED
#define OAK_GRIDWORLD_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)

constant uint ENV_RNG_STREAM       = 3u;
constant uint KEY_HELD             = 1u;
constant uint DOOR_UNLOCKED        = 2u;
constant uint KEY1_HELD            = 1u;
constant uint KEY2_HELD            = 2u;
constant uint KEY3_HELD            = 4u;
constant uint DOOR1_UNLOCKED       = 8u;
constant uint DOOR2_UNLOCKED       = 16u;
constant uint DOOR3_UNLOCKED       = 32u;

inline bool fourRooms_isWall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 5 && y != 2 && y != 8) return true;
    if (y == 5 && x != 2 && x != 8) return true;
    return false;
}

constant uchar FOUR_ROOMS_FREE_X[68] = {
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9,
    2, 8,
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9 };
constant uchar FOUR_ROOMS_FREE_Y[68] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 };

inline int fourRooms_room(int x, int y) {
    if (x < 5 && y < 5) return 0;
    if (x > 5 && y < 5) return 1;
    if (x < 5 && y > 5) return 2;
    if (x > 5 && y > 5) return 3;
    return -1;
}

inline void fourRooms_obs(uint x, uint y, uint goalX, uint goalY, uint stepsSinceGoal,
                        thread float *obs) {
    int room = fourRooms_room(int(x), int(y));
    int goalRoom = fourRooms_room(int(goalX), int(goalY));
    obs[0] = float(x) / 10.0f;
    obs[1] = float(y) / 10.0f;
    obs[2] = float(int(goalX) - int(x)) / 10.0f;
    obs[3] = float(int(goalY) - int(y)) / 10.0f;
    obs[4] = float(abs(int(goalX) - int(x)) + abs(int(goalY) - int(y))) / 20.0f;
    obs[5] = (room == 0) ? 1.0f : 0.0f;
    obs[6] = (room == 1) ? 1.0f : 0.0f;
    obs[7] = (room == 2) ? 1.0f : 0.0f;
    obs[8] = (room == 3) ? 1.0f : 0.0f;
    obs[9]  = fourRooms_isWall(int(x) + 0, int(y) - 1) ? 1.0f : 0.0f;
    obs[10] = fourRooms_isWall(int(x) + 0, int(y) + 1) ? 1.0f : 0.0f;
    obs[11] = fourRooms_isWall(int(x) - 1, int(y) + 0) ? 1.0f : 0.0f;
    obs[12] = fourRooms_isWall(int(x) + 1, int(y) + 0) ? 1.0f : 0.0f;
    obs[13] = (room == -1) ? 1.0f : 0.0f;
    obs[14] = (goalRoom == room && room != -1) ? 1.0f : 0.0f;
    obs[15] = min(float(stepsSinceGoal) / 256.0f, 1.0f);
    obs[16] = 1.0f;
}

inline float fourRooms_step(thread uint *words, uint laneSeed, uint action,
                          thread float *obs) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    int deltaX = 0, deltaY = 0;
    switch (action & 3u) {
        case 0u: deltaY = -1; break;
        case 1u: deltaY = 1; break;
        case 2u: deltaX = -1; break;
        default: deltaX = 1; break;
    }
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (fourRooms_isWall(nextX, nextY)) { nextX = int(x); nextY = int(y); }
    float reward = 0.0f;
    uint nextGoalX = goalX, nextGoalY = goalY, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint block = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, block, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                int idx = int(hc_u01(philoxWords[wi]) * 68.0f);
                if (idx < 68) {
                    uint candidateX = uint(FOUR_ROOMS_FREE_X[idx]), candidateY = uint(FOUR_ROOMS_FREE_Y[idx]);
                    if (candidateX != uint(nextX) || candidateY != uint(nextY)) {
                        nextGoalX = candidateX; nextGoalY = candidateY;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
            block += 1u;
        }
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = nextGoalX; words[3] = nextGoalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex;
    fourRooms_obs(uint(nextX), uint(nextY), nextGoalX, nextGoalY, nextStepsSinceGoal, obs);
    return reward;
}

inline void deepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    float gridExtent = 7.0f;
    obs[0] = float(row) / gridExtent;
    obs[1] = float(column) / gridExtent;
    obs[2] = float(7 - int(column)) / gridExtent;
    obs[3] = float(7 - int(row)) / gridExtent;
    obs[4] = (column == 7u) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = float(lastAction);
    obs[7] = 1.0f;
    for (int i = 8; i < 17; i++) obs[i] = 0.0f;
}

inline float deepSea_step(thread uint *words, uint action, thread float *obs) {
    uint row = words[0], column = words[1];
    uint mapBitIndex = row * 8u + column;
    uint rightwardAction = (mapBitIndex < 32u) ? ((words[3] >> mapBitIndex) & 1u)
                              : ((words[4] >> (mapBitIndex - 32u)) & 1u);
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = clamp(int(column) + columnDelta, 0, 7);
    uint nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? -0.00125f : 0.0f;
    uint nextLastAction = action & 1u;
    if (nextRow >= 7u) {
        if (nextColumn == 7) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    words[0] = nextRow;
    words[1] = uint(nextColumn);
    words[2] = nextLastAction;
    deepSea_obs(nextRow, uint(nextColumn), nextLastAction, obs);
    return reward;
}

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
        uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        uint agentCellIndex = uint(hc_u01(raw.x) * 15.0f);
        uint keyCellIndex = uint(hc_u01(raw.y) * 15.0f);
        uint goalCellIndex = uint(hc_u01(raw.z) * 15.0f);
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

inline float chain_step(thread uint *words, uint action, thread float *obs,
                           thread uint *doneOut) {
    uint pos = words[0];
    float reward = 0.0f;
    uint done = 0u;
    if ((action & 1u) != 0u) {
        pos += 1u;
        if (pos >= 15u) { reward = 1.0f; pos = 0u; done = 1u; }
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    words[0] = pos;
    obs[0] = float(pos) / 16.0f;
    obs[1] = 1.0f;
    for (int i = 2; i < 17; i++) obs[i] = 0.0f;
    *doneOut = done;
    return reward;
}

inline void largeDeepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    float gridExtent = 15.0f;
    obs[0] = float(row) / gridExtent;
    obs[1] = float(column) / gridExtent;
    obs[2] = float(15 - int(column)) / gridExtent;
    obs[3] = float(15 - int(row)) / gridExtent;
    obs[4] = (column == 15u) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = float(lastAction);
    obs[7] = 1.0f;
    for (int i = 8; i < 17; i++) obs[i] = 0.0f;
}

inline float largeDeepSea_step(thread uint *words, uint action, thread float *obs) {
    uint row = words[0] & 15u, column = (words[0] >> 4u) & 15u;
    uint mapBitIndex = row * 16u + column;
    uint rightwardAction = (words[1u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) & 1u;
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = clamp(int(column) + columnDelta, 0, 15);
    uint nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? -0.000625f : 0.0f;
    uint nextLastAction = action & 1u;
    if (nextRow >= 15u) {
        if (nextColumn == 15) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    words[0] = nextRow | (uint(nextColumn) << 4u) | (nextLastAction << 8u);
    largeDeepSea_obs(nextRow, uint(nextColumn), nextLastAction, obs);
    return reward;
}

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

inline float multiKeyCorridor_step(thread uint *words, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], key1Packed = words[2], key2Packed = words[3], key3Packed = words[4];
    uint goalX = words[5], goalY = words[6], keyDoorBits = words[7], stepsSinceGoal = words[8], resampleIndex = words[9];
    int deltaX = 0, deltaY = 0;
    switch (action & 3u) {
        case 0u: deltaY = -1; break;
        case 1u: deltaY = 1; break;
        case 2u: deltaX = -1; break;
        default: deltaX = 1; break;
    }
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
        uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        uint4 raw2 = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                   uint4(resampleIndex, 1u, 0u, 0u));
        uint agentCellIndex = uint(hc_u01(raw.x) * 15.0f);
        uint key1CellIndex = uint(hc_u01(raw.y) * 15.0f);
        uint key2CellIndex = uint(hc_u01(raw.z) * 15.0f);
        uint key3CellIndex = uint(hc_u01(raw.w) * 15.0f);
        uint goalCellIndex = uint(hc_u01(raw2.x) * 15.0f);
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

inline void memoryChain_obs(uint pos, uint cue, uint lastAction, thread float *obs) {
    obs[0] = float(pos) / 32.0f;
    obs[1] = (pos == 0u) ? float(cue) : 0.0f;
    obs[2] = (pos == 0u) ? 1.0f : 0.0f;
    obs[3] = (pos == 31u) ? 1.0f : 0.0f;
    obs[4] = float(lastAction);
    obs[5] = 1.0f;
    for (int i = 6; i < 17; i++) obs[i] = 0.0f;
}

inline float memoryChain_step(thread uint *words, uint laneSeed, uint action,
                            thread float *obs, thread uint *doneOut) {
    uint pos = words[0], cue = words[1], resampleIndex = words[3];
    float reward = 0.0f;
    uint done = 0u;
    if (pos >= 31u) {
        reward = ((action & 1u) == cue) ? 1.0f : 0.0f;
        done = 1u;
        uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        cue = raw.x & 1u;
        resampleIndex = resampleIndex + 1u;
        pos = 0u;
    } else if ((action & 1u) != 0u) {
        pos += 1u;
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    uint nextLastAction = action & 1u;
    words[0] = pos; words[1] = cue; words[2] = nextLastAction; words[3] = resampleIndex;
    memoryChain_obs(pos, cue, nextLastAction, obs);
    *doneOut = done;
    return reward;
}

inline void proceduralMaze_generate(uint laneSeed, thread uint *wallBits) {
    wallBits[0] = 0xffffffffu; wallBits[1] = 0xffffffffu; wallBits[2] = 0x1ffffu;
    for (uint j = 0u; j < 5u; j++)
        for (uint i = 0u; i < 5u; i++) {
            uint idx = 2u * i + 9u * (2u * j);
            wallBits[idx >> 5u] &= ~(1u << (idx & 31u));
        }
    uint visited = 1u;
    uint stack[25];
    uint stackSize = 1u;
    stack[0] = 0u;
    uint philoxCounter = 0u, philoxWordIndex = 0u;
    uint4 philoxBlock = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                              uint4(0u, 0u, 0u, 0u));
    while (stackSize > 0u) {
        uint currentCell = stack[stackSize - 1u];
        int cellI = int(currentCell % 5u), cellJ = int(currentCell / 5u);
        uint neighborPacked = 0u, neighborCount = 0u;
        const int dxs[4] = { 1, 0, -1, 0 };
        const int dys[4] = { 0, 1, 0, -1 };
        for (uint d = 0u; d < 4u; d++) {
            int nextI = cellI + dxs[d], nextJ = cellJ + dys[d];
            if (nextI >= 0 && nextI < 5 && nextJ >= 0 && nextJ < 5 &&
                ((visited >> uint(nextI + 5 * nextJ)) & 1u) == 0u) {
                neighborPacked |= uint(nextI + 5 * nextJ) << (8u * neighborCount);
                neighborCount++;
            }
        }
        if (neighborCount == 0u) { stackSize--; continue; }
        if (philoxWordIndex == 4u) {
            philoxCounter++;
            philoxBlock = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                                uint4(philoxCounter, 0u, 0u, 0u));
            philoxWordIndex = 0u;
        }
        uint philoxWords[4] = { philoxBlock.x, philoxBlock.y, philoxBlock.z, philoxBlock.w };
        uint nextCell = (neighborPacked >> (8u * (philoxWords[philoxWordIndex] % neighborCount))) & 0xffu;
        philoxWordIndex++;
        uint nextI = nextCell % 5u, nextJ = nextCell / 5u;
        uint idx = (uint(cellI) + nextI) + 9u * (uint(cellJ) + nextJ);
        wallBits[idx >> 5u] &= ~(1u << (idx & 31u));
        visited |= 1u << nextCell;
        stack[stackSize] = nextCell;
        stackSize++;
    }
}

inline bool proceduralMaze_isWall(uint wallBits0, uint wallBits1, uint wallBits2, int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    uint idx = uint(x - 1) + 9u * uint(y - 1);
    uint wallBits = (idx < 32u) ? wallBits0 : ((idx < 64u) ? wallBits1 : wallBits2);
    return ((wallBits >> (idx & 31u)) & 1u) != 0u;
}

inline void proceduralMaze_obs(uint x, uint y, uint goalX, uint goalY, uint stepsSinceGoal,
                         uint wallBits0, uint wallBits1, uint wallBits2, thread float *obs) {
    int ix = int(x), iy = int(y);
    obs[0] = float(x) / 10.0f;
    obs[1] = float(y) / 10.0f;
    obs[2] = float(int(goalX) - ix) / 10.0f;
    obs[3] = float(int(goalY) - iy) / 10.0f;
    obs[4] = float(abs(int(goalX) - ix) + abs(int(goalY) - iy)) / 16.0f;
    obs[5] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix + 0, iy - 1) ? 1.0f : 0.0f;
    obs[6] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix + 0, iy + 1) ? 1.0f : 0.0f;
    obs[7] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix - 1, iy + 0) ? 1.0f : 0.0f;
    obs[8] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix + 1, iy + 0) ? 1.0f : 0.0f;
    obs[9] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix + 1, iy - 1) ? 1.0f : 0.0f;
    obs[10] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix - 1, iy - 1) ? 1.0f : 0.0f;
    obs[11] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix + 1, iy + 1) ? 1.0f : 0.0f;
    obs[12] = proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, ix - 1, iy + 1) ? 1.0f : 0.0f;
    obs[13] = (4.0f - obs[5] - obs[6] - obs[7] - obs[8]) / 4.0f;
    obs[14] = min(float(stepsSinceGoal) / 256.0f, 1.0f);
    obs[15] = (x == goalX || y == goalY) ? 1.0f : 0.0f;
    obs[16] = 1.0f;
}

inline float proceduralMaze_step(thread uint *words, uint laneSeed, uint action,
                           thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    uint wallBits0 = words[6], wallBits1 = words[7], wallBits2 = words[8];
    int deltaX = 0, deltaY = 0;
    switch (action & 3u) {
        case 0u: deltaY = -1; break;
        case 1u: deltaY = 1; break;
        case 2u: deltaX = -1; break;
        default: deltaX = 1; break;
    }
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, nextX, nextY)) { nextX = int(x); nextY = int(y); }
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint sampleBlock = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, candidateX, candidateY)) {
                    nextX = candidateX; nextY = candidateY; found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
        sampleBlock = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, candidateX, candidateY) &&
                    (candidateX != nextX || candidateY != nextY)) {
                    goalX = uint(candidateX); goalY = uint(candidateY); found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex;
    proceduralMaze_obs(uint(nextX), uint(nextY), goalX, goalY, nextStepsSinceGoal, wallBits0, wallBits1, wallBits2, obs);
    *doneOut = done;
    return reward;
}

inline bool highDimMaze_isWall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 3 && y != 9) return true;
    if (x == 7 && y != 1) return true;
    return false;
}

inline void highDimMaze_obs(uint x, uint y, uint goalX, uint goalY, thread float *obs) {
    int ix = int(x), iy = int(y);
    obs[0] = float(x) / 16.0f;
    obs[1] = float(y) / 16.0f;
    obs[2] = highDimMaze_isWall(ix + 0, iy - 1) ? 1.0f : 0.0f;
    obs[3] = highDimMaze_isWall(ix + 0, iy + 1) ? 1.0f : 0.0f;
    obs[4] = highDimMaze_isWall(ix - 1, iy + 0) ? 1.0f : 0.0f;
    obs[5] = highDimMaze_isWall(ix + 1, iy + 0) ? 1.0f : 0.0f;
    obs[6] = float(int(goalX) - ix) / 16.0f;
    obs[7] = float(int(goalY) - iy) / 16.0f;
    for (uint bit = 0u; bit < 4u; bit++) {
        obs[8 + bit] = float((x >> bit) & 1u);
        obs[12 + bit] = float((y >> bit) & 1u);
    }
    obs[16] = 0.0f;
}

inline float highDimMaze_step(thread uint *words, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    uint lastDirection = words[6];
    uint actionIndex = action % 6u;
    int nextX = int(x), nextY = int(y);
    if (actionIndex < 4u) {
        int deltaX = 0, deltaY = 0;
        switch (actionIndex) {
            case 0u: deltaY = -1; break;
            case 1u: deltaY = 1; break;
            case 2u: deltaX = -1; break;
            default: deltaX = 1; break;
        }
        lastDirection = actionIndex;
        if (!highDimMaze_isWall(nextX + deltaX, nextY + deltaY)) { nextX += deltaX; nextY += deltaY; }
    } else if (actionIndex == 5u) {
        int deltaX = 0, deltaY = 0;
        switch (lastDirection & 3u) {
            case 0u: deltaY = -1; break;
            case 1u: deltaY = 1; break;
            case 2u: deltaX = -1; break;
            default: deltaX = 1; break;
        }
        if (!highDimMaze_isWall(nextX + deltaX, nextY + deltaY)) {
            nextX += deltaX; nextY += deltaY;
            if (!highDimMaze_isWall(nextX + deltaX, nextY + deltaY)) { nextX += deltaX; nextY += deltaY; }
        }
    }
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint sampleBlock = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!highDimMaze_isWall(candidateX, candidateY)) {
                    nextX = candidateX; nextY = candidateY; found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
        sampleBlock = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!highDimMaze_isWall(candidateX, candidateY) && (candidateX != nextX || candidateY != nextY)) {
                    goalX = uint(candidateX); goalY = uint(candidateY); found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex; words[6] = lastDirection;
    highDimMaze_obs(uint(nextX), uint(nextY), goalX, goalY, obs);
    *doneOut = done;
    return reward;
}

inline bool poMaze_isWall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (y == 3 && x != 8) return true;
    if (y == 7 && x != 2) return true;
    return false;
}

inline void poMaze_obs(uint x, uint y, uint goalX, uint goalY, thread float *obs) {
    int ix = int(x), iy = int(y);
    obs[0] = poMaze_isWall(ix + 0, iy - 1) ? 1.0f : 0.0f;
    obs[1] = poMaze_isWall(ix + 1, iy - 1) ? 1.0f : 0.0f;
    obs[2] = poMaze_isWall(ix + 1, iy + 0) ? 1.0f : 0.0f;
    obs[3] = poMaze_isWall(ix + 1, iy + 1) ? 1.0f : 0.0f;
    obs[4] = poMaze_isWall(ix + 0, iy + 1) ? 1.0f : 0.0f;
    obs[5] = poMaze_isWall(ix - 1, iy + 1) ? 1.0f : 0.0f;
    obs[6] = poMaze_isWall(ix - 1, iy + 0) ? 1.0f : 0.0f;
    obs[7] = poMaze_isWall(ix - 1, iy - 1) ? 1.0f : 0.0f;
    obs[8] = float(x) / 10.0f;
    obs[9] = float(y) / 10.0f;
    obs[10] = (goalX > x) ? 1.0f : 0.0f;
    obs[11] = (goalX < x) ? 1.0f : 0.0f;
    obs[12] = (goalY > y) ? 1.0f : 0.0f;
    obs[13] = (goalY < y) ? 1.0f : 0.0f;
    obs[14] = 1.0f;
    obs[15] = 0.0f;
    obs[16] = 0.0f;
}

inline float poMaze_step(thread uint *words, uint laneSeed, uint action,
                        thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    int deltaX = 0, deltaY = 0;
    switch (action & 3u) {
        case 0u: deltaY = -1; break;
        case 1u: deltaY = 1; break;
        case 2u: deltaX = -1; break;
        default: deltaX = 1; break;
    }
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (poMaze_isWall(nextX, nextY)) { nextX = int(x); nextY = int(y); }
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        uint sampleBlock = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!poMaze_isWall(candidateX, candidateY)) {
                    nextX = candidateX; nextY = candidateY; found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
        sampleBlock = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(philoxWords[wi]) * 81.0f);
                int candidateX = 1 + int(idx % 9u), candidateY = 1 + int(idx / 9u);
                if (!poMaze_isWall(candidateX, candidateY) && (candidateX != nextX || candidateY != nextY)) {
                    goalX = uint(candidateX); goalY = uint(candidateY); found = true; break;
                }
            }
            if (found) break;
            sampleBlock += 2u;
        }
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex;
    poMaze_obs(uint(nextX), uint(nextY), goalX, goalY, obs);
    *doneOut = done;
    return reward;
}

inline float delayedChain_step(thread uint *words, uint action, thread float *obs,
                            thread uint *doneOut) {
    uint pos = words[0];
    float reward = 0.0f;
    uint done = 0u;
    if ((action & 1u) != 0u) {
        pos += 1u;
        if (pos >= 31u) { reward = 1.0f; pos = 0u; done = 1u; }
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    uint nextLastAction = action & 1u;
    words[0] = pos;
    words[1] = nextLastAction;
    obs[0] = float(pos) / 32.0f;
    obs[1] = float(nextLastAction);
    obs[2] = 1.0f;
    for (int i = 3; i < 17; i++) obs[i] = 0.0f;
    *doneOut = done;
    return reward;
}

#endif
