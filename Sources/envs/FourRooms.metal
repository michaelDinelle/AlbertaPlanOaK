#pragma METAL fp math_mode(safe)

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
            uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, block, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                int idx = int(u01(philoxWords[wi]) * 68.0f);
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
