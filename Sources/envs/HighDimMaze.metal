#pragma METAL fp math_mode(safe)

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
            uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(u01(philoxWords[wi]) * 81.0f);
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
            uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(u01(philoxWords[wi]) * 81.0f);
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
