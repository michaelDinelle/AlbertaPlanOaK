#pragma METAL fp math_mode(safe)

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
            uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(u01(philoxWords[wi]) * 81.0f);
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
            uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                      uint4(resampleIndex, sampleBlock, 0u, 0u));
            uint philoxWords[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(u01(philoxWords[wi]) * 81.0f);
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
