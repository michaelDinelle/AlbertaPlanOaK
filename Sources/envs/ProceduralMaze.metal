#pragma METAL fp math_mode(safe)

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
    uint4 philoxBlock = philox4x32(uint2(laneSeed, RNG_STREAM_RESET),
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
            philoxBlock = philox4x32(uint2(laneSeed, RNG_STREAM_RESET),
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

struct ProcMazeWall {
    uint b0, b1, b2;
    bool operator()(int x, int y) const { return proceduralMaze_isWall(b0, b1, b2, x, y); }
};

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
    int2 delta = gridDelta(action & 3u);
    int deltaX = delta.x, deltaY = delta.y;
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (proceduralMaze_isWall(wallBits0, wallBits1, wallBits2, nextX, nextY)) { nextX = int(x); nextY = int(y); }
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        ProcMazeWall wall = { wallBits0, wallBits1, wallBits2 };
        int2 agentCell = sampleFreeCell(laneSeed, resampleIndex, 0u, wall, 0, 0);
        nextX = agentCell.x; nextY = agentCell.y;
        int2 goalCell = sampleFreeCell(laneSeed, resampleIndex, 1u, wall, nextX, nextY);
        goalX = uint(goalCell.x); goalY = uint(goalCell.y);
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex;
    proceduralMaze_obs(uint(nextX), uint(nextY), goalX, goalY, nextStepsSinceGoal, wallBits0, wallBits1, wallBits2, obs);
    *doneOut = done;
    return reward;
}
