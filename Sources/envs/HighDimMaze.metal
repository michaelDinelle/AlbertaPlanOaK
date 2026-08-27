#pragma METAL fp math_mode(safe)

inline bool highDimMaze_isWall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 3 && y != 9) return true;
    if (x == 7 && y != 1) return true;
    return false;
}

struct HighDimWall { bool operator()(int x, int y) const { return highDimMaze_isWall(x, y); } };

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

inline float highDimMaze_step(device uint *words, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    uint lastDirection = words[6];
    uint actionIndex = action % 6u;
    int nextX = int(x), nextY = int(y);
    if (actionIndex < 4u) {
        int2 delta = gridDelta(actionIndex);
        int deltaX = delta.x, deltaY = delta.y;
        lastDirection = actionIndex;
        if (!highDimMaze_isWall(nextX + deltaX, nextY + deltaY)) { nextX += deltaX; nextY += deltaY; }
    } else if (actionIndex == 5u) {
        int2 delta = gridDelta(lastDirection & 3u);
        int deltaX = delta.x, deltaY = delta.y;
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
        HighDimWall wall;
        int2 agentCell = sampleFreeCell(laneSeed, resampleIndex, 0u, wall, 0, 0);
        nextX = agentCell.x; nextY = agentCell.y;
        int2 goalCell = sampleFreeCell(laneSeed, resampleIndex, 1u, wall, nextX, nextY);
        goalX = uint(goalCell.x); goalY = uint(goalCell.y);
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex; words[6] = lastDirection;
    highDimMaze_obs(uint(nextX), uint(nextY), goalX, goalY, obs);
    *doneOut = done;
    return reward;
}
