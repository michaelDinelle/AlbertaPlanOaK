#pragma METAL fp math_mode(safe)

inline bool poMaze_isWall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (y == 3 && x != 8) return true;
    if (y == 7 && x != 2) return true;
    return false;
}

struct PoMazeWall { bool operator()(int x, int y) const { return poMaze_isWall(x, y); } };

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

inline float poMaze_step(device uint *words, uint laneSeed, uint action,
                        thread float *obs, thread uint *doneOut) {
    uint x = words[0], y = words[1], goalX = words[2], goalY = words[3], stepsSinceGoal = words[4], resampleIndex = words[5];
    int2 delta = gridDelta(action & 3u);
    int deltaX = delta.x, deltaY = delta.y;
    int nextX = int(x) + deltaX, nextY = int(y) + deltaY;
    if (poMaze_isWall(nextX, nextY)) { nextX = int(x); nextY = int(y); }
    float reward = 0.0f;
    uint done = 0u, nextStepsSinceGoal = stepsSinceGoal + 1u, nextResampleIndex = resampleIndex;
    if (uint(nextX) == goalX && uint(nextY) == goalY) {
        reward = 1.0f;
        done = 1u;
        nextStepsSinceGoal = 0u;
        nextResampleIndex = resampleIndex + 1u;
        PoMazeWall wall;
        int2 agentCell = sampleFreeCell(laneSeed, resampleIndex, 0u, wall, 0, 0);
        nextX = agentCell.x; nextY = agentCell.y;
        int2 goalCell = sampleFreeCell(laneSeed, resampleIndex, 1u, wall, nextX, nextY);
        goalX = uint(goalCell.x); goalY = uint(goalCell.y);
    }
    words[0] = uint(nextX); words[1] = uint(nextY); words[2] = goalX; words[3] = goalY;
    words[4] = nextStepsSinceGoal; words[5] = nextResampleIndex;
    poMaze_obs(uint(nextX), uint(nextY), goalX, goalY, obs);
    *doneOut = done;
    return reward;
}
