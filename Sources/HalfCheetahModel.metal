constant float HC_TIMESTEP = 0.01f;
constant float HC_GRAVITY_Z = -9.81f;
constant float HC_SOLVER_TOL = 1e-5f;
constant float HC_LS_TOL = 0.01f;
constant int HC_SOLVER_ITER = 10;
constant int HC_LS_ITER = 15;
constant int HC_FRAME_SKIP = 5;
constant float HC_RESET_NOISE = 0.1f;
constant int HC_BODY_PARENT[8] = { 0, 0, 1, 2, 3, 1, 5, 6 };
constant float HC_BODY_POS[24] = {
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.7f,
    -0.5f, 0.0f, 0.0f,
    0.16f, 0.0f, -0.25f,
    -0.28f, 0.0f, -0.14f,
    0.5f, 0.0f, 0.0f,
    -0.14f, 0.0f, -0.24f,
    0.13f, 0.0f, -0.18f
};
constant float HC_BODY_QUAT[32] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f
};
constant float HC_BODY_IPOS[24] = {
    0.0f, 0.0f, 0.0f,
    0.152389878f, 0.0f, 0.025398313f,
    0.1f, 0.0f, -0.13f,
    -0.14f, 0.0f, -0.07f,
    0.03f, 0.0f, -0.097f,
    -0.07f, 0.0f, -0.12f,
    0.065f, 0.0f, -0.09f,
    0.045f, 0.0f, -0.07f
};
constant float HC_BODY_IQUAT[32] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.522429792f, 0.476515596f, 0.476515596f, 0.522429792f,
    -0.323289567f, -0.0f, -0.946300088f, -0.0f,
    0.527619932f, -0.0f, -0.849480552f, -0.0f,
    0.990901331f, -0.0f, -0.134590311f, -0.0f,
    0.966389978f, 0.0f, 0.257080552f, 0.0f,
    0.955336489f, -0.0f, -0.295520207f, -0.0f,
    0.955336489f, -0.0f, -0.295520207f, -0.0f
};
constant float HC_BODY_MASS[8] = {
    0.0f, 6.25020921f, 1.54351464f, 1.5874477f, 1.09539749f, 1.43807531f, 1.20083682f, 0.884518828f
};
constant float HC_BODY_INERTIA[24] = {
    0.0f, 0.0f, 0.0f,
    0.897117688f, 0.885655452f, 0.0179609234f,
    0.0168443396f, 0.0168443396f, 0.00157602159f,
    0.0182674191f, 0.0182674191f, 0.00162250276f,
    0.00635242326f, 0.00635242326f, 0.00110191364f,
    0.0137396433f, 0.0137396433f, 0.00146446678f,
    0.00822210862f, 0.00822210862f, 0.00121346845f,
    0.00352910946f, 0.00352910946f, 0.000878804017f
};
constant int HC_JNT_TYPE[9] = { 2, 2, 3, 3, 3, 3, 3, 3, 3 };
constant int HC_JNT_BODY[9] = { 1, 1, 1, 2, 3, 4, 5, 6, 7 };
constant float HC_JNT_POS[27] = {
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f
};
constant float HC_JNT_AXIS[27] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f
};
constant int HC_JNT_LIMITED[9] = { 0, 0, 0, 1, 1, 1, 1, 1, 1 };
constant float HC_JNT_RANGE[18] = {
    0.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 0.0f,
    -0.52f, 1.05f,
    -0.785f, 0.785f,
    -0.4f, 0.785f,
    -1.0f, 0.7f,
    -1.2f, 0.87f,
    -0.5f, 0.5f
};
constant float HC_JNT_SOLREF[18] = {
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f
};
constant float HC_JNT_SOLIMP[45] = {
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.03f, 0.5f, 2.0f
};
constant float HC_JNT_MARGIN[9] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};
constant float HC_DOF_ARMATURE[9] = {
    0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f
};
constant float HC_DOF_DAMPING[9] = {
    0.0f, 0.0f, 0.0f, 6.0f, 4.5f, 3.0f, 4.5f, 3.0f, 1.5f
};
constant float HC_JNT_STIFFNESS[9] = {
    0.0f, 0.0f, 0.0f, 240.0f, 180.0f, 120.0f, 180.0f, 120.0f, 60.0f
};
constant float HC_QPOS_SPRING[9] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};
constant float HC_QPOS0[9] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};
constant int HC_DOF_PARENT[9] = { -1, 0, 1, 2, 3, 4, 2, 6, 7 };
constant int HC_GEOM_BODY[9] = { 0, 1, 1, 2, 3, 4, 5, 6, 7 };
constant float HC_GEOM_POS[27] = {
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.6f, 0.0f, 0.1f,
    0.1f, 0.0f, -0.13f,
    -0.14f, 0.0f, -0.07f,
    0.03f, 0.0f, -0.097f,
    -0.07f, 0.0f, -0.12f,
    0.065f, 0.0f, -0.09f,
    0.045f, 0.0f, -0.07f
};
constant float HC_GEOM_QUAT[36] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.707106781f, 0.0f, -0.707106781f, 0.0f,
    0.906870042f, 0.0f, 0.421410401f, 0.0f,
    -0.323289567f, -0.0f, -0.946300088f, -0.0f,
    0.527619932f, -0.0f, -0.849480552f, -0.0f,
    0.990901331f, -0.0f, -0.134590311f, -0.0f,
    0.966389978f, 0.0f, 0.257080552f, 0.0f,
    0.955336489f, -0.0f, -0.295520207f, -0.0f,
    0.955336489f, -0.0f, -0.295520207f, -0.0f
};
constant float HC_GEOM_SIZE[27] = {
    40.0f, 40.0f, 40.0f,
    0.046f, 0.5f, 0.0f,
    0.046f, 0.15f, 0.0f,
    0.046f, 0.145f, 0.0f,
    0.046f, 0.15f, 0.0f,
    0.046f, 0.094f, 0.0f,
    0.046f, 0.133f, 0.0f,
    0.046f, 0.106f, 0.0f,
    0.046f, 0.07f, 0.0f
};
constant float HC_GEOM_FRICTION[27] = {
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f,
    0.4f, 0.1f, 0.1f
};
constant float HC_GEOM_SOLREF[18] = {
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f,
    0.02f, 1.0f
};
constant float HC_GEOM_SOLIMP[45] = {
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f,
    0.0f, 0.8f, 0.01f, 0.5f, 2.0f
};
constant int HC_ACT_TRNID[6] = { 3, 4, 5, 6, 7, 8 };
constant float HC_ACT_GEAR[6] = {
    120.0f, 90.0f, 60.0f, 120.0f, 60.0f, 30.0f
};
constant float HC_ACT_CTRLRANGE[12] = {
    -1.0f, 1.0f,
    -1.0f, 1.0f,
    -1.0f, 1.0f,
    -1.0f, 1.0f,
    -1.0f, 1.0f,
    -1.0f, 1.0f
};
