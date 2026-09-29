#include <string.h>
#include "../env.h"

#define PLANAR_BODIES 7
#define PLANAR_DOFS 9
#define PLANAR_JOINTS 6
#define PLANAR_CAPSULES 8
#define PLANAR_UNITS 22
#define PLANAR_ROWS 54
#define PLANAR_SUBSTEPS 5
#define PLANAR_TIMESTEP 0.01f
#define PLANAR_GRAVITY 9.81f
#define PLANAR_RADIUS 0.046f
#define PLANAR_FRICTION 0.4f
#define PLANAR_ARMATURE 0.1f
#define PLANAR_TIMECONST 0.02f
#define PLANAR_DAMPRATIO 1.0f
#define PLANAR_IMPEDANCE_LOW 0.0001f
#define PLANAR_IMPEDANCE_HIGH 0.8f
#define PLANAR_IMPEDANCE_MID 0.5f
#define PLANAR_LIMIT_WIDTH 0.03f
#define PLANAR_CONTACT_WIDTH 0.01f
#define PLANAR_TOLERANCE 1e-5f
#define PLANAR_ITERATIONS 10
#define PLANAR_SEARCH_TOLERANCE 0.01f
#define PLANAR_SEARCH_EVALUATIONS 15
#define PLANAR_MINVAL 1e-15f
#define PLANAR_BAD 1e10f
#define PLANAR_RESET_NOISE 0.1f
#define PLANAR_CONTROL_COST 0.1f
#define PLANAR_GAIT_RATE 863u
#define PLANAR_GAIT_CYCLE 1024u
#define PLANAR_GAIT_JOINTS 6
#define PLANAR_GAIT_FEATURES 22
#define PLANAR_GAIT_VELOCITY_SCALE 0.1f
#define PLANAR_GAIT_WEIGHTS (PLANAR_GAIT_JOINTS * PLANAR_GAIT_FEATURES)
#define PLANAR_FRICTION_LOW 0.02f
#define PLANAR_FRICTION_HIGH 2.0f

typedef struct { uint32_t robot, frictionPeriod; } planarKnobs;

static const planarKnobs PLANAR_KNOBS[5] = {
    { 0u, 0u },
    { 0u, 200000u },
    { 1u, 0u },
    { 2u, 0u },
    { 3u, 0u }
};

static const int PLANAR_PARENT[PLANAR_BODIES] = { -1, 0, 1, 2, 0, 4, 5 };
static const float PLANAR_BODY_X[PLANAR_BODIES] = { 0.0f, -0.5f, 0.16f, -0.28f, 0.5f, -0.14f, 0.13f };
static const float PLANAR_BODY_Z[PLANAR_BODIES] = { 0.7f, 0.0f, -0.25f, -0.14f, 0.0f, -0.24f, -0.18f };
static const float PLANAR_COM_X[PLANAR_BODIES] = { 0.15238988f, 0.1f, -0.14f, 0.03f, -0.07f, 0.065f, 0.045f };
static const float PLANAR_COM_Z[PLANAR_BODIES] = { 0.025398312f, -0.13f, -0.07f, -0.097f, -0.12f, -0.09f, -0.07f };
static const float PLANAR_MASS[PLANAR_BODIES] = { 6.2502093f, 1.5435146f, 1.5874476f, 1.0953975f, 1.4380753f, 1.2008368f, 0.8845188f };
static const float PLANAR_INERTIA_A[PLANAR_BODIES] = { 0.8971177f, 0.01684434f, 0.01826742f, 0.006352423f, 0.013739644f, 0.008222109f, 0.0035291095f };
static const float PLANAR_INERTIA_B[PLANAR_BODIES] = { 0.88565546f, 0.01684434f, 0.01826742f, 0.006352423f, 0.013739644f, 0.008222109f, 0.0035291095f };
static const float PLANAR_INERTIA_C[PLANAR_BODIES] = { 0.017960923f, 0.0015760215f, 0.0016225028f, 0.0011019136f, 0.0014644668f, 0.0012134685f, 0.000878804f };
static const float PLANAR_FRAME_W[PLANAR_BODIES] = { 0.52242976f, -0.32328957f, 0.52761996f, 0.99090135f, 0.96638995f, 0.9553365f, 0.9553365f };
static const float PLANAR_FRAME_X[PLANAR_BODIES] = { 0.4765156f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float PLANAR_FRAME_Y[PLANAR_BODIES] = { 0.4765156f, -0.9463001f, -0.84948057f, -0.13459031f, 0.25708055f, -0.29552022f, -0.29552022f };
static const float PLANAR_FRAME_Z[PLANAR_BODIES] = { 0.52242976f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float PLANAR_BODY_INVWEIGHT[PLANAR_BODIES] = { 0.064157516f, 0.09691101f, 0.12720923f, 0.24374929f, 0.081483796f, 0.13199683f, 0.2661441f };
static const float PLANAR_JOINT_INVWEIGHT[PLANAR_JOINTS] = { 2.7297885f, 5.893249f, 8.894877f, 3.0813925f, 6.882796f, 9.468135f };
static const float PLANAR_RANGE_LOW[PLANAR_JOINTS] = { -0.52f, -0.785f, -0.4f, -1.0f, -1.2f, -0.5f };
static const float PLANAR_RANGE_HIGH[PLANAR_JOINTS] = { 1.05f, 0.785f, 0.785f, 0.7f, 0.87f, 0.5f };
static const float PLANAR_STIFFNESS[PLANAR_JOINTS] = { 240.0f, 180.0f, 120.0f, 180.0f, 120.0f, 60.0f };
static const float PLANAR_DAMPING[PLANAR_JOINTS] = { 6.0f, 4.5f, 3.0f, 4.5f, 3.0f, 1.5f };
static const float PLANAR_GEAR[PLANAR_JOINTS] = { 120.0f, 90.0f, 60.0f, 120.0f, 60.0f, 30.0f };
static const int PLANAR_CAPSULE_BODY[PLANAR_CAPSULES] = { 0, 0, 1, 2, 3, 4, 5, 6 };
static const float PLANAR_CAPSULE_X[PLANAR_CAPSULES] = { 0.0f, 0.6f, 0.1f, -0.14f, 0.03f, -0.07f, 0.065f, 0.045f };
static const float PLANAR_CAPSULE_Z[PLANAR_CAPSULES] = { 0.0f, 0.1f, -0.13f, -0.07f, -0.097f, -0.12f, -0.09f, -0.07f };
static const float PLANAR_CAPSULE_W[PLANAR_CAPSULES] = { 0.70710677f, 0.90687007f, -0.32328957f, 0.52761996f, 0.99090135f, 0.96638995f, 0.9553365f, 0.9553365f };
static const float PLANAR_CAPSULE_Y[PLANAR_CAPSULES] = { -0.70710677f, 0.4214104f, -0.9463001f, -0.84948057f, -0.13459031f, 0.25708055f, -0.29552022f, -0.29552022f };
static const float PLANAR_CAPSULE_HALF[PLANAR_CAPSULES] = { 0.5f, 0.15f, 0.145f, 0.15f, 0.094f, 0.133f, 0.106f, 0.07f };
static const float PLANAR_REACH[2] = { 0.55f, 0.07f };
static const float PLANAR_BALANCE[6] = { 0.091540473f, 0.872626488f, 4.317510383f, 0.154926278f, 0.56800279f, 0.641819711f };
static const float PLANAR_HOP[36] = { -0.254116767f, -9.1988082f, 3.18724541f, 64.2199359f, -0.468416905f, -0.123377112f, 0.381628916f, -0.341405304f, -1.14930506f, 6.18497176f, -0.0123994589f, -8.78957185f, 0.278523473f, 27.0984002f, 15.687303f, -0.514467356f, -0.605789051f, 0.29073547f, -0.878402801f, -0.353249293f, 0.699791895f, 0.0255354191f, 0.352292136f, -6.18342889f, 1.78476235f, 18.1336768f, -3.33080265f, -1.19273871f, -2.09562655f, 0.0640835855f, 0.538891323f, -0.540366236f, -0.233984322f, 1.57064081f, 13.5649563f, 1.84428763f };
static const float PLANAR_GAIT[PLANAR_GAIT_WEIGHTS] = {
    0.11497268f, 0.45801052f, -0.14563811f, -0.16641575f, -0.49688584f, 0.018740816f, 0.26709297f, 0.29564553f, 0.17028837f, -0.03834736f, -7.787545f, -0.18268165f, -0.36449438f, -0.33905995f, -0.21346816f, 0.26423457f, 0.12415638f, 0.086894974f, 4.0523224f, -0.52783436f, 0.55602235f, 0.058483846f,
    -0.37348303f, 2.4635518f, 1.1626332f, 0.25923362f, 0.3762549f, -0.32470426f, -0.2156724f, 0.0039253663f, 0.39417097f, -0.18204193f, -0.7828645f, -0.13824913f, 0.12644279f, -0.17205298f, 0.17275746f, 0.10730755f, -0.099952124f, 0.24874848f, 0.17965643f, 0.19026783f, -0.13318212f, -1.6883546f,
    -0.24894585f, 1.2850789f, -0.6960399f, -0.07301445f, 0.5884383f, 0.49041712f, 0.15360023f, 0.4447213f, -0.20493819f, -0.17245655f, -1.6527301f, 0.064139456f, 0.092524804f, -0.1503786f, -0.45257348f, -0.6502207f, -0.2866701f, -2.0311363f, 0.82456475f, -0.4520124f, 0.2625945f, -2.0190122f,
    -0.7553374f, 2.025553f, -0.09560377f, 0.6192333f, 0.007354124f, 0.3372569f, 0.8478269f, -0.3448508f, 0.34587184f, -0.12269863f, 2.5857184f, -0.07159611f, 0.1038109f, 0.28196228f, 0.066299096f, -0.003317496f, -0.075372525f, 0.1396294f, -0.6941928f, 0.59090835f, -0.13326104f, -0.6962181f,
    1.3508446f, 0.35733682f, 0.0049215113f, -0.34968394f, 0.159393f, -0.5793149f, 0.82104874f, -0.14622867f, 0.012984726f, 0.67777693f, 2.715285f, 0.05452874f, 0.24944545f, 0.13297698f, 0.32070172f, 0.19900973f, -0.13688526f, -0.21083029f, -0.94304025f, -0.065114595f, -0.31424943f, -0.85279405f,
    -0.29504102f, -0.16538349f, 0.56299126f, -0.33034608f, 0.036004126f, 0.48399404f, -0.39242572f, 0.04924929f, -0.30804688f, -0.42593196f, 7.547394f, -0.3208801f, -0.12516011f, -0.3499814f, 0.06835229f, -0.14447162f, -0.029443515f, 0.05062591f, 0.32741156f, -0.2041525f, -0.16651943f, -3.2611573f
};

typedef double planarReal;
typedef long planarWord;
typedef planarReal planarPair __attribute__((ext_vector_type(2)));
typedef planarWord planarLane __attribute__((ext_vector_type(2)));
typedef planarReal planarLoose __attribute__((may_alias));

typedef struct { planarReal friction; planarPair pyramid[4]; } planarGrip;

typedef struct {
    planarReal mass, torsoInertia, torsoMomentX, torsoMomentZ, torsoHeight, gravity, timestep, stiffness, damping;
    planarReal lowImpedance, highImpedance, impedanceSpan, limitScale, contactScale, radius, bad, flightForce;
    planarPair inertia[3], momentX[3], momentZ[3], torsoPair[3][2], legPair[3][2], linkX[3], linkZ[3], armature;
    planarPair capsuleX[4], capsuleZ[4], axisX[4], axisZ[4], reach[4], sphereX[4][2], sphereZ[4][2];
    planarGrip grip;
    planarPair low[3], high[3], spring[3], damper[3], drag[3], still[3], limitWeight[3];
    planarPair sine[6], cosine[7], quadrant[3];
    planarLane lane[2], hitBit[4][2];
    float4 gait[PLANAR_GAIT_FEATURES][2];
} planarModel;

typedef planarModel planarTable __attribute__((aligned(4)));
typedef planarPair planarPairTable __attribute__((aligned(4)));
typedef planarGrip planarGripTable __attribute__((aligned(4)));

typedef struct { planarReal r[3]; planarPair l[3]; } planarVec;
typedef struct { planarReal r[6]; planarPair c[9], l[6]; } planarArrow;
typedef struct { planarPair adj[6], legInv; planarReal s[6], rootInv; } planarFactor;
typedef struct { planarVec n, t; planarReal an, at, d, tangent; int slot, lane; } planarUnit;
typedef struct { double x; planarReal q[3], v[3], w[3]; planarPair ql[3], vl[3], wl[3]; } planarState;
typedef float planarHalf __attribute__((ext_vector_type(2)));

typedef struct {
    planarArrow m;
    planarFactor mf, df;
    planarVec fs, smooth, qacc, force;
    planarUnit unit[PLANAR_UNITS];
    uint64_t active;
    int units, limits, rows, torso, path;
} planarForward;

static const double PLANAR_SINE[6] = { 1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0, -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0 };
static const double PLANAR_COSINE[7] = { -1.0 / 87178291200.0, 1.0 / 479001600.0, -1.0 / 3628800.0, 1.0 / 40320.0, -1.0 / 720.0, 1.0 / 24.0, -0.5 };

static planarPair planar_pair(planarReal a, planarReal b) { planarPair v = { a, b }; return v; }
static planarReal planar_sum(planarPair v) { return v.x + v.y; }
static planarPair planar_fma(planarPair a, planarPair b, planarPair c) { return __builtin_elementwise_fma(a, b, c); }
static planarReal planar_mac(planarReal a, planarReal b, planarReal c) { return __builtin_elementwise_fma(a, b, c); }

static void planar_grip(float friction, planarGrip *grip) {
    double mu2 = (double)friction * (double)friction;
    grip->friction = (planarReal)friction;
    for (int s = 0; s < 4; s++) {
        double pair[2];
        for (int l = 0; l < 2; l++) {
            double weight = (double)PLANAR_BODY_INVWEIGHT[PLANAR_CAPSULE_BODY[s == 0 ? l : 2 + 3 * l + (s - 1)]];
            pair[l] = fma(mu2, weight, weight) * 2.0 * mu2;
        }
        grip->pyramid[s] = planar_pair((planarReal)pair[0], (planarReal)pair[1]);
    }
}

static void planar_derive(planarModel *P) {
    double subtree[PLANAR_BODIES], moment[PLANAR_BODIES][2], inertia[PLANAR_BODIES];
    for (int k = PLANAR_BODIES - 1; k >= 0; k--) {
        double mass = (double)PLANAR_MASS[k], cx = (double)PLANAR_COM_X[k], cz = (double)PLANAR_COM_Z[k];
        double q[4] = { (double)PLANAR_FRAME_W[k], (double)PLANAR_FRAME_X[k], (double)PLANAR_FRAME_Y[k], (double)PLANAR_FRAME_Z[k] };
        double r0 = 2.0 * (q[1] * q[2] + q[0] * q[3]), r1 = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3], r2 = 2.0 * (q[2] * q[3] - q[0] * q[1]);
        subtree[k] = mass;
        moment[k][0] = mass * cx;
        moment[k][1] = mass * cz;
        inertia[k] = (double)PLANAR_INERTIA_A[k] * r0 * r0 + (double)PLANAR_INERTIA_B[k] * r1 * r1 + (double)PLANAR_INERTIA_C[k] * r2 * r2 + mass * (cx * cx + cz * cz);
        for (int c = k + 1; c < PLANAR_BODIES; c++) {
            if (PLANAR_PARENT[c] != k) continue;
            double lx = (double)PLANAR_BODY_X[c], lz = (double)PLANAR_BODY_Z[c];
            subtree[k] += subtree[c];
            moment[k][0] += subtree[c] * lx;
            moment[k][1] += subtree[c] * lz;
            inertia[k] += subtree[c] * (lx * lx + lz * lz);
        }
    }
    P->mass = (planarReal)subtree[0];
    P->torsoInertia = (planarReal)inertia[0];
    P->torsoMomentX = (planarReal)moment[0][0];
    P->torsoMomentZ = (planarReal)moment[0][1];
    P->torsoHeight = (planarReal)PLANAR_BODY_Z[0];
    for (int m = 0; m < 3; m++) {
        int back = 1 + m, front = 4 + m, ancestor = m == 2 ? 1 : 0, child = m == 0 ? 1 : 2;
        P->inertia[m] = planar_pair((planarReal)inertia[back], (planarReal)inertia[front]);
        P->momentX[m] = planar_pair((planarReal)moment[back][0], (planarReal)moment[front][0]);
        P->momentZ[m] = planar_pair((planarReal)moment[back][1], (planarReal)moment[front][1]);
        P->linkX[m] = planar_pair((planarReal)PLANAR_BODY_X[back], (planarReal)PLANAR_BODY_X[front]);
        P->linkZ[m] = planar_pair((planarReal)PLANAR_BODY_Z[back], (planarReal)PLANAR_BODY_Z[front]);
        for (int t = 0; t < 2; t++) {
            double torso[2], leg[2];
            for (int l = 0; l < 2; l++) {
                int top = 1 + 3 * l, k = top + m, n = top + child, link = top + ancestor + 1;
                double tx = (double)PLANAR_BODY_X[top], tz = (double)PLANAR_BODY_Z[top], ex = (double)PLANAR_BODY_X[link], ez = (double)PLANAR_BODY_Z[link];
                torso[l] = t ? tx * moment[k][1] - tz * moment[k][0] : tx * moment[k][0] + tz * moment[k][1];
                leg[l] = t ? ex * moment[n][1] - ez * moment[n][0] : ex * moment[n][0] + ez * moment[n][1];
            }
            P->torsoPair[m][t] = planar_pair((planarReal)torso[0], (planarReal)torso[1]);
            P->legPair[m][t] = planar_pair((planarReal)leg[0], (planarReal)leg[1]);
        }
    }
    double radius = (double)PLANAR_RADIUS;
    for (int s = 0; s < 4; s++) {
        double cx[2], cz[2], ax[2], az[2], ex[2][2], ez[2][2], reach[2];
        for (int l = 0; l < 2; l++) {
            int g = s == 0 ? l : 2 + 3 * l + (s - 1);
            double w = (double)PLANAR_CAPSULE_W[g], y = (double)PLANAR_CAPSULE_Y[g], h = (double)PLANAR_CAPSULE_HALF[g];
            cx[l] = (double)PLANAR_CAPSULE_X[g];
            cz[l] = (double)PLANAR_CAPSULE_Z[g];
            ax[l] = 2.0 * w * y;
            az[l] = w * w - y * y;
            reach[l] = radius + h;
            for (int e = 0; e < 2; e++) {
                ex[e][l] = cx[l] + (e ? -h : h) * ax[l];
                ez[e][l] = cz[l] + (e ? -h : h) * az[l];
            }
        }
        P->capsuleX[s] = planar_pair((planarReal)cx[0], (planarReal)cx[1]);
        P->capsuleZ[s] = planar_pair((planarReal)cz[0], (planarReal)cz[1]);
        P->axisX[s] = planar_pair((planarReal)ax[0], (planarReal)ax[1]);
        P->axisZ[s] = planar_pair((planarReal)az[0], (planarReal)az[1]);
        P->reach[s] = planar_pair((planarReal)reach[0], (planarReal)reach[1]);
        for (int e = 0; e < 2; e++) {
            P->sphereX[s][e] = planar_pair((planarReal)ex[e][0], (planarReal)ex[e][1]);
            P->sphereZ[s][e] = planar_pair((planarReal)ez[e][0], (planarReal)ez[e][1]);
        }
    }
    double timestep = (double)PLANAR_TIMESTEP;
    for (int m = 0; m < 3; m++) {
        P->low[m] = planar_pair((planarReal)PLANAR_RANGE_LOW[m], (planarReal)PLANAR_RANGE_LOW[3 + m]);
        P->high[m] = planar_pair((planarReal)PLANAR_RANGE_HIGH[m], (planarReal)PLANAR_RANGE_HIGH[3 + m]);
        P->spring[m] = planar_pair((planarReal)PLANAR_STIFFNESS[m], (planarReal)PLANAR_STIFFNESS[3 + m]);
        P->damper[m] = planar_pair((planarReal)PLANAR_DAMPING[m], (planarReal)PLANAR_DAMPING[3 + m]);
        P->drag[m] = planar_pair((planarReal)(timestep * (double)PLANAR_DAMPING[m]), (planarReal)(timestep * (double)PLANAR_DAMPING[3 + m]));
        P->limitWeight[m] = planar_pair((planarReal)PLANAR_JOINT_INVWEIGHT[m], (planarReal)PLANAR_JOINT_INVWEIGHT[3 + m]);
        P->still[m] = (planarPair)0;
    }
    P->armature = planar_pair((planarReal)PLANAR_ARMATURE, (planarReal)PLANAR_ARMATURE);
    double timeconst = fmax((double)PLANAR_TIMECONST, 2.0 * timestep), high = (double)PLANAR_IMPEDANCE_HIGH, ratio = (double)PLANAR_DAMPRATIO;
    P->stiffness = (planarReal)(1.0 / (high * high * timeconst * timeconst * ratio * ratio));
    P->damping = (planarReal)(2.0 / (high * timeconst));
    P->lowImpedance = (planarReal)PLANAR_IMPEDANCE_LOW;
    P->highImpedance = (planarReal)high;
    P->impedanceSpan = (planarReal)(high - (double)PLANAR_IMPEDANCE_LOW);
    P->limitScale = (planarReal)(1.0 / (double)PLANAR_LIMIT_WIDTH);
    P->contactScale = (planarReal)(1.0 / (double)PLANAR_CONTACT_WIDTH);
    P->gravity = (planarReal)PLANAR_GRAVITY;
    P->timestep = (planarReal)timestep;
    planar_grip(PLANAR_FRICTION, &P->grip);
    P->radius = (planarReal)radius;
    P->bad = (planarReal)PLANAR_BAD;
    double fw = (double)PLANAR_FRAME_W[0], fx = (double)PLANAR_FRAME_X[0], fy = (double)PLANAR_FRAME_Y[0], fz = (double)PLANAR_FRAME_Z[0];
    double a0 = 2.0 * (fx * fy + fw * fz), a1 = fw * fw - fx * fx + fy * fy - fz * fz, a2 = 2.0 * (fy * fz - fw * fx);
    double spin = (double)PLANAR_INERTIA_A[0] * a0 * a0 + (double)PLANAR_INERTIA_B[0] * a1 * a1 + (double)PLANAR_INERTIA_C[0] * a2 * a2;
    double torsoMass = (double)PLANAR_MASS[0], offset = (double)PLANAR_COM_X[0] * (double)PLANAR_COM_X[0] + (double)PLANAR_COM_Z[0] * (double)PLANAR_COM_Z[0], turn = spin + torsoMass * offset;
    double floor = fmin(fmin(torsoMass, 0.5 * (torsoMass + turn) - sqrt(0.25 * (torsoMass - turn) * (torsoMass - turn) + torsoMass * torsoMass * offset)), (double)PLANAR_ARMATURE);
    P->flightForce = (planarReal)(0.25 * (double)PLANAR_BAD * (double)PLANAR_BAD * floor * floor);
    for (int k = 0; k < 6; k++) P->sine[k] = (planarPair)(planarReal)PLANAR_SINE[k];
    for (int k = 0; k < 7; k++) P->cosine[k] = (planarPair)(planarReal)PLANAR_COSINE[k];
    planarReal half = (planarReal)1.5707963267948966;
    P->quadrant[0] = (planarPair)(planarReal)0.63661977236758134;
    P->quadrant[1] = (planarPair)half;
    P->quadrant[2] = (planarPair)(planarReal)(1.5707963267948966 - (double)half + 6.123233995736766e-17);
    for (int k = 0; k < 4; k++)
        for (int e = 0; e < 2; e++) {
            planarLane bit = { (planarWord)1 << (4 * k + 2 * e), (planarWord)1 << (4 * k + 2 * e + 1) };
            P->hitBit[k][e] = bit;
        }
    planarLane back = { -1, 0 }, front = { 0, -1 };
    P->lane[0] = back;
    P->lane[1] = front;
    for (int k = 0; k < PLANAR_GAIT_FEATURES; k++)
        for (int j = 0; j < 8; j++) ((float *)&P->gait[k][j / 4])[j % 4] = j < PLANAR_GAIT_JOINTS ? PLANAR_GAIT[PLANAR_GAIT_FEATURES * j + k] : 0.0f;
}

static planarPair planar_select(planarLane mask, planarPair yes, planarPair no) { return (planarPair)(((planarLane)yes & mask) | ((planarLane)no & ~mask)); }

static inline __attribute__((always_inline)) void planar_sincos(const planarTable *P, int count, const planarPair *x, planarPair *s, planarPair *c) {
    planarPair j[4], r[4], r2[4], sp[4], cp[4];
    for (int v = 0; v < count; v++) {
        j[v] = __builtin_elementwise_floor(planar_fma(x[v], P->quadrant[0], (planarPair)(planarReal)0.5));
        r[v] = planar_fma(-j[v], P->quadrant[2], planar_fma(-j[v], P->quadrant[1], x[v]));
        r2[v] = r[v] * r[v];
        sp[v] = P->sine[0];
        cp[v] = P->cosine[0];
    }
    for (int k = 1; k < 6; k++)
        for (int v = 0; v < count; v++) sp[v] = planar_fma(sp[v], r2[v], P->sine[k]);
    for (int k = 1; k < 7; k++)
        for (int v = 0; v < count; v++) cp[v] = planar_fma(cp[v], r2[v], P->cosine[k]);
    for (int v = 0; v < count; v++) {
        planarPair sr = planar_fma(r[v] * r2[v], sp[v], r[v]), cr = planar_fma(r2[v], cp[v], (planarPair)1);
        planarLane q = __builtin_convertvector(j[v], planarLane), odd = -(q & 1), sign = (planarLane)(planarPair)(-(planarReal)0);
        s[v] = (planarPair)((planarLane)planar_select(odd, cr, sr) ^ (((q & 2) != 0) & sign));
        c[v] = (planarPair)((planarLane)planar_select(odd, sr, cr) ^ ((((q + 1) & 2) != 0) & sign));
    }
}

static inline __attribute__((always_inline)) planarReal planar_impedance(planarReal low, planarReal high, planarReal span, planarReal x) {
    planarReal mid = (planarReal)PLANAR_IMPEDANCE_MID, y = x <= mid ? x * x / mid : 1 - (1 - x) * (1 - x) / (1 - mid);
    planarPair v = (planarPair)x, inside = (planarPair)planar_mac(y, span, low);
    return planar_select(v >= 1, (planarPair)high, planar_select(v <= 0, (planarPair)low, inside)).x;
}

static planarPair planar_dot3(planarPair a0, planarPair b0, planarPair a1, planarPair b1, planarPair a2, planarPair b2) { return planar_fma(a2, b2, planar_fma(a1, b1, a0 * b0)); }
static planarReal planar_sdot3(planarReal a0, planarReal b0, planarReal a1, planarReal b1, planarReal a2, planarReal b2) { return planar_mac(a2, b2, planar_mac(a1, b1, a0 * b0)); }

static planarReal planar_dot(const planarVec *a, const planarVec *b) {
    return planar_mac(a->r[2], b->r[2], planar_mac(a->r[1], b->r[1], planar_mac(a->r[0], b->r[0], planar_sum(planar_dot3(a->l[0], b->l[0], a->l[1], b->l[1], a->l[2], b->l[2])))));
}

static planarReal planar_entry(const planarVec *x, const planarUnit *q) { return q->tangent * ((const planarLoose *)&x->l[q->slot])[q->lane]; }

static void planar_axpy(planarVec *y, planarReal s, const planarVec *x) {
    for (int i = 0; i < 3; i++) {
        y->r[i] = planar_mac(s, x->r[i], y->r[i]);
        y->l[i] = planar_fma((planarPair)s, x->l[i], y->l[i]);
    }
}

static inline __attribute__((always_inline)) void planar_factorBody(const planarArrow *a, const planarPairTable *drag, planarFactor *f) {
    const planarPair *c = a->c;
    planarPair l[6] = { a->l[0] + drag[0], a->l[1], a->l[2], a->l[3] + drag[1], a->l[4], a->l[5] + drag[2] }, *b = f->adj, x[3][3], y[6];
    b[0] = planar_fma(-l[4], l[4], l[3] * l[5]);
    b[1] = planar_fma(-l[1], l[5], l[2] * l[4]);
    b[2] = planar_fma(-l[2], l[3], l[1] * l[4]);
    b[3] = planar_fma(-l[2], l[2], l[0] * l[5]);
    b[4] = planar_fma(-l[0], l[4], l[1] * l[2]);
    b[5] = planar_fma(-l[1], l[1], l[0] * l[3]);
    planarPair det = planar_dot3(l[0], b[0], l[1], b[1], l[2], b[2]), other = det.yx;
    for (int i = 0; i < 3; i++) {
        x[i][0] = planar_dot3(b[0], c[3 * i], b[1], c[3 * i + 1], b[2], c[3 * i + 2]);
        x[i][1] = planar_dot3(b[1], c[3 * i], b[3], c[3 * i + 1], b[4], c[3 * i + 2]);
        x[i][2] = planar_dot3(b[2], c[3 * i], b[4], c[3 * i + 1], b[5], c[3 * i + 2]);
    }
    y[0] = planar_dot3(c[0], x[0][0], c[1], x[0][1], c[2], x[0][2]);
    y[1] = planar_dot3(c[0], x[1][0], c[1], x[1][1], c[2], x[1][2]);
    y[2] = planar_dot3(c[0], x[2][0], c[1], x[2][1], c[2], x[2][2]);
    y[3] = planar_dot3(c[3], x[1][0], c[4], x[1][1], c[5], x[1][2]);
    y[4] = planar_dot3(c[3], x[2][0], c[4], x[2][1], c[5], x[2][2]);
    y[5] = planar_dot3(c[6], x[2][0], c[7], x[2][1], c[8], x[2][2]);
    planarReal k = det.x * det.y, s[6];
    for (int i = 0; i < 6; i++) s[i] = planar_mac(a->r[i], k, -planar_sum(other * y[i]));
    f->s[0] = planar_mac(-s[4], s[4], s[3] * s[5]);
    f->s[1] = planar_mac(-s[1], s[5], s[2] * s[4]);
    f->s[2] = planar_mac(-s[2], s[3], s[1] * s[4]);
    f->s[3] = planar_mac(-s[2], s[2], s[0] * s[5]);
    f->s[4] = planar_mac(-s[0], s[4], s[1] * s[2]);
    f->s[5] = planar_mac(-s[1], s[1], s[0] * s[3]);
    planarReal ds = planar_sdot3(s[0], f->s[0], s[1], f->s[1], s[2], f->s[2]), z = 1 / (k * ds);
    f->rootInv = k * k * z;
    f->legInv = (planarPair)(ds * z) * other;
}

static void planar_factor(const planarArrow *a, const planarPairTable *drag, planarFactor *f) { planar_factorBody(a, drag, f); }

static void planar_solve(const planarArrow *a, const planarFactor *f, const planarVec *g, planarVec *u) {
    const planarPair *b = f->adj, *c = a->c, *gl = g->l;
    planarPair t0 = planar_dot3(b[0], gl[0], b[1], gl[1], b[2], gl[2]), t1 = planar_dot3(b[1], gl[0], b[3], gl[1], b[4], gl[2]), t2 = planar_dot3(b[2], gl[0], b[4], gl[1], b[5], gl[2]);
    planarReal r0 = g->r[0] - planar_sum(f->legInv * planar_dot3(c[0], t0, c[1], t1, c[2], t2));
    planarReal r1 = g->r[1] - planar_sum(f->legInv * planar_dot3(c[3], t0, c[4], t1, c[5], t2));
    planarReal r2 = g->r[2] - planar_sum(f->legInv * planar_dot3(c[6], t0, c[7], t1, c[8], t2));
    planarReal u0 = f->rootInv * planar_sdot3(f->s[0], r0, f->s[1], r1, f->s[2], r2);
    planarReal u1 = f->rootInv * planar_sdot3(f->s[1], r0, f->s[3], r1, f->s[4], r2);
    planarReal u2 = f->rootInv * planar_sdot3(f->s[2], r0, f->s[4], r1, f->s[5], r2);
    planarPair w0 = gl[0] - planar_dot3(c[0], (planarPair)u0, c[3], (planarPair)u1, c[6], (planarPair)u2);
    planarPair w1 = gl[1] - planar_dot3(c[1], (planarPair)u0, c[4], (planarPair)u1, c[7], (planarPair)u2);
    planarPair w2 = gl[2] - planar_dot3(c[2], (planarPair)u0, c[5], (planarPair)u1, c[8], (planarPair)u2);
    u->r[0] = u0;
    u->r[1] = u1;
    u->r[2] = u2;
    u->l[0] = f->legInv * planar_dot3(b[0], w0, b[1], w1, b[2], w2);
    u->l[1] = f->legInv * planar_dot3(b[1], w0, b[3], w1, b[4], w2);
    u->l[2] = f->legInv * planar_dot3(b[2], w0, b[4], w1, b[5], w2);
}

static void planar_factorSolve(const planarArrow *a, const planarPairTable *drag, const planarVec *g, planarVec *u) {
    planarFactor f;
    planar_factor(a, drag, &f);
    planar_solve(a, &f, g, u);
}

static void planar_mul(const planarArrow *a, const planarVec *x, planarVec *y) {
    const planarPair *c = a->c, *l = a->l, *xl = x->l;
    planarReal x0 = x->r[0], x1 = x->r[1], x2 = x->r[2];
    y->r[0] = planar_mac(a->r[2], x2, planar_mac(a->r[1], x1, planar_mac(a->r[0], x0, planar_sum(planar_dot3(c[0], xl[0], c[1], xl[1], c[2], xl[2])))));
    y->r[1] = planar_mac(a->r[4], x2, planar_mac(a->r[3], x1, planar_mac(a->r[1], x0, planar_sum(planar_dot3(c[3], xl[0], c[4], xl[1], c[5], xl[2])))));
    y->r[2] = planar_mac(a->r[5], x2, planar_mac(a->r[4], x1, planar_mac(a->r[2], x0, planar_sum(planar_dot3(c[6], xl[0], c[7], xl[1], c[8], xl[2])))));
    y->l[0] = planar_fma(c[6], (planarPair)x2, planar_fma(c[3], (planarPair)x1, planar_fma(c[0], (planarPair)x0, planar_dot3(l[0], xl[0], l[1], xl[1], l[2], xl[2]))));
    y->l[1] = planar_fma(c[7], (planarPair)x2, planar_fma(c[4], (planarPair)x1, planar_fma(c[1], (planarPair)x0, planar_dot3(l[1], xl[0], l[3], xl[1], l[4], xl[2]))));
    y->l[2] = planar_fma(c[8], (planarPair)x2, planar_fma(c[5], (planarPair)x1, planar_fma(c[2], (planarPair)x0, planar_dot3(l[2], xl[0], l[4], xl[1], l[5], xl[2]))));
}

static inline __attribute__((always_inline)) void planar_rank(planarReal *r, planarPair *c, planarPair *l, planarVec *g, const planarVec *u, planarReal w, planarReal t) {
    planarReal d0 = w * u->r[0], d1 = w * u->r[1], d2 = w * u->r[2], wt = w * t;
    planarPair e0 = w * u->l[0], e1 = w * u->l[1], e2 = w * u->l[2];
    r[0] = planar_mac(d0, u->r[0], r[0]); r[1] = planar_mac(d0, u->r[1], r[1]); r[2] = planar_mac(d0, u->r[2], r[2]);
    r[3] = planar_mac(d1, u->r[1], r[3]); r[4] = planar_mac(d1, u->r[2], r[4]); r[5] = planar_mac(d2, u->r[2], r[5]);
    for (int k = 0; k < 3; k++) {
        c[k] = planar_fma((planarPair)d0, u->l[k], c[k]);
        c[3 + k] = planar_fma((planarPair)d1, u->l[k], c[3 + k]);
        c[6 + k] = planar_fma((planarPair)d2, u->l[k], c[6 + k]);
    }
    l[0] = planar_fma(e0, u->l[0], l[0]); l[1] = planar_fma(e0, u->l[1], l[1]); l[2] = planar_fma(e0, u->l[2], l[2]);
    l[3] = planar_fma(e1, u->l[1], l[3]); l[4] = planar_fma(e1, u->l[2], l[4]); l[5] = planar_fma(e2, u->l[2], l[5]);
    if (g) planar_axpy(g, wt, u);
}

static void planar_combine(const planarUnit *c, planarReal sign, planarVec *u) {
    for (int k = 0; k < 3; k++) {
        u->r[k] = planar_mac(sign, c->t.r[k], c->n.r[k]);
        u->l[k] = planar_fma((planarPair)sign, c->t.l[k], c->n.l[k]);
    }
}

static inline __attribute__((always_inline)) void planar_hessian(const planarTable *P, const planarForward *f, uint64_t mask, planarArrow *h, planarFactor *hf, planarVec *rhs) {
    planarReal r[6];
    planarPair c[9], l[6];
    planarVec g = f->fs, u, *gp = rhs ? &g : 0;
    for (int k = 0; k < 6; k++) { r[k] = f->m.r[k]; l[k] = f->m.l[k]; }
    for (int k = 0; k < 9; k++) c[k] = f->m.c[k];
    for (int i = 0; i < f->limits; i++)
        if (mask >> i & 1u) {
            const planarUnit *q = &f->unit[i];
            ((planarLoose *)&l[q->slot == 0 ? 0 : q->slot == 1 ? 3 : 5])[q->lane] += q->d;
            if (gp) ((planarLoose *)&g.l[q->slot])[q->lane] += q->d * q->an * q->tangent;
        }
    for (int i = f->limits, row = f->limits; i < f->units; i++, row += 3) {
        const planarUnit *q = &f->unit[i];
        unsigned bits = (unsigned)(mask >> row) & 7u;
        if (bits == 7u) {
            planar_rank(r, c, l, gp, &q->n, 4 * q->d, q->an);
            planar_rank(r, c, l, gp, &q->t, 2 * q->d * q->tangent * q->tangent, q->at);
            continue;
        }
        if (bits & 1u) {
            planar_combine(q, q->tangent, &u);
            planar_rank(r, c, l, gp, &u, q->d, planar_mac(q->tangent, q->at, q->an));
        }
        if (bits & 2u) {
            planar_combine(q, -q->tangent, &u);
            planar_rank(r, c, l, gp, &u, q->d, planar_mac(-q->tangent, q->at, q->an));
        }
        if (bits & 4u) planar_rank(r, c, l, gp, &q->n, 2 * q->d, q->an);
    }
    for (int k = 0; k < 6; k++) { h->r[k] = r[k]; h->l[k] = l[k]; }
    for (int k = 0; k < 9; k++) h->c[k] = c[k];
    if (rhs) *rhs = g;
    planar_factorBody(h, P->still, hf);
}

static inline __attribute__((always_inline)) void planar_readBits(const planarForward *f, const planarVec *x, const planarVec *y, planarReal *xn, planarReal *xt, planarReal *yn, planarReal *yt, uint64_t *mx, uint64_t *my) {
    uint64_t a = 0u, b = 0u;
    for (int i = 0; i < f->limits; i++) {
        xn[i] = planar_entry(x, &f->unit[i]) - f->unit[i].an;
        a |= (uint64_t)(xn[i] < 0) << i;
        if (y) {
            yn[i] = planar_entry(y, &f->unit[i]) - f->unit[i].an;
            b |= (uint64_t)(yn[i] < 0) << i;
        }
    }
    for (int i = f->limits, row = f->limits; i < f->units; i++, row += 3) {
        const planarUnit *q = &f->unit[i];
        xn[i] = planar_dot(&q->n, x) - q->an;
        xt[i] = planar_dot(&q->t, x) - q->at;
        planarReal sx = q->tangent * xt[i];
        a |= ((uint64_t)(xn[i] + sx < 0) | (uint64_t)(xn[i] - sx < 0) << 1 | (uint64_t)(xn[i] < 0) << 2) << row;
        if (y) {
            yn[i] = planar_dot(&q->n, y) - q->an;
            yt[i] = planar_dot(&q->t, y) - q->at;
            planarReal sy = q->tangent * yt[i];
            b |= ((uint64_t)(yn[i] + sy < 0) | (uint64_t)(yn[i] - sy < 0) << 1 | (uint64_t)(yn[i] < 0) << 2) << row;
        }
    }
    *mx = a;
    if (y) *my = b;
}

static planarReal planar_cost(const planarForward *f, const planarReal *xn, const planarReal *xt, uint64_t mask) {
    planarReal cost = 0;
    for (int i = 0; i < f->limits; i++) {
        planarReal term = f->unit[i].d * xn[i] * xn[i] / 2;
        cost += mask >> i & 1u ? term : 0;
    }
    for (int i = f->limits, row = f->limits; i < f->units; i++, row += 3) {
        unsigned bits = (unsigned)(mask >> row) & 7u;
        planarReal d = f->unit[i].d, s = f->unit[i].tangent * xt[i], j0 = xn[i] + s, j1 = xn[i] - s;
        planarReal t0 = d * j0 * j0 / 2, t1 = d * j1 * j1 / 2, t2 = 2 * d * xn[i] * xn[i] / 2;
        cost += bits & 1u ? t0 : 0;
        cost += bits & 2u ? t1 : 0;
        cost += bits & 4u ? t2 : 0;
    }
    return cost;
}

static planarReal planar_gauss(const planarForward *f, const planarVec *ma, const planarVec *qacc) {
    planarPair g = planar_dot3(ma->l[0] - f->fs.l[0], qacc->l[0] - f->smooth.l[0], ma->l[1] - f->fs.l[1], qacc->l[1] - f->smooth.l[1], ma->l[2] - f->fs.l[2], qacc->l[2] - f->smooth.l[2]);
    return planar_mac(ma->r[2] - f->fs.r[2], qacc->r[2] - f->smooth.r[2], planar_mac(ma->r[1] - f->fs.r[1], qacc->r[1] - f->smooth.r[1], planar_mac(ma->r[0] - f->fs.r[0], qacc->r[0] - f->smooth.r[0], planar_sum(g)))) / 2;
}

typedef struct { planarReal alpha, cost, slope, curve; } planarProbe;
typedef struct { int pairs, evaluations; const planarPair *jar, *jv, *t0, *t1, *t2; const planarReal *quad; } planarLine;

static inline __attribute__((always_inline)) planarProbe planar_probe(planarLine *line, planarReal alpha) {
    planarLane on = planar_fma((planarPair)alpha, line->jv[0], line->jar[0]) < 0;
    planarPair q0 = (planarPair)((planarLane)line->t0[0] & on), q1 = (planarPair)0 + (planarPair)((planarLane)line->t1[0] & on), q2 = (planarPair)((planarLane)line->t2[0] & on);
    for (int i = 1; i < line->pairs; i++) {
        on = planar_fma((planarPair)alpha, line->jv[i], line->jar[i]) < 0;
        q0 += (planarPair)((planarLane)line->t0[i] & on);
        q1 += (planarPair)((planarLane)line->t1[i] & on);
        q2 += (planarPair)((planarLane)line->t2[i] & on);
    }
    planarReal c0 = line->quad[0] + planar_sum(q0), c1 = line->quad[1] + planar_sum(q1), c2 = line->quad[2] + planar_sum(q2);
    planarProbe p = { alpha, c2 * alpha * alpha + c1 * alpha + c0, 2 * c2 * alpha + c1, 2 * c2 };
    if (p.curve <= 0) p.curve = (planarReal)PLANAR_MINVAL;
    line->evaluations++;
    return p;
}

static inline __attribute__((always_inline)) int planar_bracket(planarLine *line, planarProbe *p, const planarProbe *candidate, planarProbe *next) {
    int moved = 0;
    for (int k = 0; k < 3; k++)
        if ((p->slope < 0 && candidate[k].slope < 0 && candidate[k].slope > p->slope) || (p->slope > 0 && candidate[k].slope > 0 && candidate[k].slope < p->slope)) {
            *p = candidate[k];
            moved = 1;
        }
    if (moved) *next = planar_probe(line, p->alpha - p->slope / p->curve);
    return moved;
}

static inline __attribute__((always_inline)) planarReal planar_search(planarLine *line, planarReal norm, planarReal scale, planarReal precision, int evaluations) {
    if (norm < (planarReal)PLANAR_MINVAL) return 0;
    planarReal tolerance = precision * norm / scale;
    planarProbe p0 = planar_probe(line, 0), p1 = planar_probe(line, p0.alpha - p0.slope / p0.curve);
    if (__builtin_elementwise_abs(p1.slope) < tolerance) return p1.alpha;
    planarReal direction = p1.slope < 0 ? 1 : -1;
    planarProbe p2 = p0;
    while (p1.slope * direction <= -tolerance && line->evaluations < evaluations) {
        p2 = p1;
        p1 = planar_probe(line, p1.alpha - p1.slope / p1.curve);
        if (__builtin_elementwise_abs(p1.slope) < tolerance) return p1.alpha;
    }
    if (line->evaluations >= evaluations) return p1.alpha;
    planarProbe p2next = p1, p1next = planar_probe(line, p1.alpha - p1.slope / p1.curve);
    while (line->evaluations < evaluations) {
        planarProbe middle = planar_probe(line, (p1.alpha + p2.alpha) / 2);
        planarProbe candidate[3] = { p1next, p2next, middle }, best = middle;
        int found = 0;
        for (int k = 0; k < 3; k++)
            if (__builtin_elementwise_abs(candidate[k].slope) < tolerance && (!found || candidate[k].cost < best.cost)) {
                best = candidate[k];
                found = 1;
            }
        if (found) return best.alpha;
        int moved1 = planar_bracket(line, &p1, candidate, &p1next);
        int moved2 = planar_bracket(line, &p2, candidate, &p2next);
        if (!moved1 && !moved2) return middle.alpha;
    }
    if (p1.cost <= p2.cost && p1.cost < p0.cost) return p1.alpha;
    if (p2.cost <= p1.cost && p2.cost < p0.cost) return p2.alpha;
    return 0;
}

static void planar_expand(const planarForward *f, const planarReal *xn, const planarReal *xt, planarReal pad, planarPair *pairs) {
    planarLoose *row = (planarLoose *)pairs;
    int r = 0;
    for (int i = 0; i < f->units; i++) {
        if (i < f->limits) {
            row[r++] = xn[i];
            continue;
        }
        planarReal s = f->unit[i].tangent * xt[i];
        row[r] = xn[i] + s;
        row[r + 1] = xn[i] - s;
        row[r + 2] = xn[i];
        r += 3;
    }
    if (r & 1) row[r] = pad;
}

static uint64_t planar_pairMask(const planarPair *jar, int pairs) {
    uint64_t mask = 0u;
    for (int i = 0; i < pairs; i++) {
        planarLane on = jar[i] < 0;
        mask |= ((uint64_t)(on.x & 1) | (uint64_t)(on.y & 1) << 1) << (2 * i);
    }
    return mask;
}

static void planar_flip(const planarForward *f, planarArrow *h, uint64_t added, uint64_t removed) {
    if (!((added | removed) >> f->limits)) {
        for (uint64_t rest = added | removed; rest; rest &= rest - 1u) {
            int row = __builtin_ctzll(rest);
            const planarUnit *q = &f->unit[row];
            ((planarLoose *)&h->l[q->slot == 0 ? 0 : q->slot == 1 ? 3 : 5])[q->lane] += (added >> row & 1u ? 1 : -1) * q->d;
        }
        return;
    }
    planarReal r[6];
    planarPair c[9], l[6];
    planarVec u;
    for (int k = 0; k < 6; k++) { r[k] = h->r[k]; l[k] = h->l[k]; }
    for (int k = 0; k < 9; k++) c[k] = h->c[k];
    for (uint64_t rest = added | removed; rest; rest &= rest - 1u) {
        int row = __builtin_ctzll(rest);
        planarReal sign = added >> row & 1u ? 1 : -1;
        if (row < f->limits) {
            const planarUnit *q = &f->unit[row];
            ((planarLoose *)&l[q->slot == 0 ? 0 : q->slot == 1 ? 3 : 5])[q->lane] += sign * q->d;
            continue;
        }
        int i = f->limits + (row - f->limits) / 3, t = (row - f->limits) % 3;
        const planarUnit *q = &f->unit[i];
        if (t == 2) {
            planar_rank(r, c, l, 0, &q->n, sign * 2 * q->d, 0);
            continue;
        }
        planar_combine(q, t ? -q->tangent : q->tangent, &u);
        planar_rank(r, c, l, 0, &u, sign * q->d, 0);
    }
    for (int k = 0; k < 6; k++) { h->r[k] = r[k]; h->l[k] = l[k]; }
    for (int k = 0; k < 9; k++) h->c[k] = c[k];
}

static inline __attribute__((always_inline)) void planar_rowForce(const planarForward *f, const planarLoose *weight, const planarPair *jar, uint64_t mask, planarVec *fc) {
    const planarLoose *row = (const planarLoose *)jar;
    memset(fc, 0, sizeof *fc);
    for (int i = 0; i < f->limits; i++)
        if (mask >> i & 1u) ((planarLoose *)&fc->l[f->unit[i].slot])[f->unit[i].lane] -= weight[i] * row[i] * f->unit[i].tangent;
    for (int i = f->limits, r = f->limits; i < f->units; i++, r += 3) {
        unsigned bits = (unsigned)(mask >> r) & 7u;
        if (!bits) continue;
        const planarUnit *q = &f->unit[i];
        planarReal a = bits & 1u ? -weight[r] * row[r] : 0, b = bits & 2u ? -weight[r + 1] * row[r + 1] : 0, n = bits & 4u ? -weight[r + 2] * row[r + 2] : 0;
        planar_axpy(fc, a + b + n, &q->n);
        planar_axpy(fc, q->tangent * (a - b), &q->t);
    }
}

__attribute__((noinline)) static void planar_newton(const planarTable *P, planarForward *f, planarArrow *h, planarFactor *hf, const planarVec *target, const planarReal *tn, const planarReal *tt, const planarReal *xn, const planarReal *xt, uint64_t mask, planarVec ma, planarReal gauss, planarReal cost) {
    planarVec mv, grad, dir, force, *qacc = &f->qacc;
    planarPair jar[PLANAR_ROWS / 2 + 1], jv[PLANAR_ROWS / 2 + 1], weight[PLANAR_ROWS / 2 + 1], t0[PLANAR_ROWS / 2 + 1], t1[PLANAR_ROWS / 2 + 1], t2[PLANAR_ROWS / 2 + 1];
    planarReal dn[PLANAR_UNITS], dt[PLANAR_UNITS];
    planarLoose *w = (planarLoose *)weight;
    int pairs = (f->rows + 1) / 2;
    const planarArrow *m = &f->m;
    planarReal scale = 1 / (m->r[0] + m->r[3] + m->r[5] + planar_sum(m->l[0] + m->l[3] + m->l[5]));
    planar_expand(f, xn, xt, 1, jar);
    for (int i = 0, r = 0; i < f->units; i++) {
        if (i < f->limits) { w[r++] = f->unit[i].d; continue; }
        w[r] = f->unit[i].d;
        w[r + 1] = f->unit[i].d;
        w[r + 2] = 2 * f->unit[i].d;
        r += 3;
    }
    if (f->rows & 1) w[f->rows] = 0;
    for (int i = 0; i < 3; i++) {
        dir.r[i] = target->r[i] - qacc->r[i];
        dir.l[i] = target->l[i] - qacc->l[i];
    }
    for (int i = 0; i < f->units; i++) {
        dn[i] = tn[i] - xn[i];
        dt[i] = i < f->limits ? 0 : tt[i] - xt[i];
    }
    int moved = 0;
    for (int iteration = 0; iteration < PLANAR_ITERATIONS; iteration++) {
        planar_mul(m, &dir, &mv);
        planarReal quad[3] = { gauss, planar_dot(&dir, &ma) - planar_dot(&f->fs, &dir), planar_dot(&dir, &mv) / 2 };
        if (iteration) {
            for (int i = 0; i < f->limits; i++) dn[i] = planar_entry(&dir, &f->unit[i]);
            for (int i = f->limits; i < f->units; i++) {
                dn[i] = planar_dot(&f->unit[i].n, &dir);
                dt[i] = planar_dot(&f->unit[i].t, &dir);
            }
        }
        planar_expand(f, dn, dt, 0, jv);
        for (int i = 0; i < pairs; i++) {
            planarPair dj = weight[i] * jar[i], dv = weight[i] * jv[i];
            t0[i] = dj * jar[i] / 2;
            t1[i] = dj * jv[i];
            t2[i] = dv * jv[i] / 2;
        }
        planarLine line = { pairs, 0, jar, jv, t0, t1, t2, quad };
        planarReal alpha = planar_search(&line, __builtin_elementwise_sqrt(planar_dot(&dir, &dir)), scale, (planarReal)PLANAR_TOLERANCE * (planarReal)PLANAR_SEARCH_TOLERANCE, PLANAR_SEARCH_EVALUATIONS);
        if (alpha == 0) break;
        planar_axpy(qacc, alpha, &dir);
        planar_axpy(&ma, alpha, &mv);
        planarPair rowCost = 0;
        for (int i = 0; i < pairs; i++) {
            jar[i] = planar_fma((planarPair)alpha, jv[i], jar[i]);
            rowCost += (planarPair)((planarLane)(weight[i] * jar[i] * jar[i] / 2) & (jar[i] < 0));
        }
        uint64_t next = planar_pairMask(jar, pairs);
        planarReal previous = cost;
        planar_rowForce(f, w, jar, next, &force);
        moved = 1;
        gauss = planar_gauss(f, &ma, qacc);
        cost = planar_sum(rowCost) + gauss;
        for (int i = 0; i < 3; i++) {
            grad.r[i] = ma.r[i] - f->fs.r[i] - force.r[i];
            grad.l[i] = ma.l[i] - f->fs.l[i] - force.l[i];
        }
        uint64_t added = next & ~mask, removed = mask & ~next;
        mask = next;
        if (scale * (previous - cost) < (planarReal)PLANAR_TOLERANCE || scale * __builtin_elementwise_sqrt(planar_dot(&grad, &grad)) < (planarReal)PLANAR_TOLERANCE) break;
        if (added | removed) {
            planar_flip(f, h, added, removed);
            planar_factorBody(h, P->still, hf);
        }
        planar_solve(h, hf, &grad, &dir);
        for (int i = 0; i < 3; i++) {
            dir.r[i] = -dir.r[i];
            dir.l[i] = -dir.l[i];
        }
    }
    if (!moved) planar_rowForce(f, w, jar, mask, &force);
    f->force = f->fs;
    planar_axpy(&f->force, 1, &force);
    f->active = mask;
}

__attribute__((noinline)) static void planar_dynamics(const planarTable *P, const planarGripTable *grip, const planarState *s, const planarVec *tau, planarForward *f) {
    planarReal theta = s->q[2], v0 = s->v[0], v1 = s->v[1], omega = s->v[2], p0z = P->torsoHeight + s->q[1];
    planarPair q[3], v[3], phi[4], w[3], w2[3], trig[2][4], *sn = trig[0] + 1, *cs = trig[1] + 1, px[3], pz[3];
    phi[0] = (planarPair)theta;
    for (int m = 0; m < 3; m++) {
        q[m] = s->ql[m];
        v[m] = s->vl[m];
        phi[m + 1] = phi[m] + q[m];
        w[m] = (m ? w[m - 1] : (planarPair)omega) + v[m];
        w2[m] = w[m] * w[m];
    }
    planar_sincos(P, 4, phi, trig[0], trig[1]);
    planarReal s0 = trig[0][0].x, c0 = trig[1][0].x, w0 = omega * omega;
    for (int m = 0; m < 3; m++) {
        planarPair rs = m ? sn[m - 1] : (planarPair)s0, rc = m ? cs[m - 1] : (planarPair)c0;
        px[m] = planar_fma(rs, P->linkZ[m], planar_fma(rc, P->linkX[m], m ? px[m - 1] : (planarPair)0));
        pz[m] = planar_fma(rc, P->linkZ[m], planar_fma(-rs, P->linkX[m], m ? pz[m - 1] : (planarPair)p0z));
    }
    planarReal gx0 = planar_mac(-s0, P->torsoMomentX, c0 * P->torsoMomentZ), gz0 = planar_mac(-s0, P->torsoMomentZ, -c0 * P->torsoMomentX), bias0 = P->gravity * gz0;
    planarPair gx[3], gz[3], bias[3], torso[3], pair[3], bxs = 0, bzs = 0;
    for (int m = 0; m < 3; m++) {
        gx[m] = planar_fma(-sn[m], P->momentX[m], cs[m] * P->momentZ[m]);
        gz[m] = planar_fma(-sn[m], P->momentZ[m], -cs[m] * P->momentX[m]);
        bxs = planar_fma(gz[m], w2[m], bxs);
        bzs = planar_fma(gx[m], w2[m], bzs);
        planarPair cd = planar_fma(sn[m], (planarPair)s0, cs[m] * c0), sd = planar_fma(-cs[m], (planarPair)s0, sn[m] * c0);
        planarPair slope = planar_fma(-P->torsoPair[m][0], sd, P->torsoPair[m][1] * cd);
        torso[m] = planar_fma(P->torsoPair[m][1], sd, P->torsoPair[m][0] * cd);
        bias[m] = planar_fma(-slope, (planarPair)w0, P->gravity * gz[m]);
        bias0 += planar_sum(slope * w2[m]);
    }
    for (int k = 0; k < 3; k++) {
        int a = k == 2 ? 1 : 0, n = k == 0 ? 1 : 2;
        planarPair cd = planar_fma(sn[n], sn[a], cs[n] * cs[a]), sd = planar_fma(-cs[n], sn[a], sn[n] * cs[a]);
        planarPair slope = planar_fma(-P->legPair[k][0], sd, P->legPair[k][1] * cd);
        pair[k] = planar_fma(P->legPair[k][1], sd, P->legPair[k][0] * cd);
        bias[n] = planar_fma(-slope, w2[a], bias[n]);
        bias[a] = planar_fma(slope, w2[n], bias[a]);
    }
    planarArrow *m = &f->m;
    planarPair l22 = P->inertia[2], l12 = pair[2] + l22, l11 = P->inertia[1] + pair[2] + l12, l02 = pair[1] + l12;
    planarPair l01 = pair[0] + pair[1] + l11, l00 = P->inertia[0] + pair[0] + pair[1] + l01;
    planarPair sp2 = torso[2], sp1 = torso[1] + sp2, sp0 = torso[0] + sp1;
    m->c[2] = gx[2]; m->c[1] = gx[1] + m->c[2]; m->c[0] = gx[0] + m->c[1];
    m->c[5] = gz[2]; m->c[4] = gz[1] + m->c[5]; m->c[3] = gz[0] + m->c[4];
    m->c[6] = sp0 + l00; m->c[7] = sp1 + l01; m->c[8] = sp2 + l02;
    m->l[0] = l00 + P->armature; m->l[1] = l01; m->l[2] = l02;
    m->l[3] = l11 + P->armature; m->l[4] = l12; m->l[5] = l22 + P->armature;
    m->r[0] = P->mass; m->r[1] = 0; m->r[2] = gx0 + planar_sum(m->c[0]);
    m->r[3] = P->mass; m->r[4] = gz0 + planar_sum(m->c[3]); m->r[5] = P->torsoInertia + planar_sum(2 * sp0 + l00);
    planarPair jb2 = bias[2], jb1 = bias[1] + jb2, jb0 = bias[0] + jb1;
    f->fs.r[0] = -planar_mac(gz0, w0, planar_sum(bxs));
    f->fs.r[1] = planar_mac(gx0, w0, planar_sum(bzs)) - P->mass * P->gravity;
    f->fs.r[2] = -(bias0 + planar_sum(jb0));
    f->fs.l[0] = planar_fma(-P->damper[0], v[0], planar_fma(-P->spring[0], q[0], tau->l[0] - jb0));
    f->fs.l[1] = planar_fma(-P->damper[1], v[1], planar_fma(-P->spring[1], q[1], tau->l[1] - jb1));
    f->fs.l[2] = planar_fma(-P->damper[2], v[2], planar_fma(-P->spring[2], q[2], tau->l[2] - jb2));
    f->units = 0;
    f->limits = 0;
    f->torso = 0;
    planarPair low[3], high[3];
    planarLane limited = 0;
    for (int k = 0; k < 3; k++) {
        low[k] = q[k] - P->low[k];
        high[k] = P->high[k] - q[k];
        limited |= (low[k] < 0) | (high[k] < 0);
    }
    uint32_t violated = 0u;
    if (limited.x | limited.y)
        for (int k = 0; k < 3; k++) {
            planarLane under = low[k] < 0, over = high[k] < 0;
            violated |= (uint32_t)(under.x & 1) << (2 * k) | (uint32_t)(over.x & 1) << (2 * k + 1) | (uint32_t)(under.y & 1) << (2 * k + 6) | (uint32_t)(over.y & 1) << (2 * k + 7);
        }
    while (violated) {
        int bit = __builtin_ctz(violated), j = bit >> 1, side = bit & 1, lane = j / 3, level = j % 3;
        violated &= violated - 1u;
        planarReal dist = side ? high[level][lane] : low[level][lane];
        planarUnit *u = &f->unit[f->units];
        planarReal sign = side ? -1 : 1, imp = planar_impedance(P->lowImpedance, P->highImpedance, P->impedanceSpan, -dist * P->limitScale);
        u->tangent = sign;
        u->slot = level;
        u->lane = lane;
        u->an = planar_mac(-P->damping, sign * v[level][lane], -P->stiffness * imp * dist);
        u->d = imp / ((1 - imp) * P->limitWeight[level][lane]);
        u->at = 0;
        f->units++;
    }
    f->limits = f->units;
    planarPair centre[4], bz[4];
    planarLane near[4], any = 0;
    for (int k = 0; k < 4; k++) {
        bz[k] = k ? pz[k - 1] : (planarPair)p0z;
        centre[k] = planar_fma(trig[1][k], P->capsuleZ[k], planar_fma(-trig[0][k], P->capsuleX[k], bz[k]));
        near[k] = centre[k] <= P->reach[k];
        any |= near[k];
    }
    if (any.x | any.y) {
        planarPair sphereZ[4][2];
        planarLane bits = 0;
        for (int k = 0; k < 4; k++)
            for (int e = 0; e < 2; e++) {
                sphereZ[k][e] = planar_fma(trig[1][k], P->sphereZ[k][e], planar_fma(-trig[0][k], P->sphereX[k][e], bz[k]));
                bits |= near[k] & (sphereZ[k][e] - P->radius < 0) & P->hitBit[k][e];
            }
        uint32_t hits = (uint32_t)(bits.x | bits.y);
        while (hits) {
            int bit = __builtin_ctz(hits), k = bit >> 2, e = (bit >> 1) & 1, lane = bit & 1;
            hits &= hits - 1u;
            planarReal bs = trig[0][k][lane], bc = trig[1][k][lane], bx = k ? px[k - 1][lane] : 0;
            planarReal sx = planar_mac(bs, P->sphereZ[k][e][lane], planar_mac(bc, P->sphereX[k][e][lane], bx)), sz = sphereZ[k][e][lane], dist = sz - P->radius;
            planarReal imp = planar_impedance(P->lowImpedance, P->highImpedance, P->impedanceSpan, -dist * P->contactScale), ax = planar_mac(bs, P->axisZ[k][lane], bc * P->axisX[k][lane]);
            planarReal cz = planar_mac(-(planarReal)0.5, dist, sz) - P->radius;
            planarUnit *u = &f->unit[f->units];
            u->tangent = ax < 0 && -ax >= (planarReal)PLANAR_MINVAL ? -grip->friction : grip->friction;
            u->n.r[0] = 0;
            u->n.r[1] = 1;
            u->n.r[2] = -sx;
            u->t.r[0] = 1;
            u->t.r[1] = 0;
            u->t.r[2] = cz - p0z;
            for (int a = 0; a < 3; a++) {
                planarLane keep = a < k ? P->lane[lane] : (planarLane)0;
                u->n.l[a] = (planarPair)((planarLane)(px[a] - sx) & keep);
                u->t.l[a] = (planarPair)((planarLane)(cz - pz[a]) & keep);
            }
            planarReal vn = planar_mac(u->n.r[2], omega, v1 + planar_sum(planar_dot3(u->n.l[0], v[0], u->n.l[1], v[1], u->n.l[2], v[2])));
            planarReal vt = planar_mac(u->t.r[2], omega, v0 + planar_sum(planar_dot3(u->t.l[0], v[0], u->t.l[1], v[1], u->t.l[2], v[2])));
            u->an = planar_mac(-P->damping, vn, -P->stiffness * imp * dist);
            u->at = -P->damping * vt;
            u->d = imp / ((1 - imp) * grip->pyramid[k][lane]);
            f->units++;
            f->torso |= k == 0;
        }
    }
    f->rows = f->limits + 3 * (f->units - f->limits);
}

static inline __attribute__((always_inline)) int planar_target(const planarTable *P, const planarForward *f, uint64_t mask, planarArrow *h, planarFactor *hf, planarVec *target, planarReal *tn, planarReal *tt, uint64_t *reached) {
    planarVec rhs;
    if (mask >> f->limits) {
        planar_hessian(P, f, mask, h, hf, &rhs);
        planar_solve(h, hf, &rhs, target);
        planar_readBits(f, target, 0, tn, tt, 0, 0, reached, 0);
        return 1;
    }
    planarPair drag[3] = { 0, 0, 0 };
    rhs = f->fs;
    for (int i = 0; i < f->limits; i++)
        if (mask >> i & 1u) {
            const planarUnit *q = &f->unit[i];
            ((planarLoose *)&drag[q->slot])[q->lane] = q->d;
            ((planarLoose *)&rhs.l[q->slot])[q->lane] += q->d * q->an * q->tangent;
        }
    planar_factor(&f->m, drag, hf);
    planar_solve(&f->m, hf, &rhs, target);
    planar_readBits(f, target, 0, tn, tt, 0, 0, reached, 0);
    return 0;
}

__attribute__((noinline)) static void planar_constrained(const planarTable *P, const planarState *s, planarForward *f, planarVec *acc) {
    planar_factor(&f->m, P->still, &f->mf);
    planar_solve(&f->m, &f->mf, &f->fs, &f->smooth);
    planarVec warm, mw, target;
    planarArrow h;
    planarFactor hf;
    for (int i = 0; i < 3; i++) {
        warm.r[i] = s->w[i];
        warm.l[i] = s->wl[i];
    }
    planarReal wn[PLANAR_UNITS], wt[PLANAR_UNITS], sn[PLANAR_UNITS], st[PLANAR_UNITS], tn[PLANAR_UNITS], tt[PLANAR_UNITS], warmGauss = 0, warmCost = 0, smoothCost = 0;
    uint64_t warmMask, smoothMask, reached = 0;
    planar_readBits(f, &warm, 0, wn, wt, 0, 0, &warmMask, 0);
    int built = 0;
    if (warmMask) built = planar_target(P, f, warmMask, &h, &hf, &target, tn, tt, &reached);
    planar_readBits(f, &f->smooth, 0, sn, st, 0, 0, &smoothMask, 0);
    int smooth = -1;
    if (warmMask != smoothMask) {
        planar_mul(&f->m, &warm, &mw);
        warmGauss = planar_gauss(f, &mw, &warm);
        warmCost = planar_cost(f, wn, wt, warmMask) + warmGauss;
        smoothCost = planar_cost(f, sn, st, smoothMask);
        smooth = warmCost > smoothCost;
    }
    uint64_t mask = smooth == 1 ? smoothMask : warmMask;
    const planarReal *tnp = tn, *ttp = tt;
    if (!mask) {
        built = 0;
        hf = f->mf;
        target = f->smooth;
        tnp = sn;
        ttp = st;
        reached = smoothMask;
    } else if (mask != warmMask) built = planar_target(P, f, mask, &h, &hf, &target, tn, tt, &reached);
    if (reached == mask) {
        f->active = mask;
        f->path = 1;
        f->qacc = target;
        if (mask) planar_mul(&f->m, &target, &f->force);
        else f->force = f->fs;
        planar_solve(&f->m, &f->df, &f->force, acc);
        return;
    }
    f->path = 2;
    if (!built) {
        h = f->m;
        for (int i = 0; i < f->limits; i++)
            if (mask >> i & 1u) {
                const planarUnit *q = &f->unit[i];
                ((planarLoose *)&h.l[q->slot == 0 ? 0 : q->slot == 1 ? 3 : 5])[q->lane] += q->d;
            }
    }
    if (smooth < 0) {
        planar_mul(&f->m, &warm, &mw);
        warmGauss = planar_gauss(f, &mw, &warm);
        warmCost = planar_cost(f, wn, wt, warmMask) + warmGauss;
        smoothCost = planar_cost(f, sn, st, smoothMask);
        smooth = warmCost > smoothCost;
    }
    if (smooth) {
        f->qacc = f->smooth;
        planar_mul(&f->m, &f->smooth, &mw);
        planar_newton(P, f, &h, &hf, &target, tnp, ttp, sn, st, smoothMask, mw, planar_gauss(f, &mw, &f->smooth), smoothCost);
    } else {
        f->qacc = warm;
        planar_newton(P, f, &h, &hf, &target, tnp, ttp, wn, wt, warmMask, mw, warmGauss, warmCost);
    }
    planar_solve(&f->m, &f->df, &f->force, acc);
}

static int planar_badState(const planarState *s) {
    planarPair limit = (planarPair)(planarReal)PLANAR_BAD;
    planarLane fine = (__builtin_elementwise_abs(s->ql[0]) <= limit) & (__builtin_elementwise_abs(s->ql[1]) <= limit) & (__builtin_elementwise_abs(s->ql[2]) <= limit) & (__builtin_elementwise_abs(s->vl[0]) <= limit) & (__builtin_elementwise_abs(s->vl[1]) <= limit) & (__builtin_elementwise_abs(s->vl[2]) <= limit);
    fine &= (__builtin_elementwise_abs(planar_pair(s->q[1], s->q[2])) <= limit) & (__builtin_elementwise_abs(planar_pair(s->v[0], s->v[1])) <= limit) & (__builtin_elementwise_abs(planar_pair(s->v[2], s->q[0])) <= limit);
    return !(fine.x & fine.y & 1);
}

static int planar_badVec(const planarTable *P, const planarVec *a) {
    planarPair limit = (planarPair)P->bad;
    planarLane fine = (__builtin_elementwise_abs(a->l[0]) <= limit) & (__builtin_elementwise_abs(a->l[1]) <= limit) & (__builtin_elementwise_abs(a->l[2]) <= limit);
    return !((fine.x & fine.y & 1) & (__builtin_elementwise_abs(a->r[0]) <= P->bad) & (__builtin_elementwise_abs(a->r[1]) <= P->bad) & (__builtin_elementwise_abs(a->r[2]) <= P->bad));
}

static void planar_clear(planarState *s, planarVec *tau) {
    memset(s, 0, sizeof *s);
    memset(tau, 0, sizeof *tau);
}

static planarPair planar_round(planarPair x) { return __builtin_convertvector(__builtin_convertvector(x, planarHalf), planarPair); }

static void planar_warm(const planarTable *P, const planarForward *f, planarState *s) {
    planarVec a;
    planar_factorSolve(&f->m, P->still, &f->fs, &a);
    for (int i = 0; i < 3; i++) {
        s->w[i] = (planarReal)(float)a.r[i];
        s->wl[i] = planar_round(a.l[i]);
    }
}

static int planar_flightBad(const planarTable *P, planarForward *f) {
    if (planar_dot(&f->fs, &f->fs) <= P->flightForce) return 0;
    planar_factor(&f->m, P->still, &f->mf);
    planar_solve(&f->m, &f->mf, &f->fs, &f->smooth);
    return planar_badVec(P, &f->smooth);
}

static inline __attribute__((always_inline)) int planar_substep(const planarTable *P, const planarGripTable *grip, planarState *s, planarVec *tau, double *travel, planarForward *f, planarForward *previous, int *pending, int *bad, int last) {
    int cleared = *bad, fault = 0;
    if (cleared) {
        planar_clear(s, tau);
        *pending = 0;
    }
    planarVec acc;
    for (int attempt = 0; attempt < 2; attempt++) {
        planar_dynamics(P, grip, s, tau, f);
        if (f->units) {
            if (*pending) planar_warm(P, previous, s);
            *pending = 0;
            planar_factor(&f->m, P->drag, &f->df);
            planar_constrained(P, s, f, &acc);
            fault = planar_badVec(P, &f->qacc);
        } else {
            f->path = 0;
            f->active = 0u;
            fault = planar_flightBad(P, f);
            planar_factorSolve(&f->m, P->drag, &f->fs, &acc);
        }
        if (!fault || attempt) break;
        cleared = 1;
        planar_clear(s, tau);
        *pending = 0;
    }
    if (cleared) *travel = 0.0;
    planarReal h = P->timestep;
    for (int i = 0; i < 3; i++) {
        planarReal vr = planar_mac(h, acc.r[i], s->v[i]), qr = planar_mac(h, vr, s->q[i]);
        planarPair vl = planar_fma((planarPair)h, acc.l[i], s->vl[i]), ql = planar_fma((planarPair)h, vl, s->ql[i]);
        s->v[i] = (planarReal)(float)vr;
        s->q[i] = (planarReal)(float)qr;
        s->vl[i] = planar_round(vl);
        s->ql[i] = planar_round(ql);
        if (i == 0) {
            *travel += (double)qr;
            s->q[0] = 0;
        }
    }
    if (f->units)
        for (int i = 0; i < 3; i++) {
            s->w[i] = (planarReal)(float)f->qacc.r[i];
            s->wl[i] = planar_round(f->qacc.l[i]);
        }
    else *pending = 1;
    if (!last) *bad = planar_badState(s);
    return f->torso | cleared << 1;
}

static void planar_load(const uint32_t *words, planarState *s) {
    memcpy(&s->x, words, sizeof s->x);
    s->q[0] = 0;
    s->q[1] = (planarReal)wordFloat(words[2]);
    s->q[2] = (planarReal)wordFloat(words[3]);
    for (int i = 0; i < 3; i++) {
        s->ql[i] = planar_pair((planarReal)wordFloat(words[4 + i]), (planarReal)wordFloat(words[7 + i]));
        s->v[i] = (planarReal)wordFloat(words[10 + i]);
        s->vl[i] = planar_pair((planarReal)wordFloat(words[13 + i]), (planarReal)wordFloat(words[16 + i]));
        s->w[i] = (planarReal)wordFloat(words[19 + i]);
        s->wl[i] = planar_pair((planarReal)wordFloat(words[22 + i]), (planarReal)wordFloat(words[25 + i]));
    }
}

static void planar_store(const planarState *s, uint32_t *words) {
    memcpy(words, &s->x, sizeof s->x);
    words[2] = floatWord((float)s->q[1]);
    words[3] = floatWord((float)s->q[2]);
    for (int i = 0; i < 3; i++) {
        words[4 + i] = floatWord((float)s->ql[i].x);
        words[7 + i] = floatWord((float)s->ql[i].y);
        words[10 + i] = floatWord((float)s->v[i]);
        words[13 + i] = floatWord((float)s->vl[i].x);
        words[16 + i] = floatWord((float)s->vl[i].y);
        words[19 + i] = floatWord((float)s->w[i]);
        words[22 + i] = floatWord((float)s->wl[i].x);
        words[25 + i] = floatWord((float)s->wl[i].y);
    }
}

static void planar_draw(uint32_t seed, uint32_t *words) {
    uint32_t resets = words[28];
    float q[12], v[12];
    for (uint32_t b = 0u; b < 3u; b++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, resets, b, 0u, 0u);
        q[4 * b] = uniformRange(r.x, -PLANAR_RESET_NOISE, PLANAR_RESET_NOISE);
        q[4 * b + 1] = uniformRange(r.y, -PLANAR_RESET_NOISE, PLANAR_RESET_NOISE);
        q[4 * b + 2] = uniformRange(r.z, -PLANAR_RESET_NOISE, PLANAR_RESET_NOISE);
        q[4 * b + 3] = uniformRange(r.w, -PLANAR_RESET_NOISE, PLANAR_RESET_NOISE);
        u4 n = philox4x32(seed, RNG_STREAM_RESET, resets, 8u + b, 0u, 0u);
        uint32_t pair[4] = { n.x, n.y, n.z, n.w };
        for (int h = 0; h < 2; h++) {
            float radius = sqrtf(-2.0f * oak_log(u01open(pair[2 * h]))), sn, cs;
            oak_sincos((2.0f * OAK_PI) * u01(pair[2 * h + 1]), &sn, &cs);
            v[4 * b + 2 * h] = PLANAR_RESET_NOISE * (radius * cs);
            v[4 * b + 2 * h + 1] = PLANAR_RESET_NOISE * (radius * sn);
        }
    }
    double x = (double)(0.0f + q[0]);
    memcpy(words, &x, sizeof x);
    for (int j = 1; j < PLANAR_DOFS; j++) words[1 + j] = floatWord(0.0f + q[j]);
    for (int j = 0; j < PLANAR_DOFS; j++) {
        words[10 + j] = floatWord(v[j]);
        words[19 + j] = floatWord(0.0f);
    }
    words[28] = resets + 1u;
}

static void planar_obs(const uint32_t *words, float *obs) {
    for (int j = 0; j < 8; j++) obs[j] = wordFloat(words[2 + j]);
    for (int j = 0; j < PLANAR_DOFS; j++) obs[8 + j] = wordFloat(words[10 + j]);
}

enum { PLANAR_ROBOT_CHEETAH, PLANAR_ROBOT_REACHER, PLANAR_ROBOT_PENDULUM, PLANAR_ROBOT_HOPPER };
enum { PLANAR_FREEDOM = 6, PLANAR_BONDS = 14, PLANAR_LINES = 32, PLANAR_NEWTON_ITERATIONS = 100, PLANAR_NEWTON_EVALUATIONS = 50 };

typedef struct { double tolerance, search, timeconst, low, high, width; } planarSolver;

static const planarSolver PLANAR_SOLVER = { 1e-8, 0.01, 0.02, 0.9, 0.95, 0.001 };

typedef struct { double mass[3], spin[3], center[3], link, target, range, weight, ref[2], step, gear, armature, damping; } planarArm;

static const planarArm PLANAR_ARM = {
    { 0.03560471674068433, 0.03560471674068433, 0.004188790204786391 },
    { 3.9175660390264734e-05, 3.9175660390264734e-05, 1.6755160819145565e-07 },
    { 0.05, 0.05, 0.11 },
    0.1, 0.0030536280592892784, 3.0, 0.9998211549602818, { 0.1, -0.1 }, 0.01, 200.0, 1.0, 1.0
};

typedef struct { double mass[3], spin, center, pole, gravity[2], step, gear, damping, range, margin, weight, alive; } planarCart;

static const planarCart PLANAR_CART = {
    { 10.47197551196598, 4.198738581522758, 4.198738581522758 },
    0.15497066975016235, 0.3, 0.6, { 1e-5, -9.81 }, 0.01, 500.0, 0.05, 1.0, 0.01, 0.08467261918769675, 10.0
};

typedef struct {
    double mass[4], spin[4], joint[3][2], center[4][2], top[4][2], bottom[4][2], radius[4], reach[4], friction[4], bodyWeight[4], dofWeight[3], low[3], high[3];
    double step, gear, armature, damping, gravity, margin, impedance, rest, noise, alive, control, height, angle, state;
    int pair[3][2];
} planarLeg;

static const planarLeg PLANAR_LEG = {
    { 3.665191429188093, 4.057890510886817, 2.7813566959781637, 5.31557476987393 },
    { 0.06924593807287505, 0.0932987568269219, 0.07230254017320971, 0.10352308059000538 },
    { { 0.0, -0.2 }, { 0.0, -0.7 + 0.25 }, { 0.13 - 0.13, -0.25 + (-0.35 + 0.1) } },
    { { 0.0, 0.0 }, { 0.0, -0.225 }, { 0.0, -0.25 }, { 0.13 - 0.065, -0.1 + 0.1 } },
    { { 0.0, 0.2 }, { 0.0, -0.225 + 0.225 }, { 0.0, -0.25 + 0.25 }, { 0.13 - 0.065 - 0.195, 0.0 } },
    { { 0.0, -0.2 }, { 0.0, -0.225 - 0.225 }, { 0.0, -0.25 - 0.25 }, { 0.13 - 0.065 + 0.195, 0.0 } },
    { 0.05, 0.05, 0.04, 0.06 },
    { 0.2 + 0.05, 0.225 + 0.05, 0.25 + 0.04, 0.195 + 0.06 },
    { 1.0, 1.0, 1.0, 2.0 },
    { 0.08492239638897525, 0.051923310146107036, 0.04959511864425975, 0.0669027107682187 },
    { 0.9173573040079748, 0.8423092317158417, 0.9000381439194955 },
    { -2.6179938779914944, -2.6179938779914944, -0.7853981633974483 },
    { 0.0, 0.0, 0.7853981633974483 },
    0.002, 200.0, 1.0, 1.0, -9.81, 0.002, 0.8, 1.25, 5e-3, 1.0, 1e-3, 0.7, 0.2, 100.0,
    { { 0, 2 }, { 0, 3 }, { 1, 3 } }
};

enum { PLANAR_REACHER_ROUND = 50, PLANAR_PENDULUM_SKIP = 5, PLANAR_HOPPER_SKIP = 4 };

typedef struct { double n[PLANAR_FREEDOM], t[PLANAR_FREEDOM], an, at, d, tangent; int pyramid; } planarBond;
typedef struct { int count, rows; planarBond bond[PLANAR_BONDS]; } planarBonds;

static inline __attribute__((always_inline)) double planar_jointBody(double pos, double vel, double weight, double *d) {
    const planarSolver *S = (const planarSolver *)opaque(&PLANAR_SOLVER);
    double imp = planar_impedance(S->low, S->high, S->high - S->low, -pos / S->width);
    double stiffness = 1 / (S->high * S->high * S->timeconst * S->timeconst), damping = 2 / (S->high * S->timeconst);
    *d = imp / ((1 - imp) * weight);
    return -damping * vel - stiffness * imp * pos;
}

static double planar_joint(double pos, double vel, double weight, double *d) { return planar_jointBody(pos, vel, weight, d); }

static inline __attribute__((always_inline)) void planar_decompose(int n, double (*l)[PLANAR_FREEDOM], double *inv) {
    double u[PLANAR_FREEDOM][PLANAR_FREEDOM];
    for (int c = 0; c < PLANAR_FREEDOM; c++)
        for (int i = 0; i < PLANAR_FREEDOM; i++) {
            if (i < c || i >= n) continue;
            double t = l[i][c];
            for (int k = 0; k < PLANAR_FREEDOM; k++)
                if (k < c) t = planar_mac(-u[i][k], l[c][k], t);
            u[i][c] = t;
            if (i == c) inv[c] = 1 / t;
            else l[i][c] = t * inv[c];
        }
}

static inline __attribute__((always_inline)) void planar_substitute(int n, const double (*l)[PLANAR_FREEDOM], const double *inv, const double *b, double *x) {
    double y[PLANAR_FREEDOM];
    for (int i = 0; i < PLANAR_FREEDOM; i++) {
        if (i >= n) continue;
        double s = b[i];
        for (int k = 0; k < PLANAR_FREEDOM; k++)
            if (k < i) s = planar_mac(-l[i][k], y[k], s);
        y[i] = s;
    }
    for (int i = PLANAR_FREEDOM - 1; i >= 0; i--) {
        if (i >= n) continue;
        double s = y[i] * inv[i];
        for (int k = PLANAR_FREEDOM - 1; k >= 0; k--)
            if (k > i && k < n) s = planar_mac(-l[k][i], x[k], s);
        x[i] = s;
    }
}

static inline __attribute__((always_inline)) double planar_inner(int n, const double *a, const double *b) {
    double s = a[0] * b[0];
    for (int i = 1; i < n; i++) s = planar_mac(a[i], b[i], s);
    return s;
}

static inline __attribute__((always_inline)) void planar_apply(int n, const double (*m)[PLANAR_FREEDOM], const double *x, double *y) {
    for (int i = 0; i < n; i++) y[i] = planar_inner(n, m[i], x);
}

static inline __attribute__((always_inline)) uint64_t planar_bondRead(int n, const planarBonds *y, const double *x, double *xn, double *xt) {
    uint64_t mask = 0u;
    for (int b = 0, row = 0; b < y->count; b++) {
        const planarBond *u = &y->bond[b];
        xn[b] = planar_inner(n, u->n, x) - u->an;
        if (u->pyramid) {
            xt[b] = planar_inner(n, u->t, x) - u->at;
            double s = u->tangent * xt[b];
            mask |= ((uint64_t)(xn[b] + s < 0) | (uint64_t)(xn[b] - s < 0) << 1 | (uint64_t)(xn[b] < 0) << 2) << row;
            row += 3;
        } else mask |= (uint64_t)(xn[b] < 0) << row++;
    }
    return mask;
}

static inline __attribute__((always_inline)) double planar_bondCost(const planarBonds *y, const double *xn, const double *xt, uint64_t mask) {
    double cost = 0;
    for (int b = 0, row = 0; b < y->count; b++) {
        const planarBond *u = &y->bond[b];
        if (u->pyramid) {
            unsigned bits = (unsigned)(mask >> row) & 7u;
            double s = u->tangent * xt[b], j0 = xn[b] + s, j1 = xn[b] - s;
            cost += bits & 1u ? u->d * j0 * j0 / 2 : 0;
            cost += bits & 2u ? u->d * j1 * j1 / 2 : 0;
            cost += bits & 4u ? 2 * u->d * xn[b] * xn[b] / 2 : 0;
            row += 3;
        } else {
            cost += mask >> row & 1u ? u->d * xn[b] * xn[b] / 2 : 0;
            row++;
        }
    }
    return cost;
}

static inline __attribute__((always_inline)) void planar_rankOne(int n, double (*h)[PLANAR_FREEDOM], double w, const double *u) {
    for (int i = 0; i < n; i++) {
        double wi = w * u[i];
        for (int k = 0; k <= i; k++) h[i][k] = planar_mac(wi, u[k], h[i][k]);
    }
}

static inline __attribute__((always_inline)) void planar_bondHessian(int n, const double (*m)[PLANAR_FREEDOM], const planarBonds *y, uint64_t mask, double (*h)[PLANAR_FREEDOM], const double *fs, double *rhs) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k <= i; k++) h[i][k] = m[i][k];
        if (rhs) rhs[i] = fs[i];
    }
    for (int b = 0, row = 0; b < y->count; b++) {
        const planarBond *u = &y->bond[b];
        if (!u->pyramid) {
            if (mask >> row & 1u) {
                planar_rankOne(n, h, u->d, u->n);
                if (rhs)
                    for (int i = 0; i < n; i++) rhs[i] = planar_mac(u->d * u->an, u->n[i], rhs[i]);
            }
            row++;
            continue;
        }
        unsigned bits = (unsigned)(mask >> row) & 7u;
        row += 3;
        if (!bits) continue;
        double tau = u->tangent, d = u->d, e[PLANAR_FREEDOM];
        if (bits == 7u) {
            planar_rankOne(n, h, 4 * d, u->n);
            planar_rankOne(n, h, 2 * d * tau * tau, u->t);
            if (rhs)
                for (int i = 0; i < n; i++) rhs[i] = planar_mac(2 * d * tau * tau * u->at, u->t[i], planar_mac(4 * d * u->an, u->n[i], rhs[i]));
            continue;
        }
        for (int s = 0; s < 2; s++) {
            if (!(bits >> s & 1u)) continue;
            double sign = s ? -tau : tau;
            for (int i = 0; i < n; i++) e[i] = planar_mac(sign, u->t[i], u->n[i]);
            planar_rankOne(n, h, d, e);
            if (rhs)
                for (int i = 0; i < n; i++) rhs[i] = planar_mac(d * planar_mac(sign, u->at, u->an), e[i], rhs[i]);
        }
        if (bits & 4u) {
            planar_rankOne(n, h, 2 * d, u->n);
            if (rhs)
                for (int i = 0; i < n; i++) rhs[i] = planar_mac(2 * d * u->an, u->n[i], rhs[i]);
        }
    }
}

static inline __attribute__((always_inline)) void planar_bondForce(int n, const planarBonds *y, const double *xn, const double *xt, uint64_t mask, double *force) {
    for (int i = 0; i < n; i++) force[i] = 0;
    for (int b = 0, row = 0; b < y->count; b++) {
        const planarBond *u = &y->bond[b];
        if (!u->pyramid) {
            if (mask >> row & 1u)
                for (int i = 0; i < n; i++) force[i] = planar_mac(-u->d * xn[b], u->n[i], force[i]);
            row++;
            continue;
        }
        unsigned bits = (unsigned)(mask >> row) & 7u;
        row += 3;
        if (!bits) continue;
        double s = u->tangent * xt[b], a = bits & 1u ? -u->d * (xn[b] + s) : 0, c = bits & 2u ? -u->d * (xn[b] - s) : 0, z = bits & 4u ? -2 * u->d * xn[b] : 0;
        double normal = a + c + z, tangent = u->tangent * (a - c);
        for (int i = 0; i < n; i++) force[i] = planar_mac(tangent, u->t[i], planar_mac(normal, u->n[i], force[i]));
    }
}

static inline __attribute__((always_inline)) int planar_bondExpand(const planarBonds *y, const double *xn, const double *xt, double pad, planarPair *pairs, planarPair *weights) {
    planarLoose *row = (planarLoose *)pairs, *w = (planarLoose *)weights;
    int r = 0;
    for (int b = 0; b < y->count; b++) {
        const planarBond *u = &y->bond[b];
        if (u->pyramid) {
            double s = u->tangent * xt[b];
            row[r] = xn[b] + s;
            row[r + 1] = xn[b] - s;
            row[r + 2] = xn[b];
            if (weights) {
                w[r] = u->d;
                w[r + 1] = u->d;
                w[r + 2] = 2 * u->d;
            }
            r += 3;
        } else {
            row[r] = xn[b];
            if (weights) w[r] = u->d;
            r++;
        }
    }
    if (r & 1) {
        row[r] = pad;
        if (weights) w[r] = 0;
    }
    return (r + 1) / 2;
}

static inline __attribute__((always_inline)) double planar_quadratic(int n, const double *mx, const double *fs, const double *x, const double *smooth) {
    double g = 0;
    for (int i = 0; i < n; i++) g = planar_mac(mx[i] - fs[i], x[i] - smooth[i], g);
    return g / 2;
}

static inline __attribute__((always_inline)) void planar_iterate(int n, const double (*m)[PLANAR_FREEDOM], const double *fs, double trace, const planarBonds *y, const double *warm, double *qacc, double *force) {
    const planarSolver *S = (const planarSolver *)opaque(&PLANAR_SOLVER);
    double wn[PLANAR_BONDS], wt[PLANAR_BONDS], tn[PLANAR_BONDS], tt[PLANAR_BONDS], sn[PLANAR_BONDS], st[PLANAR_BONDS], xn[PLANAR_BONDS], xt[PLANAR_BONDS], dn[PLANAR_BONDS], dt[PLANAR_BONDS];
    double h[PLANAR_FREEDOM][PLANAR_FREEDOM], inv[PLANAR_FREEDOM], target[PLANAR_FREEDOM], f[PLANAR_FREEDOM][PLANAR_FREEDOM], finv[PLANAR_FREEDOM], smooth[PLANAR_FREEDOM], x[PLANAR_FREEDOM], mx[PLANAR_FREEDOM], mv[PLANAR_FREEDOM], dir[PLANAR_FREEDOM], grad[PLANAR_FREEDOM], rhs[PLANAR_FREEDOM], push[PLANAR_FREEDOM];
    uint64_t wm = planar_bondRead(n, y, warm, wn, wt);
    planar_bondHessian(n, m, y, wm, h, fs, rhs);
    planar_decompose(n, h, inv);
    planar_substitute(n, h, inv, rhs, target);
    planar_bondRead(n, y, target, tn, tt);
    for (int i = 0; i < n; i++)
        for (int k = 0; k <= i; k++) f[i][k] = m[i][k];
    planar_decompose(n, f, finv);
    planar_substitute(n, f, finv, fs, smooth);
    uint64_t sm = planar_bondRead(n, y, smooth, sn, st);
    planar_apply(n, m, warm, mx);
    int smoothStart = planar_quadratic(n, mx, fs, warm, smooth) + planar_bondCost(y, wn, wt, wm) > planar_bondCost(y, sn, st, sm);
    uint64_t mask = smoothStart ? sm : wm;
    if (mask != wm) {
        planar_bondHessian(n, m, y, mask, h, fs, rhs);
        planar_decompose(n, h, inv);
        planar_substitute(n, h, inv, rhs, target);
        if (planar_bondRead(n, y, target, tn, tt) == mask) {
            for (int i = 0; i < n; i++) qacc[i] = target[i];
            if (force) planar_bondForce(n, y, tn, tt, mask, force);
            return;
        }
    }
    const double *start = smoothStart ? smooth : warm, *startN = smoothStart ? sn : wn, *startT = smoothStart ? st : wt;
    for (int i = 0; i < n; i++) {
        x[i] = start[i];
        dir[i] = target[i] - x[i];
    }
    for (int b = 0; b < y->count; b++) {
        xn[b] = startN[b];
        xt[b] = startT[b];
        dn[b] = tn[b] - xn[b];
        dt[b] = y->bond[b].pyramid ? tt[b] - xt[b] : 0;
    }
    planar_apply(n, m, x, mx);
    double gauss = planar_quadratic(n, mx, fs, x, smooth), cost = gauss + planar_bondCost(y, xn, xt, mask), scale = 1 / trace;
    int moved = 0;
    planarPair pj[PLANAR_LINES / 2 + 1], pv[PLANAR_LINES / 2 + 1], wgt[PLANAR_LINES / 2 + 1], t0[PLANAR_LINES / 2 + 1], t1[PLANAR_LINES / 2 + 1], t2[PLANAR_LINES / 2 + 1];
    int pairs = planar_bondExpand(y, xn, xt, 1, pj, wgt);
    for (int iteration = 0; iteration < PLANAR_NEWTON_ITERATIONS; iteration++) {
        planar_apply(n, m, dir, mv);
        planarReal quad[3] = { gauss, planar_inner(n, dir, mx) - planar_inner(n, fs, dir), planar_inner(n, dir, mv) / 2 };
        if (iteration)
            for (int b = 0; b < y->count; b++) {
                dn[b] = planar_inner(n, y->bond[b].n, dir);
                dt[b] = y->bond[b].pyramid ? planar_inner(n, y->bond[b].t, dir) : 0;
            }
        planar_bondExpand(y, dn, dt, 0, pv, 0);
        for (int p = 0; p < pairs; p++) {
            planarPair dj = wgt[p] * pj[p], dv = wgt[p] * pv[p];
            t0[p] = dj * pj[p] / 2;
            t1[p] = dj * pv[p];
            t2[p] = dv * pv[p] / 2;
        }
        planarLine line = { pairs, 0, pj, pv, t0, t1, t2, quad };
        planarReal alpha = planar_search(&line, __builtin_elementwise_sqrt(planar_inner(n, dir, dir)), scale, S->tolerance * S->search, PLANAR_NEWTON_EVALUATIONS);
        if (alpha == 0) break;
        for (int i = 0; i < n; i++) {
            x[i] = planar_mac(alpha, dir[i], x[i]);
            mx[i] = planar_mac(alpha, mv[i], mx[i]);
        }
        planarPair rowCost = 0;
        for (int b = 0; b < y->count; b++) {
            xn[b] = planar_mac(alpha, dn[b], xn[b]);
            xt[b] = planar_mac(alpha, dt[b], xt[b]);
        }
        for (int p = 0; p < pairs; p++) {
            pj[p] = planar_fma((planarPair)alpha, pv[p], pj[p]);
            rowCost += (planarPair)((planarLane)(wgt[p] * pj[p] * pj[p] / 2) & (pj[p] < 0));
        }
        uint64_t next = 0u;
        for (int p = 0; p < pairs; p++) {
            planarLane on = pj[p] < 0;
            next |= ((uint64_t)(on.x & 1) | (uint64_t)(on.y & 1) << 1) << (2 * p);
        }
        next &= y->rows < 64 ? ((uint64_t)1 << y->rows) - 1u : ~(uint64_t)0;
        planar_bondForce(n, y, xn, xt, next, push);
        moved = 1;
        double previous = cost;
        gauss = planar_quadratic(n, mx, fs, x, smooth);
        cost = planar_sum(rowCost) + gauss;
        for (int i = 0; i < n; i++) grad[i] = mx[i] - fs[i] - push[i];
        uint64_t changed = next ^ mask;
        mask = next;
        if (scale * (previous - cost) < S->tolerance || scale * __builtin_elementwise_sqrt(planar_inner(n, grad, grad)) < S->tolerance) break;
        if (changed) {
            planar_bondHessian(n, m, y, mask, h, 0, 0);
            planar_decompose(n, h, inv);
        }
        planar_substitute(n, h, inv, grad, dir);
        for (int i = 0; i < n; i++) dir[i] = -dir[i];
    }
    if (!moved) planar_bondForce(n, y, xn, xt, mask, push);
    for (int i = 0; i < n; i++) qacc[i] = x[i];
    if (force)
        for (int i = 0; i < n; i++) force[i] = push[i];
}

__attribute__((noinline)) static void planar_detour(int n, const double (*m)[PLANAR_FREEDOM], const double *fs, double trace, const planarBonds *y, const double *warm, double *qacc, double *force) {
    if (n == 6) planar_iterate(6, m, fs, trace, y, warm, qacc, force);
    else if (n == 3) planar_iterate(3, m, fs, trace, y, warm, qacc, force);
    else planar_iterate(2, m, fs, trace, y, warm, qacc, force);
}

static inline __attribute__((always_inline)) uint64_t planar_guess(int n, const double (*m)[PLANAR_FREEDOM], const double *fs, const planarBonds *y, uint64_t mask, double *target, double *xn, double *xt) {
    double h[PLANAR_FREEDOM][PLANAR_FREEDOM], inv[PLANAR_FREEDOM], rhs[PLANAR_FREEDOM];
    planar_bondHessian(n, m, y, mask, h, fs, rhs);
    planar_decompose(n, h, inv);
    planar_substitute(n, h, inv, rhs, target);
    return planar_bondRead(n, y, target, xn, xt);
}

static inline __attribute__((always_inline)) void planar_settle(int n, const double (*m)[PLANAR_FREEDOM], const double *fs, double trace, const planarBonds *y, const double *warm, double *qacc, double *force) {
    double xn[PLANAR_BONDS], xt[PLANAR_BONDS], target[PLANAR_FREEDOM];
    uint64_t mask = planar_bondRead(n, y, warm, xn, xt), next = planar_guess(n, m, fs, y, mask, target, xn, xt);
    if (__builtin_expect(next != mask, 0)) {
        mask = next;
        next = planar_guess(n, m, fs, y, mask, target, xn, xt);
    }
    if (__builtin_expect(next == mask, 1)) {
        for (int i = 0; i < n; i++) qacc[i] = target[i];
        if (force) planar_bondForce(n, y, xn, xt, mask, force);
        return;
    }
    planar_detour(n, m, fs, trace, y, warm, qacc, force);
}

static const double PLANAR_TURN[9] = { -1.0 / 2.0, 1.0 / 24.0, -1.0 / 720.0, 1.0 / 40320.0, -1.0 / 3628800.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362880.0 };

typedef struct { planarPair c, s; } planarTwin;

static inline __attribute__((always_inline)) planarTwin planar_spin(planarTwin b, planarPair e) {
    const double *k = (const double *)opaque(PLANAR_TURN);
    planarPair z = e * e, z2 = z * z, one = (planarPair)1;
    planarPair cosine = planar_fma(z2, planar_fma(z2, planar_fma(z, (planarPair)k[4], (planarPair)k[3]), planar_fma(z, (planarPair)k[2], (planarPair)k[1])), planar_fma(z, (planarPair)k[0], one));
    planarPair sine = planar_fma(z2, planar_fma(z2, (planarPair)k[8], planar_fma(z, (planarPair)k[7], (planarPair)k[6])), planar_fma(z, (planarPair)k[5], one));
    planarTwin r = { planar_fma(b.c, cosine, -(b.s * e) * sine), planar_fma(b.s, cosine, (b.c * e) * sine) };
    return r;
}

static planarTwin planar_origin(planarPair e) {
    planarTwin b = { (planarPair)1, (planarPair)0 };
    return planar_spin(b, e);
}

__attribute__((noinline)) static planarTwin planar_far(const planarTable *P, planarPair theta) {
    planarPair x[1] = { theta }, s[1], c[1];
    planar_sincos(P, 1, x, s, c);
    planarTwin r = { c[0], s[0] };
    return r;
}

static inline __attribute__((always_inline)) planarTwin planar_turn(const planarTable *P, planarTwin b, planarPair e, planarPair theta) {
    planarLane near = __builtin_elementwise_abs(e) <= (planarPair)0.125;
    return __builtin_expect((near.x & near.y) != 0, 1) ? planar_spin(b, e) : planar_far(P, theta);
}

static inline __attribute__((always_inline)) void planar_fresh(const planarTable *P, int count, const planarPair *theta, planarTwin *out) {
    for (int g = 0; g < count; g += 4) {
        int span = count - g < 4 ? count - g : 4;
        planarPair s[4], c[4];
        planar_sincos(P, span, theta + g, s, c);
        for (int i = 0; i < span; i++) {
            out[g + i].c = c[i];
            out[g + i].s = s[i];
        }
    }
}

__attribute__((noinline)) static void planar_limit(int n, const double (*m)[PLANAR_FREEDOM], const double *fs, double trace, int dof, double sign, double gap, const double *v, double weight, const double *warm, double *acc, double *force) {
    planarBonds y;
    planarBond *u = &y.bond[0];
    for (int i = 0; i < n; i++) u->n[i] = 0;
    y.count = 1;
    y.rows = 1;
    u->pyramid = 0;
    u->n[dof] = sign;
    u->an = planar_joint(gap, sign * v[dof], weight, &u->d);
    if (n == 2) planar_settle(2, m, fs, trace, &y, warm, acc, force);
    else planar_settle(3, m, fs, trace, &y, warm, acc, force);
}

typedef struct { double base, inner, outer, moment, det[3], damping, range, trace, weight; } planarArmRig;

static planarArmRig planar_armRig(const planarArm *A) {
    planarArmRig r;
    r.inner = planar_mac(A->mass[1] * A->center[1], A->center[1], A->spin[1]) + planar_mac(A->mass[2] * A->center[2], A->center[2], A->spin[2]);
    r.moment = planar_mac(A->mass[1], A->center[1], A->mass[2] * A->center[2]) * A->link;
    r.base = A->armature + planar_mac(A->mass[0] * A->center[0], A->center[0], A->spin[0]) + r.inner + (A->mass[1] + A->mass[2]) * A->link * A->link;
    r.outer = A->armature + r.inner;
    r.det[0] = r.outer * r.base - r.inner * r.inner;
    r.det[1] = 2 * r.moment * A->armature;
    r.det[2] = -r.moment * r.moment;
    r.damping = A->damping;
    r.range = A->range;
    r.trace = 2 * A->target;
    r.weight = A->weight;
    return r;
}

typedef struct { double total, h1, h2, k, j12, j2, gx, gz, drift, damping, range, margin, weight; } planarCartRig;

static planarCartRig planar_cartRig(const planarCart *C) {
    planarCartRig r;
    double m1 = C->mass[1], m2 = C->mass[2], l = C->center, pole = C->pole;
    r.total = C->mass[0] + m1 + m2;
    r.h1 = planar_mac(m1, l, m2 * pole);
    r.h2 = m2 * l;
    r.k = m2 * pole * l;
    r.j2 = planar_mac(m2 * l, l, C->spin);
    r.j12 = planar_mac(m2 * pole, pole, planar_mac(m1 * l, l, C->spin)) + r.j2;
    r.gx = C->gravity[0];
    r.gz = C->gravity[1];
    r.drift = -r.total * r.gx;
    r.damping = C->damping;
    r.range = C->range;
    r.margin = C->margin;
    r.weight = C->weight;
    return r;
}

typedef struct { double contact[4], pair[3], spring, damping; } planarLegRig;

static planarLegRig planar_legRig(const planarLeg *L, const planarSolver *S) {
    planarLegRig r;
    for (int g = 0; g < 4; g++) {
        double mu = L->friction[g];
        r.contact[g] = L->impedance / ((1 - L->impedance) * (L->bodyWeight[g] * (1 + mu * mu) * 2 * mu * mu));
    }
    for (int p = 0; p < 3; p++) r.pair[p] = L->impedance / ((1 - L->impedance) * (L->bodyWeight[L->pair[p][0]] + L->bodyWeight[L->pair[p][1]]));
    r.spring = 1 / (L->impedance * L->impedance * S->timeconst * S->timeconst) * L->impedance;
    r.damping = 2 / (L->impedance * S->timeconst);
    return r;
}

typedef struct { planarArmRig arm; planarCartRig cart; planarLegRig leg; } planarKit;
typedef planarArmRig planarArmTable __attribute__((aligned(4)));
typedef planarCartRig planarCartTable __attribute__((aligned(4)));
typedef planarLegRig planarLegTable __attribute__((aligned(4)));
typedef planarKit planarKitTable __attribute__((aligned(4)));

static const planarKitTable *planar_kit(const planarTable *P) { return (const planarKitTable *)((const float *)P + sizeof(planarModel) / sizeof(float)); }

uint64_t planar_shared(float *table) {
    if (table) {
        planarModel model;
        planar_derive(&model);
        memcpy(table, &model, sizeof model);
        planarKit kit = { planar_armRig(&PLANAR_ARM), planar_cartRig(&PLANAR_CART), planar_legRig(&PLANAR_LEG, &PLANAR_SOLVER) };
        memcpy(table + sizeof model / sizeof(float), &kit, sizeof kit);
    }
    return (sizeof(planarModel) + sizeof(planarKit)) / sizeof(float);
}

static inline __attribute__((always_inline)) double planar_reacherBond(double sign, double an, double x0, double x1) { return planar_mac(sign, x1, 0.0 * x0) - an; }

static inline __attribute__((always_inline)) planarPair planar_reacherGuess(double inv0, double l10, double u11, double y0, double y1) {
    double x1 = y1 * (1 / u11);
    return planar_pair(x1, planar_mac(-l10, x1, y0 * inv0));
}

__attribute__((noinline)) static planarPair planar_reacherLimit(const planarArmTable *R, planarPair mass, planarPair force, planarPair speed, planarPair warm, double angle, double gap) {
    double m00 = mass.x, m01 = mass.y, f0 = force.x, f1 = force.y;
    double sign = angle < 0 ? 1 : -1, d, an = planar_jointBody(gap, sign * speed.y, R->weight, &d);
    double inv0 = 1 / m00, l10 = m01 * inv0, da = d * an, pushed = planar_mac(da, 0.0, f0);
    planarPair free = planar_reacherGuess(inv0, l10, planar_mac(-m01, l10, R->outer), f0, planar_mac(-l10, f0, f1));
    planarPair held = planar_reacherGuess(inv0, l10, planar_mac(-m01, l10, planar_mac(d * sign, sign, R->outer)), pushed, planar_mac(-l10, pushed, planar_mac(da, sign, f1)));
    int loose = !(planar_reacherBond(sign, an, free.y, free.x) < 0), tight = planar_reacherBond(sign, an, held.y, held.x) < 0;
    if (planar_reacherBond(sign, an, warm.x, warm.y) < 0) {
        if (__builtin_expect(tight, 1)) return held;
        if (__builtin_expect(loose, 1)) return free;
    } else {
        if (__builtin_expect(loose, 1)) return free;
        if (__builtin_expect(tight, 1)) return held;
    }
    double m[PLANAR_FREEDOM][PLANAR_FREEDOM], fs[2] = { f0, f1 }, v[2] = { speed.x, speed.y }, start[2] = { warm.x, warm.y }, out[2];
    m[0][0] = m00;
    m[0][1] = m[1][0] = m01;
    m[1][1] = R->outer;
    planar_limit(2, m, fs, m00 + R->outer + R->trace, 1, sign, gap, v, R->weight, start, out, 0);
    return planar_pair(out[1], out[0]);
}

static inline __attribute__((always_inline)) planarPair planar_reacherForward(const planarArmTable *R, const double *q, const double *v, planarTwin trig, const double *tau, const double *warm) {
    double c = trig.c.x, s = trig.s.x;
    double m00 = planar_mac(2 * R->moment, c, R->base), m01 = planar_mac(R->moment, c, R->inner), det = planar_mac(c, planar_mac(R->det[2], c, R->det[1]), R->det[0]);
    double coupling = m01 / det, bs = R->moment * s;
    planarPair scaled = planar_pair(m00, R->outer) / (planarPair)det;
    double f0 = planar_mac(bs * v[1], planar_mac(2, v[0], v[1]), planar_mac(-R->damping, v[0], tau[0]));
    double f1 = planar_mac(-bs * v[0], v[0], planar_mac(-R->damping, v[1], tau[1]));
    double gap = R->range - fabs(q[1]);
    if (__builtin_expect(gap < 0, 0)) return planar_reacherLimit(R, planar_pair(m00, m01), planar_pair(f0, f1), planar_pair(v[0], v[1]), planar_pair(warm[0], warm[1]), q[1], gap);
    return planar_fma(scaled, planar_pair(f1, f0), (planarPair)(-coupling) * planar_pair(f0, f1));
}

static inline __attribute__((always_inline)) double planar_pendulumForward(const planarCartTable *R, const double *q, const double *v, const planarTwin *trig, const double *tau, const double *warm, double *acc) {
    double c1 = trig[0].c.x, s1 = trig[0].s.x, c12 = trig[0].c.y, s12 = trig[0].s.y;
    double c2 = planar_mac(c12, c1, s12 * s1), s2 = planar_mac(s12, c1, -c12 * s1);
    double m01 = planar_mac(R->h1, c1, R->h2 * c12), m02 = R->h2 * c12, m11 = planar_mac(2 * R->k, c2, R->j12), m12 = planar_mac(R->k, c2, R->j2), m22 = R->j2;
    double w1 = v[1], w2 = v[1] + v[2], ww1 = w1 * w1, ww2 = w2 * w2;
    double bx = planar_mac(-R->h2 * s12, ww2, planar_mac(-R->h1 * s1, ww1, R->drift));
    double b2 = planar_mac(R->k * s2, ww1, R->h2 * planar_mac(s12, R->gz, -c12 * R->gx));
    double b1 = planar_mac(-R->k * s2, ww2, R->h1 * planar_mac(s1, R->gz, -c1 * R->gx));
    double fs[3] = { planar_mac(-R->damping, v[0], tau[0]) - bx, planar_mac(-R->damping, v[1], -b2) - b1, planar_mac(-R->damping, v[2], -b2) };
    double a00 = planar_mac(m11, m22, -m12 * m12), a01 = planar_mac(m02, m12, -m01 * m22), a02 = planar_mac(m01, m12, -m02 * m11);
    double a11 = planar_mac(R->total, m22, -m02 * m02), a12 = planar_mac(m01, m02, -R->total * m12), a22 = planar_mac(R->total, m11, -m01 * m01);
    double inv = 1 / planar_mac(m02, a02, planar_mac(m01, a01, R->total * a00));
    double e00 = a00 * inv, e01 = a01 * inv, e02 = a02 * inv, e11 = a11 * inv, e12 = a12 * inv, e22 = a22 * inv;
    acc[0] = planar_mac(e01, fs[1], planar_mac(e00, fs[0], e02 * fs[2]));
    acc[1] = planar_mac(e11, fs[1], planar_mac(e01, fs[0], e12 * fs[2]));
    acc[2] = planar_mac(e12, fs[1], planar_mac(e02, fs[0], e22 * fs[2]));
    double gap = (q[0] < 0 ? q[0] + R->range : R->range - q[0]) - R->margin, force[3] = { 0, 0, 0 };
    if (__builtin_expect(gap < 0, 0)) {
        double m[PLANAR_FREEDOM][PLANAR_FREEDOM], speed[3] = { v[0], v[1], v[2] }, start[3] = { warm[0], warm[1], warm[2] }, out[3] = { acc[0], acc[1], acc[2] }, push[3];
        m[0][0] = R->total;
        m[1][0] = m[0][1] = m01;
        m[2][0] = m[0][2] = m02;
        m[1][1] = m11;
        m[2][1] = m[1][2] = m12;
        m[2][2] = m22;
        planar_limit(3, m, fs, R->total + m11 + m22, 0, q[0] < 0 ? 1 : -1, gap, speed, R->weight, start, out, push);
        for (int i = 0; i < 3; i++) acc[i] = out[i];
        force[0] = push[0];
    }
    return force[0];
}

static void planar_rotate(double c, double s, const double *u, double *out) {
    out[0] = planar_mac(u[0], c, u[1] * s);
    out[1] = planar_mac(-u[0], s, u[1] * c);
}

static double planar_cross(const double *u, const double *w) { return planar_mac(u[1], w[0], -u[0] * w[1]); }

static void planar_closest(const double *p0, const double *p1, const double *q0, const double *q1, double *pa, double *pb) {
    double u[2] = { p1[0] - p0[0], p1[1] - p0[1] }, v[2] = { q1[0] - q0[0], q1[1] - q0[1] }, w[2] = { p0[0] - q0[0], p0[1] - q0[1] };
    double a = planar_mac(u[0], u[0], u[1] * u[1]), b = planar_mac(u[0], v[0], u[1] * v[1]), c = planar_mac(v[0], v[0], v[1] * v[1]);
    double d = planar_mac(u[0], w[0], u[1] * w[1]), e = planar_mac(v[0], w[0], v[1] * w[1]), den = planar_mac(a, c, -b * b);
    double s = den > (double)PLANAR_MINVAL ? fmin(fmax(planar_mac(b, e, -c * d) / den, 0), 1) : 0, t = planar_mac(b, s, e) / c;
    if (t < 0) {
        t = 0;
        s = fmin(fmax(-d / a, 0), 1);
    } else if (t > 1) {
        t = 1;
        s = fmin(fmax((b - d) / a, 0), 1);
    }
    for (int i = 0; i < 2; i++) {
        pa[i] = planar_mac(s, u[i], p0[i]);
        pb[i] = planar_mac(t, v[i], q0[i]);
    }
}

static int planar_crosses(const double *p0, const double *p1, const double *q0, const double *q1) {
    double d1 = planar_mac(q1[0] - q0[0], p0[1] - q0[1], -(q1[1] - q0[1]) * (p0[0] - q0[0])), d2 = planar_mac(q1[0] - q0[0], p1[1] - q0[1], -(q1[1] - q0[1]) * (p1[0] - q0[0]));
    double d3 = planar_mac(p1[0] - p0[0], q0[1] - p0[1], -(p1[1] - p0[1]) * (q0[0] - p0[0])), d4 = planar_mac(p1[0] - p0[0], q1[1] - p0[1], -(p1[1] - p0[1]) * (q1[0] - p0[0]));
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

static double planar_axis(int a) { return a ? -1 : 1; }

__attribute__((noinline)) static void planar_touch(const planarLegTable *R, int p, const double (*end)[2][2], const double (*joint)[2], const double *v, planarBonds *y) {
    const planarLeg *L = (const planarLeg *)opaque(&PLANAR_LEG);
    int g1 = PLANAR_LEG.pair[p][0], g2 = PLANAR_LEG.pair[p][1];
    if (planar_crosses(end[g1][0], end[g1][1], end[g2][0], end[g2][1])) return;
    double pa[2], pb[2];
    planar_closest(end[g1][0], end[g1][1], end[g2][0], end[g2][1], pa, pb);
    double diff[2] = { pb[0] - pa[0], pb[1] - pa[1] }, length = __builtin_elementwise_sqrt(planar_mac(diff[0], diff[0], diff[1] * diff[1]));
    double r1 = L->radius[g1], dist = length - r1 - L->radius[g2];
    if (dist >= L->margin) return;
    double normal[2] = { diff[0] / length, diff[1] / length }, pos[2];
    for (int i = 0; i < 2; i++) pos[i] = planar_mac(normal[i], r1 + dist / 2, pa[i]);
    planarBond *u = &y->bond[y->count++];
    for (int i = 0; i < 6; i++) u->n[i] = 0;
    for (int a = g1 + 1; a <= g2; a++) {
        double lever[2] = { pos[0] - joint[a][0], pos[1] - joint[a][1] };
        u->n[2 + a] = planar_axis(a) * planar_mac(normal[0], lever[1], -normal[1] * lever[0]);
    }
    u->pyramid = 0;
    u->an = planar_mac(-R->damping, planar_inner(6, u->n, v), -R->spring * (dist - L->margin));
    u->d = R->pair[p];
    y->rows++;
}

__attribute__((noinline)) static void planar_hopperForward(const planarLegTable *R, const double *q, const double *v, const planarTwin *trig, const double *tau, const double *warm, double *acc) {
    const planarLeg *L = (const planarLeg *)opaque(&PLANAR_LEG);
    double c[4] = { trig[0].c.x, trig[0].c.y, trig[1].c.x, trig[1].c.y }, s[4] = { trig[0].s.x, trig[0].s.y, trig[1].s.x, trig[1].s.y };
    double joint[4][2], link[3][2], arm[4][2], omega[4];
    joint[0][0] = 0;
    joint[0][1] = q[1];
    omega[0] = v[2];
    for (int i = 0; i < 4; i++) {
        planar_rotate(c[i], s[i], L->center[i], arm[i]);
        if (i) omega[i] = omega[i - 1] - v[2 + i];
        if (i < 3) {
            planar_rotate(c[i], s[i], L->joint[i], link[i]);
            joint[i + 1][0] = joint[i][0] + link[i][0];
            joint[i + 1][1] = joint[i][1] + link[i][1];
        }
    }
    double mass[4], moment[4][2], spin[4];
    for (int i = 3; i >= 0; i--) {
        double mi = L->mass[i];
        mass[i] = mi;
        moment[i][0] = mi * arm[i][0];
        moment[i][1] = mi * arm[i][1];
        spin[i] = planar_mac(mi, planar_mac(arm[i][0], arm[i][0], arm[i][1] * arm[i][1]), L->spin[i]);
        if (i < 3) {
            double *d = link[i], dd = planar_mac(d[0], d[0], d[1] * d[1]);
            spin[i] += spin[i + 1] + planar_mac(mass[i + 1], dd, 2 * planar_mac(d[0], moment[i + 1][0], d[1] * moment[i + 1][1]));
            moment[i][0] += planar_mac(mass[i + 1], d[0], moment[i + 1][0]);
            moment[i][1] += planar_mac(mass[i + 1], d[1], moment[i + 1][1]);
            mass[i] += mass[i + 1];
        }
    }
    double m[PLANAR_FREEDOM][PLANAR_FREEDOM];
    m[0][0] = m[1][1] = mass[0];
    m[0][1] = m[1][0] = 0;
    for (int a = 0; a < 4; a++) {
        double sa = planar_axis(a);
        m[0][2 + a] = m[2 + a][0] = sa * moment[a][1];
        m[1][2 + a] = m[2 + a][1] = -sa * moment[a][0];
        for (int b = a; b < 4; b++) {
            double gap[2] = { joint[b][0] - joint[a][0], joint[b][1] - joint[a][1] };
            m[2 + a][2 + b] = m[2 + b][2 + a] = sa * planar_axis(b) * (spin[b] + planar_mac(gap[0], moment[b][0], gap[1] * moment[b][1]));
        }
    }
    for (int a = 3; a < 6; a++) m[a][a] += L->armature;
    double accel[2] = { 0, 0 }, pull[4][2], net[2] = { 0, 0 }, torque = 0, fs[PLANAR_FREEDOM];
    for (int i = 0; i < 4; i++) {
        double w2 = omega[i] * omega[i];
        pull[i][0] = L->mass[i] * planar_mac(-w2, arm[i][0], accel[0]);
        pull[i][1] = L->mass[i] * (planar_mac(-w2, arm[i][1], accel[1]) - L->gravity);
        if (i < 3) {
            accel[0] = planar_mac(-w2, link[i][0], accel[0]);
            accel[1] = planar_mac(-w2, link[i][1], accel[1]);
        }
    }
    for (int i = 3; i >= 0; i--) {
        if (i < 3) torque += planar_cross(link[i], net);
        torque += planar_cross(arm[i], pull[i]);
        net[0] += pull[i][0];
        net[1] += pull[i][1];
        double drive = i ? planar_mac(-L->damping, v[2 + i], tau[i - 1]) : 0;
        fs[2 + i] = drive - planar_axis(i) * torque;
    }
    fs[0] = -net[0];
    fs[1] = -net[1];
    planarBonds y;
    y.count = 0;
    y.rows = 0;
    for (int k = 0; k < 3; k++) {
        double gap = q[3 + k] - L->low[k], sign = 1;
        if (gap >= 0) {
            gap = L->high[k] - q[3 + k];
            sign = -1;
        }
        if (gap < 0) {
            planarBond *u = &y.bond[y.count++];
            for (int i = 0; i < 6; i++) u->n[i] = 0;
            u->n[3 + k] = sign;
            u->pyramid = 0;
            u->an = planar_joint(gap, sign * v[3 + k], L->dofWeight[k], &u->d);
            y.rows++;
        }
    }
    double spring = R->spring, damping = R->damping;
    double end[4][2][2];
    for (int g = 0; g < 4; g++) {
        double top[2], bottom[2];
        planar_rotate(c[g], s[g], L->top[g], top);
        planar_rotate(c[g], s[g], L->bottom[g], bottom);
        for (int i = 0; i < 2; i++) {
            end[g][0][i] = joint[g][i] + top[i];
            end[g][1][i] = joint[g][i] + bottom[i];
        }
        double radius = L->radius[g];
        for (int e = 0; e < 2; e++) {
            double dist = end[g][e][1] - radius;
            if (dist >= L->margin) continue;
            double ax = top[0] - bottom[0], mu = L->friction[g], pos[2] = { end[g][e][0], end[g][e][1] - radius - dist / 2 };
            planarBond *u = &y.bond[y.count++];
            u->pyramid = 1;
            u->tangent = ax < 0 && -ax >= (double)PLANAR_MINVAL ? -mu : mu;
            u->d = R->contact[g];
            u->n[0] = 0;
            u->n[1] = 1;
            u->t[0] = 1;
            u->t[1] = 0;
            for (int a = 0; a < 4; a++) {
                double sa = a <= g ? planar_axis(a) : 0;
                u->n[2 + a] = -sa * (pos[0] - joint[a][0]);
                u->t[2 + a] = sa * (pos[1] - joint[a][1]);
            }
            u->an = planar_mac(-damping, planar_inner(6, u->n, v), -spring * (dist - L->margin));
            u->at = -damping * planar_inner(6, u->t, v);
            y.rows += 3;
        }
    }
    for (int p = 0; p < 3; p++) {
        int g1 = PLANAR_LEG.pair[p][0], g2 = PLANAR_LEG.pair[p][1];
        double reach = L->reach[g1] + L->reach[g2] + L->margin, dx = joint[g1][0] + arm[g1][0] - joint[g2][0] - arm[g2][0], dz = joint[g1][1] + arm[g1][1] - joint[g2][1] - arm[g2][1];
        if (__builtin_expect(planar_mac(dx, dx, dz * dz) < reach * reach, 0)) planar_touch(R, p, end, joint, v, &y);
    }
    planar_settle(6, m, fs, m[0][0] + m[1][1] + m[2][2] + m[3][3] + m[4][4] + m[5][5], &y, warm, acc, 0);
}

static inline __attribute__((always_inline)) void planar_pairs(int robot, const double *x, planarPair *out) {
    if (robot == PLANAR_ROBOT_PENDULUM) out[0] = planar_pair(x[1], x[1] + x[2]);
    else {
        double f1 = x[2] - x[3], f2 = f1 - x[4];
        out[0] = planar_pair(x[2], f1);
        out[1] = planar_pair(f2, f2 - x[5]);
    }
}

typedef struct { double q[PLANAR_FREEDOM], constraint; planarTwin trig[2]; } planarTail;

static inline __attribute__((always_inline)) double planar_forward(int robot, const void *rig, const double *q, const double *v, const planarTwin *trig, const double *tau, const double *warm, double *acc) {
    if (robot == PLANAR_ROBOT_PENDULUM) return planar_pendulumForward((const planarCartTable *)rig, q, v, trig, tau, warm, acc);
    planar_hopperForward((const planarLegTable *)rig, q, v, trig, tau, warm, acc);
    return 0;
}

static inline __attribute__((always_inline)) planarTwin planar_compose(planarTwin a, planarTwin b) {
    planarTwin r = { planar_fma(a.c, b.c, -(a.s * b.s)), planar_fma(a.s, b.c, a.c * b.s) };
    return r;
}

static inline __attribute__((always_inline)) void planar_integrate(const planarTable *P, int robot, const void *rig, int substeps, double h, double *q, double *v, double *w, planarTwin *trig, int stored, const double *tau, planarTail *tail) {
    enum { GROUPS = 2 };
    int n = robot == PLANAR_ROBOT_PENDULUM ? 3 : 6, g = robot == PLANAR_ROBOT_HOPPER ? 2 : 1;
    double half = 0.5 * h, sixth = 1.0 / 6.0, third = 1.0 / 3.0;
    planarPair quarter = (planarPair)(0.25 * h * h), square = (planarPair)(0.5 * h * h), early = (planarPair)(h * h / 6), late = (planarPair)(h * h / 12), twice = (planarPair)(h * h / 6);
    planarPair th[GROUPS], om[GROUPS], theta[2 * GROUPS];
    planarTwin fresh[2 * GROUPS], b0[GROUPS], b1[GROUPS], b2[GROUPS], b3[GROUPS], b4[GROUPS];
    double k0[PLANAR_FREEDOM];
    if (stored) {
        for (int i = 0; i < g; i++) b0[i] = trig[i];
        planar_forward(robot, rig, q, v, trig, tau, w, k0);
    }
    planar_pairs(robot, q, th);
    planar_pairs(robot, v, om);
    for (int i = 0; i < g; i++) {
        planarPair step = (planarPair)half * om[i];
        if (stored) b1[i] = planar_turn(P, b0[i], step, th[i] + step);
        theta[i] = stored ? planar_fma((planarPair)h, om[i], th[i]) : th[i];
        theta[g + i] = step;
    }
    planar_fresh(P, 2 * g, theta, fresh);
    for (int i = 0; i < g; i++) {
        planarTwin r = fresh[g + i];
        if (!stored) {
            b0[i] = fresh[i];
            b1[i] = planar_compose(b0[i], r);
        }
        b2[i] = stored ? fresh[i] : planar_compose(b1[i], r);
        b3[i] = planar_compose(b2[i], r);
        b4[i] = planar_compose(b3[i], r);
    }
    for (int sub = 0; sub < substeps; sub++) {
        double k1[PLANAR_FREEDOM], k2[PLANAR_FREEDOM], k3[PLANAR_FREEDOM], v1[PLANAR_FREEDOM], v2[PLANAR_FREEDOM], v3[PLANAR_FREEDOM], qs[PLANAR_FREEDOM];
        planarPair a[GROUPS], ang[GROUPS], e0[GROUPS], e1[GROUPS];
        planarTwin ts[GROUPS];
        if (sub || !stored) planar_forward(robot, rig, q, v, b0, tau, w, k0);
        for (int i = 0; i < n; i++) {
            qs[i] = planar_mac(half, v[i], q[i]);
            v1[i] = planar_mac(half, k0[i], v[i]);
        }
        planar_forward(robot, rig, qs, v1, b1, tau, w, k1);
        for (int i = 0; i < n; i++) {
            qs[i] = planar_mac(half, v1[i], q[i]);
            v2[i] = planar_mac(half, k1[i], v[i]);
        }
        planar_pairs(robot, k0, a);
        planar_pairs(robot, qs, ang);
        for (int i = 0; i < g; i++) {
            e0[i] = early * a[i];
            e1[i] = late * a[i];
            ts[i] = planar_turn(P, b1[i], quarter * a[i], ang[i]);
        }
        planar_forward(robot, rig, qs, v2, ts, tau, w, k2);
        for (int i = 0; i < n; i++) {
            qs[i] = planar_mac(h, v2[i], q[i]);
            v3[i] = planar_mac(h, k2[i], v[i]);
        }
        planar_pairs(robot, k1, a);
        planar_pairs(robot, qs, ang);
        for (int i = 0; i < g; i++) {
            e0[i] = planar_fma(early, a[i], e0[i]);
            e1[i] = planar_fma(twice, a[i], e1[i]);
            ts[i] = planar_turn(P, b2[i], square * a[i], ang[i]);
        }
        double constraint = planar_forward(robot, rig, qs, v3, ts, tau, w, k3);
        if (sub + 1 == substeps) {
            for (int i = 0; i < n; i++) tail->q[i] = qs[i];
            for (int i = 0; i < g; i++) tail->trig[i] = ts[i];
            tail->constraint = constraint;
        }
        for (int i = 0; i < n; i++) {
            double dq = planar_mac(sixth, v3[i], planar_mac(third, v2[i], planar_mac(third, v1[i], sixth * v[i])));
            double dv = planar_mac(sixth, k3[i], planar_mac(third, k2[i], planar_mac(third, k1[i], sixth * k0[i])));
            q[i] = planar_mac(h, dq, q[i]);
            v[i] = planar_mac(h, dv, v[i]);
            w[i] = k3[i];
        }
        planar_pairs(robot, k2, a);
        planar_pairs(robot, q, th);
        planar_pairs(robot, v, om);
        for (int i = 0; i < g; i++) {
            e0[i] = planar_fma(early, a[i], e0[i]);
            e1[i] = planar_fma(twice, a[i], e1[i]);
            b0[i] = planar_turn(P, b2[i], e0[i], th[i]);
        }
        if (sub + 1 == substeps) break;
        planar_pairs(robot, k3, a);
        for (int i = 0; i < g; i++) {
            planarPair rest = planar_fma(late, a[i], e1[i]);
            b1[i] = planar_turn(P, b3[i], e0[i] + rest, planar_fma((planarPair)half, om[i], th[i]));
            b2[i] = planar_turn(P, b4[i], planar_fma((planarPair)2, rest, e0[i]), planar_fma((planarPair)h, om[i], th[i]));
            theta[i] = planar_fma((planarPair)(3 * half), om[i], th[i]);
            theta[g + i] = planar_fma((planarPair)(2 * h), om[i], th[i]);
        }
        if (sub + 2 < substeps) {
            planar_fresh(P, 2 * g, theta, fresh);
            for (int i = 0; i < g; i++) {
                b3[i] = fresh[i];
                b4[i] = fresh[g + i];
            }
        }
    }
    for (int i = 0; i < g; i++) trig[i] = b0[i];
}

static void planar_loadDoubles(const uint32_t *words, double *x, int n) { memcpy(x, words, (size_t)n * sizeof(double)); }
static void planar_storeDoubles(const double *x, uint32_t *words, int n) { memcpy(words, x, (size_t)n * sizeof(double)); }

static double planar_torque(float ctrl, double gear) { return gear * (double)fmaxf(-1.0f, fminf(1.0f, ctrl)); }

static float planar_clip(double x) { return (float)fmin(fmax(x, -10.0), 10.0); }

static double planar_goal(const planarArm *A, const float *goal, int axis) { return A->ref[axis] + ((double)goal[axis] - A->ref[axis]); }

static void planar_fingertip(const planarArm *A, planarTwin t, const float *goal, double *dx, double *dy) {
    double c1 = t.c.x, s1 = t.s.x, c0 = t.c.y, s0 = t.s.y, reach = A->center[2];
    *dx = planar_mac(A->link, c0, reach * planar_mac(c0, c1, -s0 * s1)) - planar_goal(A, goal, 0);
    *dy = planar_mac(A->link, s0, reach * planar_mac(s0, c1, c0 * s1)) - planar_goal(A, goal, 1);
}

static void planar_reacherSense(planarTwin t, const double *v, double dx, double dy, float *obs) {
    obs[0] = (float)t.c.y;
    obs[1] = (float)t.c.x;
    obs[2] = (float)t.s.y;
    obs[3] = (float)t.s.x;
    obs[6] = (float)v[0];
    obs[7] = (float)v[1];
    obs[8] = (float)dx;
    obs[9] = (float)dy;
}

static void planar_reacherDraw(uint32_t seed, uint32_t *words) {
    const planarArm *A = (const planarArm *)opaque(&PLANAR_ARM);
    uint32_t resets = words[31];
    u4 r = philox4x32(seed, RNG_STREAM_RESET, resets, 0u, 0u, 0u), g = philox4x32(seed, RNG_STREAM_RESET, resets, 1u, 0u, 0u);
    double state[6] = { (double)uniformRange(r.x, -0.1f, 0.1f), (double)uniformRange(r.y, -0.1f, 0.1f), (double)uniformRange(r.z, -0.005f, 0.005f), (double)uniformRange(r.w, -0.005f, 0.005f), 0, 0 }, dx, dy;
    float radius = 0.2f * sqrtf(u01(g.x)), sn, cs, *obs = (float *)(words + 20);
    oak_sincos(2.0f * OAK_PI * u01(g.y), &sn, &cs);
    planarTwin trig = planar_origin(planar_pair(state[1], state[0]));
    obs[4] = radius * cs;
    obs[5] = radius * sn;
    planar_fingertip(A, trig, obs + 4, &dx, &dy);
    planar_reacherSense(trig, state + 2, dx, dy, obs);
    planar_storeDoubles(state, words, 6);
    memcpy(words + 12, &trig, sizeof trig);
    words[30] = 0u;
    words[31] = resets + 1u;
}

static inline __attribute__((always_inline)) planarPair planar_reacherSubstep(const planarTable *P, const planarArmTable *R, const double *tau, double h, int last, const planarPair *theta, planarPair a0, double *q, double *v, double *w, planarTwin *b, planarTwin *tail) {
    double half = 0.5 * h, sixth = 1.0 / 6.0, third = 1.0 / 3.0, qs[2], v1[2], v2[2], v3[2], k0[2] = { a0.y, a0.x };
    planarPair quarter = (planarPair)(0.25 * h * h), square = (planarPair)(0.5 * h * h), early = (planarPair)(h * h / 6), late = (planarPair)(h * h / 12);
    for (int i = 0; i < 2; i++) {
        qs[i] = planar_mac(half, v[i], q[i]);
        v1[i] = planar_mac(half, k0[i], v[i]);
    }
    planarPair a1 = planar_reacherForward(R, qs, v1, b[1], tau, w);
    double k1[2] = { a1.y, a1.x };
    if (!last) {
        planarTwin fresh[2];
        planar_fresh(P, 2, theta, fresh);
        b[2] = fresh[0];
        b[3] = planar_compose(b[2], fresh[1]);
        b[4] = planar_compose(b[3], fresh[1]);
    }
    planarTwin ts = planar_turn(P, b[1], quarter * a0, planar_pair(planar_mac(half, v1[1], q[1]), planar_mac(half, v1[0], q[0])));
    planarPair e0 = planar_fma(early, a1, early * a0), e1 = planar_fma(early, a1, late * a0);
    for (int i = 0; i < 2; i++) {
        qs[i] = planar_mac(half, v1[i], q[i]);
        v2[i] = planar_mac(half, k1[i], v[i]);
    }
    planarPair a2 = planar_reacherForward(R, qs, v2, ts, tau, w);
    double k2[2] = { a2.y, a2.x };
    *tail = planar_turn(P, b[2], square * a1, planar_pair(planar_mac(h, v2[1], q[1]), planar_mac(h, v2[0], q[0])));
    e0 = planar_fma(early, a2, e0);
    e1 = planar_fma(early, a2, e1);
    for (int i = 0; i < 2; i++) {
        qs[i] = planar_mac(h, v2[i], q[i]);
        v3[i] = planar_mac(h, k2[i], v[i]);
    }
    planarPair a3 = planar_reacherForward(R, qs, v3, *tail, tau, w);
    double k3[2] = { a3.y, a3.x };
    for (int i = 0; i < 2; i++) {
        double dq = planar_mac(sixth, v3[i], planar_mac(third, v2[i], planar_mac(third, v1[i], sixth * v[i])));
        double dv = planar_mac(sixth, k3[i], planar_mac(third, k2[i], planar_mac(third, k1[i], sixth * k0[i])));
        q[i] = planar_mac(h, dq, q[i]);
        v[i] = planar_mac(h, dv, v[i]);
        w[i] = k3[i];
    }
    planarPair th = planar_pair(q[1], q[0]), om = planar_pair(v[1], v[0]);
    if (last) {
        b[0] = planar_turn(P, b[2], e0, th);
        return a3;
    }
    planarPair rest = planar_fma(late, a3, e1);
    b[0] = planar_turn(P, b[2], e0, th);
    b[1] = planar_turn(P, b[3], e0 + rest, planar_fma((planarPair)half, om, th));
    b[2] = planar_turn(P, b[4], planar_fma((planarPair)2, rest, e0), planar_fma((planarPair)h, om, th));
    return planar_reacherForward(R, q, v, b[0], tau, w);
}

__attribute__((noinline)) static float planar_reacher(const planarTable *P, uint32_t seed, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    const planarArm *A = (const planarArm *)opaque(&PLANAR_ARM);
    const planarArmTable *R = &planar_kit(P)->arm;
    double q[2], v[2], w[2], tau[2], cost = 0, dx, dy;
    planarTwin b[5], tail;
    planar_loadDoubles(words, q, 2);
    planar_loadDoubles(words + 4, v, 2);
    planar_loadDoubles(words + 8, w, 2);
    memcpy(&b[0], words + 12, sizeof b[0]);
    for (int u = 0; u < 2; u++) {
        tau[u] = planar_torque(ctrl[u], A->gear);
        cost = planar_mac((double)ctrl[u], (double)ctrl[u], cost);
    }
    double h = A->step;
    planarPair th = planar_pair(q[1], q[0]), om = planar_pair(v[1], v[0]), step = (planarPair)(0.5 * h) * om, theta[2] = { planar_fma((planarPair)h, om, th), step };
    planarPair a0 = planar_reacherForward(R, q, v, b[0], tau, w);
    b[1] = planar_turn(P, b[0], step, th + step);
    a0 = planar_reacherSubstep(P, R, tau, h, 0, theta, a0, q, v, w, b, &tail);
    planar_reacherSubstep(P, R, tau, h, 1, theta, a0, q, v, w, b, &tail);
    float *state = (float *)(words + 20);
    planar_fingertip(A, tail, state + 4, &dx, &dy);
    float reward = (float)(-__builtin_elementwise_sqrt(planar_mac(dx, dx, dy * dy)) - cost);
    planar_reacherSense(b[0], v, dx, dy, state);
    planar_storeDoubles(q, words, 2);
    planar_storeDoubles(v, words + 4, 2);
    planar_storeDoubles(w, words + 8, 2);
    memcpy(words + 12, &b[0], sizeof b[0]);
    words[30] += 1u;
    *done = words[30] >= PLANAR_REACHER_ROUND;
    if (*done) planar_reacherDraw(seed, words);
    memcpy(obs, words + 20, 10 * sizeof(float));
    return reward;
}

static void planar_pendulumSense(const double *q, const double *v, planarTwin t, double constraint, float *obs) {
    double c1 = t.c.x, s1 = t.s.x, c12 = t.c.y, s12 = t.s.y;
    obs[0] = (float)q[0];
    obs[1] = (float)s1;
    obs[2] = (float)planar_mac(s12, c1, -c12 * s1);
    obs[3] = (float)c1;
    obs[4] = (float)planar_mac(c12, c1, s12 * s1);
    for (int i = 0; i < 3; i++) obs[5 + i] = planar_clip(v[i]);
    obs[8] = planar_clip(constraint);
}

static void planar_pendulumDraw(uint32_t seed, uint32_t *words) {
    uint32_t resets = words[35];
    u44 r = philox4(seed, RNG_STREAM_RESET, (lane4)resets, (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
    float4 normal = normal4(r.x, r.y);
    double q[3], v[3], zero[3] = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) {
        q[i] = (double)uniformRange(r.z[i], -0.1f, 0.1f);
        v[i] = (double)(0.1f * normal[i]);
    }
    planarTwin trig = planar_origin(planar_pair(q[1], q[1] + q[2]));
    planar_storeDoubles(q, words, 3);
    planar_storeDoubles(v, words + 6, 3);
    planar_storeDoubles(zero, words + 12, 3);
    memcpy(words + 18, &trig, sizeof trig);
    planar_pendulumSense(q, v, trig, 0, (float *)(words + 26));
    words[35] = resets + 1u;
}

__attribute__((noinline)) static float planar_pendulum(const planarTable *P, uint32_t seed, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    const planarCart *C = (const planarCart *)opaque(&PLANAR_CART);
    double q[3], v[3], w[3], tau[3] = { planar_torque(ctrl[0], C->gear), 0, 0 };
    planarTwin trig;
    planar_loadDoubles(words, q, 3);
    planar_loadDoubles(words + 6, v, 3);
    planar_loadDoubles(words + 12, w, 3);
    memcpy(&trig, words + 18, sizeof trig);
    planarTail tail;
    planar_integrate(P, PLANAR_ROBOT_PENDULUM, &planar_kit(P)->cart, PLANAR_PENDULUM_SKIP, C->step, q, v, w, &trig, 1, tau, &tail);
    double pole = C->pole, x = planar_mac(pole, tail.trig[0].s.y, planar_mac(pole, tail.trig[0].s.x, tail.q[0])), y = planar_mac(pole, tail.trig[0].c.y, pole * tail.trig[0].c.x);
    int fallen = y <= 1;
    double reward = (fallen ? 0 : C->alive) - planar_mac(0.01 * x, x, (y - 2) * (y - 2)) - planar_mac(1e-3 * v[1], v[1], 5e-3 * v[2] * v[2]);
    planar_storeDoubles(q, words, 3);
    planar_storeDoubles(v, words + 6, 3);
    planar_storeDoubles(w, words + 12, 3);
    memcpy(words + 18, &trig, sizeof trig);
    planar_pendulumSense(q, v, trig, tail.constraint, (float *)(words + 26));
    *done = (uint32_t)fallen;
    if (fallen) planar_pendulumDraw(seed, words);
    memcpy(obs, words + 26, 9 * sizeof(float));
    return (float)reward;
}

static void planar_hopperSense(const uint32_t *words, float *obs) {
    double q[6], v[6];
    planar_loadDoubles(words, q, 6);
    planar_loadDoubles(words + 12, v, 6);
    for (int i = 0; i < 5; i++) obs[i] = (float)q[1 + i];
    for (int i = 0; i < 6; i++) obs[5 + i] = planar_clip(v[i]);
}

static void planar_hopperDraw(uint32_t seed, uint32_t *words) {
    const planarLeg *L = (const planarLeg *)opaque(&PLANAR_LEG);
    uint32_t resets = words[36];
    float noise = (float)L->noise;
    double state[18];
    for (uint32_t b = 0u; b < 3u; b++) {
        u4 r = philox4x32(seed, RNG_STREAM_RESET, resets, b, 0u, 0u);
        uint32_t raw[4] = { r.x, r.y, r.z, r.w };
        for (int i = 0; i < 4; i++) state[4 * b + i] = (double)uniformRange(raw[i], -noise, noise);
    }
    state[1] += L->rest;
    for (int i = 12; i < 18; i++) state[i] = 0;
    planar_storeDoubles(state, words, 18);
    words[36] = resets + 1u;
}

__attribute__((noinline)) static float planar_hopper(const planarTable *P, uint32_t seed, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    const planarLeg *L = (const planarLeg *)opaque(&PLANAR_LEG);
    double q[6], v[6], w[6], tau[3], cost = 0, start;
    planarTwin trig[2];
    planar_loadDoubles(words, q, 6);
    planar_loadDoubles(words + 12, v, 6);
    planar_loadDoubles(words + 24, w, 6);
    start = q[0];
    q[0] = 0;
    for (int u = 0; u < 3; u++) {
        tau[u] = planar_torque(ctrl[u], L->gear);
        cost = planar_mac((double)ctrl[u], (double)ctrl[u], cost);
    }
    planarTail tail;
    planar_integrate(P, PLANAR_ROBOT_HOPPER, &planar_kit(P)->leg, PLANAR_HOPPER_SKIP, L->step, q, v, w, trig, 0, tau, &tail);
    int healthy = q[1] > L->height && q[1] < (double)INFINITY && q[2] > -L->angle && q[2] < L->angle;
    for (int i = 2; i < 6; i++) healthy &= q[i] > -L->state && q[i] < L->state;
    for (int i = 0; i < 6; i++) healthy &= v[i] > -L->state && v[i] < L->state;
    double reward = q[0] / (L->step * PLANAR_HOPPER_SKIP) + (healthy ? L->alive : 0) - L->control * cost;
    q[0] += start;
    planar_storeDoubles(q, words, 6);
    planar_storeDoubles(v, words + 12, 6);
    planar_storeDoubles(w, words + 24, 6);
    *done = (uint32_t)!healthy;
    if (!healthy) planar_hopperDraw(seed, words);
    planar_hopperSense(words, obs);
    return (float)reward;
}

void planar_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)extra; (void)steps;
    switch (PLANAR_KNOBS[instance].robot) {
        case PLANAR_ROBOT_REACHER: planar_reacherDraw(seed, words); return;
        case PLANAR_ROBOT_PENDULUM: planar_pendulumDraw(seed, words); return;
        case PLANAR_ROBOT_HOPPER: planar_hopperDraw(seed, words); return;
        default: planar_draw(seed, words); return;
    }
}

static float planar_friction(uint32_t seed, uint32_t step, uint32_t period) {
    uint32_t regime = step / period;
    if (!regime) return PLANAR_FRICTION;
    float low = oak_log(PLANAR_FRICTION_LOW), span = oak_log(PLANAR_FRICTION_HIGH) - low;
    return oak_exp(fmaf(span, u01(philox4x32(seed, ENV_RNG_STREAM, regime, 0u, 0u, 0u).x), low));
}

__attribute__((noinline)) static float planar_cheetah(const planarTable *P, uint32_t period, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    const planarGripTable *grip = &P->grip;
    planarGrip slip;
    if (period) {
        planar_grip(planar_friction(seed, step, period), &slip);
        grip = &slip;
    }
    planarState s;
    planar_load(words, &s);
    planarVec tau;
    float cost = 0.0f;
    planarReal gear[PLANAR_JOINTS];
    for (int u = 0; u < PLANAR_JOINTS; u++) {
        gear[u] = (planarReal)PLANAR_GEAR[u] * (planarReal)fmaxf(-1.0f, fminf(1.0f, ctrl[u]));
        cost += PLANAR_CONTROL_COST * ctrl[u] * ctrl[u];
    }
    memset(&tau, 0, sizeof tau);
    for (int m = 0; m < 3; m++) tau.l[m] = planar_pair(gear[m], gear[3 + m]);
    double start = s.x, travel = 0.0;
    int torso = 0, cleared = 0, pending = 0, bad = planar_badState(&s) | !(fabs(s.x) <= (double)PLANAR_BAD);
    planarForward buffer[2];
    for (int k = 0; k < PLANAR_SUBSTEPS; k++) {
        int outcome = planar_substep(P, grip, &s, &tau, &travel, &buffer[k & 1], &buffer[(k + 1) & 1], &pending, &bad, k == PLANAR_SUBSTEPS - 1);
        torso |= outcome & 1;
        cleared |= outcome >> 1;
    }
    if (pending) planar_warm(P, &buffer[(PLANAR_SUBSTEPS - 1) & 1], &s);
    s.x = cleared ? travel : start + travel;
    float reward = (float)(cleared ? s.x - start : travel) / ((float)PLANAR_SUBSTEPS * PLANAR_TIMESTEP) - cost;
    planar_store(&s, words);
    if (torso) planar_draw(seed, words);
    *done = (uint32_t)torso;
    planar_obs(words, obs);
    return reward;
}

float planar_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)extra;
    const planarTable *P = (const planarTable *)shared;
    const planarKnobs *k = &PLANAR_KNOBS[instance];
    switch (k->robot) {
        case PLANAR_ROBOT_REACHER: return planar_reacher(P, seed, ctrl, words, obs, done);
        case PLANAR_ROBOT_PENDULUM: return planar_pendulum(P, seed, ctrl, words, obs, done);
        case PLANAR_ROBOT_HOPPER: return planar_hopper(P, seed, ctrl, words, obs, done);
        default: return planar_cheetah(P, k->frictionPeriod, seed, step, ctrl, words, obs, done);
    }
}

void planar_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    switch (PLANAR_KNOBS[instance].robot) {
        case PLANAR_ROBOT_REACHER: memcpy(obs, words + 20, 10 * sizeof(float)); return;
        case PLANAR_ROBOT_PENDULUM: memcpy(obs, words + 26, 9 * sizeof(float)); return;
        case PLANAR_ROBOT_HOPPER: planar_hopperSense(words, obs); return;
        default: planar_obs(words, obs); return;
    }
}

static void planar_reach(const uint32_t *words, float gain, float damping, float *ctrl) {
    const planarArm *A = (const planarArm *)opaque(&PLANAR_ARM);
    double v[2];
    planarTwin trig;
    planar_loadDoubles(words + 4, v, 2);
    memcpy(&trig, words + 12, sizeof trig);
    const float *goal = (const float *)(words + 20) + 4;
    double gx = planar_goal(A, goal, 0), gy = planar_goal(A, goal, 1), inner = A->link, outer = A->center[2];
    double c = fmin(fmax((gx * gx + gy * gy - inner * inner - outer * outer) / (2 * inner * outer), -1.0), 1.0), s = sqrt(1 - c * c) * (trig.s.x < 0 ? -1 : 1);
    double wx = planar_mac(outer, c, inner), wy = outer * s, norm = planar_mac(wx, wx, wy * wy);
    double c0 = planar_mac(gx, wx, gy * wy) / norm, s0 = planar_mac(wx, gy, -wy * gx) / norm;
    double e0 = planar_mac(s0, trig.c.y, -c0 * trig.s.y), e1 = planar_mac(s, trig.c.x, -c * trig.s.x);
    ctrl[0] = fmaxf(-1.0f, fminf(1.0f, (float)planar_mac((double)gain, e0, -(double)damping * v[0])));
    ctrl[1] = fmaxf(-1.0f, fminf(1.0f, (float)planar_mac((double)gain, e1, -(double)damping * v[1])));
}

static void planar_balance(const uint32_t *words, const float *gain, float *ctrl) {
    double s[6], u = 0;
    planar_loadDoubles(words, s, 3);
    planar_loadDoubles(words + 6, s + 3, 3);
    for (int i = 0; i < 6; i++) u = planar_mac(-(double)gain[i], s[i], u);
    ctrl[0] = fmaxf(-1.0f, fminf(1.0f, (float)u));
}

static void planar_hop(const float *obs, const float *weight, float *ctrl) {
    for (int a = 0; a < 3; a++) {
        float u = weight[33 + a];
        for (int i = 0; i < 11; i++) u = fmaf(weight[11 * a + i], obs[i], u);
        ctrl[a] = fmaxf(-1.0f, fminf(1.0f, u));
    }
}

void planar_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)extra;
    switch (PLANAR_KNOBS[instance].robot) {
        case PLANAR_ROBOT_REACHER: planar_reach(words, PLANAR_REACH[0], PLANAR_REACH[1], ctrl); return;
        case PLANAR_ROBOT_PENDULUM: planar_balance(words, PLANAR_BALANCE, ctrl); return;
        case PLANAR_ROBOT_HOPPER: planar_hop(obs, PLANAR_HOP, ctrl); return;
        default: break;
    }
    const planarTable *P = (const planarTable *)shared;
    float phase = 2.0f * OAK_PI * ((float)(step * PLANAR_GAIT_RATE % PLANAR_GAIT_CYCLE) / (float)PLANAR_GAIT_CYCLE), feature[PLANAR_GAIT_FEATURES];
    for (int k = 0; k < 17; k++) feature[k] = obs[k] * (k >= 8 ? PLANAR_GAIT_VELOCITY_SCALE : 1.0f);
    oak_sincos(phase, &feature[17], &feature[18]);
    oak_sincos(2.0f * phase, &feature[19], &feature[20]);
    feature[21] = 1.0f;
    float4 low = 0.0f, high = 0.0f;
    for (int k = 0; k < PLANAR_GAIT_FEATURES; k++) {
        low += P->gait[k][0] * feature[k];
        high += P->gait[k][1] * feature[k];
    }
    float u[8] = { low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w };
    for (int j = 0; j < PLANAR_GAIT_JOINTS; j++) ctrl[j] = fmaxf(-1.0f, fminf(1.0f, u[j]));
}
