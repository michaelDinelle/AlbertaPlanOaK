#ifndef OAK_GEOMETRY_H
#define OAK_GEOMETRY_H
#include "env.h"

#define OAK_CAPACITY 64u

_Static_assert(OAK_CAPACITY % 4u == 0u, "");

#define OAK_TRACE_LEVELS 7u
#define OAK_ACTION_PARAMS (2u * OAK_ENV_ACT_MAX)
#define OAK_POLICY_MAX OAK_ENV_ACT_MAX
#define OAK_OBSERVATION_LIMIT OAK_ENV_OBS_MAX
#define OAK_UNITS (OAK_CAPACITY)
#define OAK_FANIN (32u)
_Static_assert(OAK_FANIN % 4u == 0u, "");
_Static_assert(OAK_UNITS % 8u == 0u, "");
#define OAK_PROJECTED_COUNT(obs) ((obs) + OAK_ACTION_PARAMS + 2u)
#define OAK_PROJECTED(obs) (((OAK_PROJECTED_COUNT(obs) + 3u) / 4u) * 4u)
#define OAK_INPUTS(obs) (((OAK_PROJECTED(obs) + (obs) * OAK_TRACE_LEVELS + 7u) / 8u) * 8u)
#define OAK_PROJECTED_MAX OAK_PROJECTED(OAK_OBSERVATION_LIMIT)
#define OAK_INPUT_MAX OAK_INPUTS(OAK_OBSERVATION_LIMIT)
#define OAK_FEATURE_MAX (OAK_INPUT_MAX + OAK_UNITS)
#define OAK_FEATURE_TIMESCALE (OAK_CAPACITY * (1u + OAK_TRACE_LEVELS))
#define OAK_ELIGIBILITY_DECAY (1.0f - 1.0f / (float)(1u << 1u))
#define OAK_TABLE_BITS 14u
#define OAK_TABLE (1u << OAK_TABLE_BITS)
#define OAK_TRACKED_TUPLES (2u * (1u + OAK_TRACE_LEVELS))
#define OAK_TUPLES (OAK_TRACKED_TUPLES + OAK_TRACE_LEVELS)
#define OAK_CELLS (OAK_TUPLES + 2u * OAK_OBSERVATION_LIMIT)
#define OAK_CELL_STRIDE 8u
#define OAK_HEADS 3u
#define OAK_KEYS 2u
#define OAK_TABLE_ARMS 5u
#define OAK_ARMS (OAK_HEADS + OAK_TABLE_ARMS)
_Static_assert(OAK_ARMS <= 8u, "");
#define OAK_QTABLE_BITS 14u
#define OAK_QTABLE (1u << OAK_QTABLE_BITS)
#define OAK_QROW (2u * OAK_CELL_STRIDE)
#define OAK_PLAN_BUDGET OAK_FEATURE_TIMESCALE
#define OAK_BLOCK OAK_FEATURE_TIMESCALE
_Static_assert(OAK_CELL_STRIDE > OAK_POLICY_MAX, "");
#define OAK_ACTOR_ALPHA_DENOM ((float)(OAK_ACTION_PARAMS * OAK_CAPACITY))

typedef struct {
    uint32_t projected, inputs, features;
    float trackingRate, metaRate, ridgeFloor;
} OakGeometry;

static inline OakGeometry oak_geometry(uint32_t obs) {
    OakGeometry g;
    g.projected = OAK_PROJECTED(obs);
    g.inputs = OAK_INPUTS(obs);
    g.features = g.inputs + OAK_UNITS;
    g.trackingRate = 1.0f / (float)OAK_FEATURE_TIMESCALE;
    g.metaRate = g.trackingRate / (float)(OAK_FEATURE_TIMESCALE + OAK_ACTION_PARAMS);
    g.ridgeFloor = 1.0f / (float)(2u * OAK_CAPACITY);
    return g;
}

#endif
