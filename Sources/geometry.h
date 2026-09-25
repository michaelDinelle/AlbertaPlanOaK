#ifndef OAK_GEOMETRY_H
#define OAK_GEOMETRY_H
#include "env.h"

#if defined(OAK_PRESET_BASE)
#define OAK_CAPACITY 32u
#define OAK_PRESET_NAME "base"
#elif defined(OAK_PRESET_PRO)
#define OAK_CAPACITY 64u
#define OAK_PRESET_NAME "pro"
#elif defined(OAK_PRESET_MAX)
#define OAK_CAPACITY 128u
#define OAK_PRESET_NAME "max"
#elif defined(OAK_PRESET_ULTRA)
#define OAK_CAPACITY 160u
#define OAK_PRESET_NAME "ultra"
#else
#error "Choose a geometry preset: base, pro, max, or ultra"
#endif

_Static_assert(OAK_CAPACITY % 32u == 0u, "");

#define OAK_CANDIDATES_PER_SLOT 7u
#define OAK_CONSTRUCTED_PER_PERCEPT 2u
#define OAK_TRACE_LEVELS OAK_CANDIDATES_PER_SLOT
#define OAK_FEATURES_PER_PERCEPT (1u + OAK_CONSTRUCTED_PER_PERCEPT * OAK_CANDIDATES_PER_SLOT)
#define OAK_ACTIVE_PER_PERCEPT (1u + OAK_CONSTRUCTED_PER_PERCEPT)
#define OAK_TIMESCALE_PER_PERCEPT (1u + OAK_CANDIDATES_PER_SLOT)

#define OAK_ACTION_PARAMS (2u * OAK_ENV_ACT_MAX)
#define OAK_POLICY_MAX OAK_ENV_ACT_MAX
#define OAK_OBSERVATION_CAPACITY OAK_CAPACITY
#define OAK_OBSERVATION_LIMIT (OAK_ENV_OBS_MAX < OAK_OBSERVATION_CAPACITY ? OAK_ENV_OBS_MAX : OAK_OBSERVATION_CAPACITY)
#define OAK_SOURCE_MAX (OAK_OBSERVATION_CAPACITY + OAK_ACTION_PARAMS + 2u)
#define OAK_PERCEPT_MAX (((OAK_SOURCE_MAX + OAK_CAPACITY - 1u) / OAK_CAPACITY) * OAK_CAPACITY)
#define OAK_FEATURE_CAPACITY (OAK_PERCEPT_MAX * OAK_FEATURES_PER_PERCEPT)
#define OAK_ACTIVE_CAPACITY (OAK_PERCEPT_MAX * OAK_ACTIVE_PER_PERCEPT)
#define OAK_TRACE_CAPACITY (OAK_TRACE_LEVELS * OAK_OBSERVATION_CAPACITY)
#define OAK_FEATURE_STATES (2u * (OAK_FEATURE_CAPACITY + OAK_TRACE_CAPACITY))
#define OAK_POLICY_BANK (OAK_FEATURE_CAPACITY * OAK_POLICY_MAX)
#define OAK_ACTIVE_STATES (2u * OAK_ACTIVE_CAPACITY)

#define OAK_ACTOR_ALPHA_DENOM ((float)(OAK_ACTION_PARAMS * OAK_CAPACITY))
#define OAK_SAMPLE_SIGMA (1.0f / (float)OAK_ACTION_PARAMS)

typedef struct {
    uint32_t percepts, featureCapacity, activeCapacity, featureTimescale, traceChannels;
    float trackingRate, metaRate, ridgeFloor;
} OakGeometry;

static inline OakGeometry oak_geometry(uint32_t obs) {
    OakGeometry g;
    uint32_t sources = obs + OAK_ACTION_PARAMS + 2u;
    g.percepts = ((sources + OAK_CAPACITY - 1u) / OAK_CAPACITY) * OAK_CAPACITY;
    g.featureCapacity = g.percepts * OAK_FEATURES_PER_PERCEPT;
    g.activeCapacity = g.percepts * OAK_ACTIVE_PER_PERCEPT;
    g.featureTimescale = g.percepts * OAK_TIMESCALE_PER_PERCEPT;
    g.traceChannels = OAK_TRACE_LEVELS * obs;
    g.trackingRate = 1.0f / (float)g.featureTimescale;
    g.metaRate = g.trackingRate / (float)(g.featureTimescale + OAK_ACTION_PARAMS);
    g.ridgeFloor = 1.0f / (float)(2u * g.percepts);
    return g;
}

#endif
