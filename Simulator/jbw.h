#ifndef JBW_H
#define JBW_H
#include <stdint.h>
#include "rng.h"

#define JBW_EMPTY 0u
#define JBW_BANANA 1u
#define JBW_ONION 2u
#define JBW_JELLY 3u
#define JBW_WALL 4u
#define JBW_TREE 5u
#define JBW_TRUFFLE 6u
#define JBW_TYPES (JBW_TRUFFLE + 1u)

#define JBW_BLOCK 16
#define JBW_ECO_BLOCK 64
#define JBW_BLOCK_ITEMS 4u

#define JBW_STREAM_ITEMS 16u
#define JBW_STREAM_ECO 64u
#define JBW_STREAM_WALLS 80u

#define JBW_SCENT_RADIUS 17u
#define JBW_SCENT_CELLS (JBW_SCENT_RADIUS * (JBW_SCENT_RADIUS + 1u) / 2u)
#define JBW_DESTRUCTION_LIFETIME 2000u

static const float JBW_ENERGY[JBW_TYPES] = {
    0.00f, 2.04f, 1.68f, 1.29f, 1.34f, 0.53f, 0.79f
};

static const float JBW_SCENT[JBW_TYPES][3] = {
    { 0.00f, 0.00f, 0.00f },
    { 1.92f, 1.76f, 0.40f },
    { 0.68f, 0.01f, 0.99f },
    { 1.64f, 0.54f, 0.40f },
    { 0.00f, 0.00f, 0.00f },
    { 0.00f, 0.47f, 0.06f },
    { 8.40f, 4.80f, 2.60f }
};

static const float JBW_COLOR[JBW_TYPES][3] = {
    { 0.00f, 0.00f, 0.00f },
    { 0.96f, 0.88f, 0.20f },
    { 0.68f, 0.01f, 0.99f },
    { 0.82f, 0.27f, 0.20f },
    { 0.20f, 0.47f, 0.67f },
    { 0.00f, 0.47f, 0.06f },
    { 0.42f, 0.24f, 0.13f }
};

uint32_t jbw_itemAt(uint32_t seed, int x, int y);
int      jbw_blocks(uint32_t item);
int      jbw_collectible(uint32_t item);

uint32_t jbw_destroyedAt(const uint32_t *table, uint32_t mask, int x, int y);
void     jbw_destroy(uint32_t *table, uint32_t mask, int x, int y, uint32_t step);

void jbw_buildScentResponse(float *response);
void jbw_scentAt(uint32_t seed, int x, int y, uint32_t step,
                 const uint32_t *table, uint32_t mask,
                 const float *response, float *out);

#endif
