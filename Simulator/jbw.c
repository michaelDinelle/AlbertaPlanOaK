#include <stdlib.h>
#include <stdio.h>
#include "jbw.h"

#define BLOCK_MEMO_SLOTS 256u
#define BLOCK_MEMO_MASK (BLOCK_MEMO_SLOTS - 1u)
#define BLOCK_HASH_X 0x9E3779B9u
#define BLOCK_HASH_Y 0x85EBCA6Bu

#define FOREST_BLOCK_MASK 3u
#define TRUFFLE_AMONG_TREES_MASK 31u
#define WALL_BLOCK_MASK 7u
#define BLOCK_COORDINATE_MASK 15u
#define ITEM_KINDS_OUTSIDE_FOREST 3u

#define DESTRUCTION_TABLE_WORDS 3u
#define DESTRUCTION_EMPTY 0u

#define JBW_SCENT_DECAY 0.4f
#define JBW_SCENT_DIFFUSION 0.14f

static inline int floorDiv(int value, int divisor){
    int quotient = value / divisor;
    if (value < 0 && value % divisor != 0) quotient -= 1;
    return quotient;
}

static inline uint32_t byteOfSlot(uint32_t word, uint32_t slot){
    return (word >> (8u * slot)) & 255u;
}

static inline u4 blockField(uint32_t seed, int blockX, int blockY, uint32_t stream){
    return philox4x32(seed, RNG_KEY_ENVIRONMENT,
                      (u4){ (uint32_t)blockX, (uint32_t)blockY, stream, 0u });
}

static inline void candidateCell(u4 items, int blockX, int blockY, uint32_t slot,
                                 int *cellX, int *cellY){
    uint32_t packed = byteOfSlot(items.x, slot);
    *cellX = blockX * JBW_BLOCK + (int)(packed & BLOCK_COORDINATE_MASK);
    *cellY = blockY * JBW_BLOCK + (int)((packed >> 4u) & BLOCK_COORDINATE_MASK);
}

static inline int blockIsForest(u4 eco){
    return (eco.y & FOREST_BLOCK_MASK) == 0u;
}

static inline uint32_t itemKind(u4 items, u4 eco, uint32_t slot){
    uint32_t roll = byteOfSlot(items.y, slot);
    if (blockIsForest(eco))
        return (roll & TRUFFLE_AMONG_TREES_MASK) == 0u ? JBW_TRUFFLE : JBW_TREE;

    uint32_t wideRoll = roll
                      | (byteOfSlot(items.z, slot) << 8u)
                      | (byteOfSlot(items.w, slot) << 16u);
    uint32_t pick = wideRoll % ITEM_KINDS_OUTSIDE_FOREST;
    if (pick == 0u) return JBW_BANANA;
    if (pick == 1u) return JBW_JELLY;
    return JBW_ONION;
}

static inline int blockHasWall(u4 wall){
    return (wall.z & WALL_BLOCK_MASK) == 0u;
}

static inline int wallCoversCell(u4 wall, int blockX, int blockY, int x, int y){
    if (!blockHasWall(wall)) return 0;
    if ((wall.w & 1u) == 0u)
        return y == blockY * JBW_BLOCK + (int)(wall.y & BLOCK_COORDINATE_MASK);
    return x == blockX * JBW_BLOCK + (int)(wall.x & BLOCK_COORDINATE_MASK);
}

typedef struct {
    int x, y;
    uint32_t item, destroyedAt;
    const float *scent;
} BlockItem;

typedef struct {
    uint32_t seed, live, count, destructionFresh;
    int blockX, blockY;
    u4 wall;
    BlockItem item[JBW_BLOCK_ITEMS];
} BlockEntry;

static BlockEntry blockMemo[BLOCK_MEMO_SLOTS];

static inline BlockEntry *blockAt(uint32_t seed, int blockX, int blockY){
    uint32_t hash = ((uint32_t)blockX * BLOCK_HASH_X) ^ ((uint32_t)blockY * BLOCK_HASH_Y);
    BlockEntry *entry = &blockMemo[(hash ^ (hash >> 13)) & BLOCK_MEMO_MASK];
    if (entry->live && entry->blockX == blockX && entry->blockY == blockY && entry->seed == seed)
        return entry;

    u4 items = blockField(seed, blockX, blockY, JBW_STREAM_ITEMS);
    u4 wall  = blockField(seed, blockX, blockY, JBW_STREAM_WALLS);
    u4 eco   = blockField(seed,
                          floorDiv(blockX * JBW_BLOCK, JBW_ECO_BLOCK),
                          floorDiv(blockY * JBW_BLOCK, JBW_ECO_BLOCK),
                          JBW_STREAM_ECO);

    int candidateX[JBW_BLOCK_ITEMS], candidateY[JBW_BLOCK_ITEMS];
    for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; ++slot)
        candidateCell(items, blockX, blockY, slot, &candidateX[slot], &candidateY[slot]);

    entry->count = 0u;
    for (uint32_t slot = 0u; slot < JBW_BLOCK_ITEMS; ++slot){
        int x = candidateX[slot];
        int y = candidateY[slot];

        int duplicate = 0;
        for (uint32_t earlier = 0u; earlier < slot; ++earlier)
            if (candidateX[earlier] == x && candidateY[earlier] == y){
                duplicate = 1;
                break;
            }
        if (duplicate) continue;
        if (wallCoversCell(wall, blockX, blockY, x, y)) continue;

        BlockItem *placed = &entry->item[entry->count++];
        placed->x = x;
        placed->y = y;
        placed->item = itemKind(items, eco, slot);
        placed->scent = JBW_SCENT[placed->item];
    }

    entry->seed = seed;
    entry->blockX = blockX;
    entry->blockY = blockY;
    entry->wall = wall;
    entry->live = 1u;
    entry->destructionFresh = 0u;
    return entry;
}

uint32_t jbw_itemAt(uint32_t seed, int x, int y){
    int blockX = floorDiv(x, JBW_BLOCK);
    int blockY = floorDiv(y, JBW_BLOCK);
    const BlockEntry *entry = blockAt(seed, blockX, blockY);

    if (wallCoversCell(entry->wall, blockX, blockY, x, y)) return JBW_WALL;
    for (uint32_t index = 0u; index < entry->count; ++index)
        if (entry->item[index].x == x && entry->item[index].y == y)
            return entry->item[index].item;
    return JBW_EMPTY;
}

int jbw_blocks(uint32_t item){
    return item == JBW_WALL || item == JBW_TREE;
}

int jbw_collectible(uint32_t item){
    return item != JBW_EMPTY && !jbw_blocks(item);
}

static inline uint64_t cellKey(int x, int y){
    return ((uint64_t)(uint32_t)x << 32) | (uint64_t)(uint32_t)y;
}

static inline uint32_t *destructionSlot(uint32_t *table, uint32_t mask, int x, int y){
    uint64_t key = cellKey(x, y);
    uint64_t scrambled = key;
    scrambled ^= scrambled >> 30;
    scrambled *= 0xbf58476d1ce4e5b9ull;
    scrambled ^= scrambled >> 27;
    scrambled *= 0x94d049bb133111ebull;
    scrambled ^= scrambled >> 31;

    uint32_t slot = (uint32_t)scrambled & mask;
    for (uint32_t probe = 0u; ; ++probe, slot = (slot + 1u) & mask){
        uint32_t *entry = table + DESTRUCTION_TABLE_WORDS * slot;
        if (entry[2] == DESTRUCTION_EMPTY) return entry;
        if (entry[0] == (uint32_t)(key >> 32) && entry[1] == (uint32_t)key) return entry;
        if (probe >= mask){
            fprintf(stderr, "jbw: destruction table full at %u slots\n", mask + 1u);
            exit(1);
        }
    }
}

uint32_t jbw_destroyedAt(const uint32_t *table, uint32_t mask, int x, int y){
    return destructionSlot((uint32_t *)table, mask, x, y)[2];
}

void jbw_destroy(uint32_t *table, uint32_t mask, int x, int y, uint32_t step){
    uint32_t *entry = destructionSlot(table, mask, x, y);
    if (entry[2] != DESTRUCTION_EMPTY) return;

    uint64_t key = cellKey(x, y);
    entry[0] = (uint32_t)(key >> 32);
    entry[1] = (uint32_t)key;
    entry[2] = step + 1u;

    for (uint32_t slot = 0u; slot < BLOCK_MEMO_SLOTS; ++slot)
        blockMemo[slot].destructionFresh = 0u;
}

static inline float scentResponse(const float *response, uint32_t age, int dx, int dy){
    uint32_t major = (uint32_t)abs(dx);
    uint32_t minor = (uint32_t)abs(dy);
    if (major < minor){
        uint32_t swap = major;
        major = minor;
        minor = swap;
    }
    return response[age * JBW_SCENT_CELLS + major * (major + 1u) / 2u + minor];
}

void jbw_buildScentResponse(float *response){
    size_t cells = (size_t)JBW_SCENT_CELLS * JBW_DESTRUCTION_LIFETIME;
    for (size_t index = 0; index < cells; ++index) response[index] = 0.0f;
    response[0] = 1.0f;

    for (uint32_t age = 1u; age < JBW_DESTRUCTION_LIFETIME; ++age){
        uint32_t previous = (age - 1u) * JBW_SCENT_CELLS;
        uint32_t current = age * JBW_SCENT_CELLS;

        for (uint32_t index = 0u; index < JBW_SCENT_CELLS; ++index)
            response[current + index] = JBW_SCENT_DECAY * response[previous + index];
        response[current] += 1.0f;

        response[current + JBW_SCENT_CELLS - 1u]
            += (2.0f * JBW_SCENT_DIFFUSION) * response[previous + JBW_SCENT_CELLS - 2u];

        for (int minor = 0; minor < (int)JBW_SCENT_RADIUS - 1; ++minor)
            response[current + JBW_SCENT_CELLS - JBW_SCENT_RADIUS + (uint32_t)minor]
                += JBW_SCENT_DIFFUSION * (
                       scentResponse(response, age - 1u, (int)JBW_SCENT_RADIUS - 2, minor)
                     + scentResponse(response, age - 1u, (int)JBW_SCENT_RADIUS - 1, minor + 1)
                     + scentResponse(response, age - 1u, (int)JBW_SCENT_RADIUS - 1, minor - 1));

        for (int major = 0; major < (int)JBW_SCENT_RADIUS - 1; ++major)
            for (int minor = 0; minor <= major; ++minor){
                uint32_t index = (uint32_t)(major * (major + 1) / 2 + minor);
                response[current + index] += JBW_SCENT_DIFFUSION * (
                      scentResponse(response, age - 1u, major + 1, minor)
                    + scentResponse(response, age - 1u, major - 1, minor)
                    + scentResponse(response, age - 1u, major, minor + 1)
                    + scentResponse(response, age - 1u, major, minor - 1));
            }
    }
}

void jbw_scentAt(uint32_t seed, int x, int y, uint32_t step,
                 const uint32_t *table, uint32_t mask,
                 const float *response, float *out){
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;

    int reach = (int)JBW_SCENT_RADIUS - 1;
    int firstBlockX = floorDiv(x - reach, JBW_BLOCK), lastBlockX = floorDiv(x + reach, JBW_BLOCK);
    int firstBlockY = floorDiv(y - reach, JBW_BLOCK), lastBlockY = floorDiv(y + reach, JBW_BLOCK);

    for (int blockY = firstBlockY; blockY <= lastBlockY; ++blockY)
        for (int blockX = firstBlockX; blockX <= lastBlockX; ++blockX){
            BlockEntry *entry = blockAt(seed, blockX, blockY);

            if (!entry->destructionFresh){
                for (uint32_t index = 0u; index < entry->count; ++index)
                    entry->item[index].destroyedAt =
                        jbw_collectible(entry->item[index].item)
                        ? jbw_destroyedAt(table, mask, entry->item[index].x, entry->item[index].y)
                        : DESTRUCTION_EMPTY;
                entry->destructionFresh = 1u;
            }

            for (uint32_t index = 0u; index < entry->count; ++index){
                const BlockItem *item = &entry->item[index];
                int dx = item->x - x;
                int dy = item->y - y;
                if (abs(dx) >= (int)JBW_SCENT_RADIUS || abs(dy) >= (int)JBW_SCENT_RADIUS) continue;

                float strength = scentResponse(response, JBW_DESTRUCTION_LIFETIME - 1u, dx, dy);
                if (item->destroyedAt != DESTRUCTION_EMPTY){
                    uint32_t age = step - (item->destroyedAt - 1u);
                    if (age >= JBW_DESTRUCTION_LIFETIME) strength = 0.0f;
                    else strength -= scentResponse(response, age, dx, dy);
                }

                out[0] += item->scent[0] * strength;
                out[1] += item->scent[1] * strength;
                out[2] += item->scent[2] * strength;
            }
        }
}
