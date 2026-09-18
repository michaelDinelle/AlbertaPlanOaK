#ifndef JBW_EXACT_H
#define JBW_EXACT_H
#include <stdint.h>

#define JBW_EMPTY 0u
#define JBW_BANANA 1u
#define JBW_ONION 2u
#define JBW_JELLY 3u
#define JBW_WALL 4u
#define JBW_TREE 5u
#define JBW_TRUFFLE 6u
#define JBW_TYPES (JBW_TRUFFLE+1u)
#define JBW_BLOCK 16
#define JBW_ECO_BLOCK 64
#define JBW_BLOCK_ITEMS 4u
#define JBW_STREAM_ITEMS 16u
#define JBW_STREAM_ECO 64u
#define JBW_STREAM_WALLS 80u
#define JBW_RESPONSE_RADIUS 17u
#define JBW_RESPONSE_CELLS (JBW_RESPONSE_RADIUS*(JBW_RESPONSE_RADIUS+1u)/2u)
#define JBW_DESTRUCTION_LIFETIME 2000u
#define ENV_RNG_STREAM 3u
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

typedef struct { uint32_t x,y,z,w; } u4;
static const float JBW_ENERGY[JBW_TYPES] = {0.0f,2.04f,1.68f,1.29f,1.34f,0.53f,0.79f};
static const float JBW_SCENT[JBW_TYPES][3] = {
    {0.00f,0.00f,0.00f},{1.92f,1.76f,0.40f},{0.68f,0.01f,0.99f},{1.64f,0.54f,0.40f},
    {0.00f,0.00f,0.00f},{0.00f,0.47f,0.06f},{8.40f,4.80f,2.60f}};

u4 philox4x32(uint32_t k0,uint32_t k1,u4 c);
uint32_t jbw_itemAt(uint32_t seed,int x,int y);
uint32_t jbw_destroyedAt(const uint32_t*h,uint32_t mask,int x,int y);
void jbw_destroy(uint32_t*h,uint32_t mask,int x,int y,uint32_t step);
int jbw_blocks(uint32_t i);
int jbw_collectible(uint32_t i);
void jbw_buildResponse(float*r);
void jbw_scentAt(uint32_t seed,int x,int y,uint32_t step,
                 const uint32_t*hist,uint32_t mask,const float*resp,float*out);
#endif
