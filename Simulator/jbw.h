#ifndef JBW_EXACT_H
#define JBW_EXACT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define JBW_EMPTY 0u
#define JBW_BANANA 1u
#define JBW_ONION 2u
#define JBW_JELLY 3u
#define JBW_WALL 4u
#define JBW_TREE 5u
#define JBW_TRUFFLE 6u
#define JBW_RESPONSE_RADIUS 17u
#define JBW_RESPONSE_CELLS (JBW_RESPONSE_RADIUS*(JBW_RESPONSE_RADIUS+1u)/2u)
#define JBW_BLOCK 16
#define JBW_ECO_BLOCK 64
#define JBW_STREAM_ITEMS 16u
#define JBW_STREAM_ECO 64u
#define JBW_STREAM_WALLS 80u
#define JBW_HISTORY_BITS 22
#define JBW_HISTORY_MASK ((1u<<JBW_HISTORY_BITS)-1u)
#define JBW_DESTRUCTION_LIFETIME 2000u
#define JBW_BLOCK_ITEMS 4u
#define ENV_RNG_STREAM 3u
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

typedef struct { uint32_t x,y,z,w; } u4;
static inline u4 philox4x32(uint32_t k0,uint32_t k1,u4 c){
    for(int r=0;r<10;r++){
        if(r>0){k0+=PHILOX_W0;k1+=PHILOX_W1;}
        uint32_t hi0=(uint32_t)(((uint64_t)PHILOX_M0*c.x)>>32), lo0=PHILOX_M0*c.x;
        uint32_t hi1=(uint32_t)(((uint64_t)PHILOX_M1*c.z)>>32), lo1=PHILOX_M1*c.z;
        u4 n={hi1^c.y^k0, lo1, hi0^c.w^k1, lo0};
        c=n;
    }
    return c;
}
static inline int jbw_floorDiv(int x,int d){int q=x/d;return q-((x<0&&x%d!=0)?1:0);}
typedef struct { uint32_t seed,stream,live; int x,y; u4 v; } jbw_memo;
static jbw_memo jbw_fieldMemo[256];
static inline u4 jbw_field(uint32_t seed,int x,int y,uint32_t stream){
    uint32_t h=((uint32_t)x*0x9E3779B9u)^((uint32_t)y*0x85EBCA6Bu)^(stream*0xC2B2AE35u);
    jbw_memo *e=&jbw_fieldMemo[(h^(h>>13))&255u];
    if(e->live&&e->x==x&&e->y==y&&e->stream==stream&&e->seed==seed) return e->v;
    u4 r=philox4x32(seed,ENV_RNG_STREAM,(u4){(uint32_t)x,(uint32_t)y,stream,0});
    e->seed=seed; e->stream=stream; e->x=x; e->y=y; e->live=1u; e->v=r;
    return r;
}
static inline void jbw_candidate(u4 b,int bx,int by,uint32_t slot,int*cx,int*cy){
    uint32_t bits=(b.x>>(8u*slot))&255u;
    *cx=bx*JBW_BLOCK+(int)(bits&15u); *cy=by*JBW_BLOCK+(int)((bits>>4u)&15u);
}
static inline uint32_t jbw_typeOf(u4 b,u4 eco,uint32_t slot){
    uint32_t roll=(b.y>>(8u*slot))&255u;
    if((eco.y&3u)==0u) return (roll&31u)==0u?JBW_TRUFFLE:JBW_TREE;
    uint32_t roll24=roll|(((b.z>>(8u*slot))&255u)<<8u)|(((b.w>>(8u*slot))&255u)<<16u);
    uint32_t pick=roll24%3u;
    return pick==0u?JBW_BANANA:pick==1u?JBW_JELLY:JBW_ONION;
}
static inline int jbw_wallHit(u4 w,int bx,int by,int x,int y){
    if((w.z&7u)!=0u) return 0;
    return (w.w&1u)==0u ? (y==by*JBW_BLOCK+(int)(w.y&15u)) : (x==bx*JBW_BLOCK+(int)(w.x&15u));
}
static const float JBW_ENERGY[7] = {0.0f,2.04f,1.68f,1.29f,1.34f,0.53f,0.79f};
static const float JBW_SCENT[7][3] = {
    {0.00f,0.00f,0.00f},{1.92f,1.76f,0.40f},{0.68f,0.01f,0.99f},{1.64f,0.54f,0.40f},
    {0.00f,0.00f,0.00f},{0.00f,0.47f,0.06f},{8.40f,4.80f,2.60f}};
typedef struct { int px,py; uint32_t item; float scent[3]; } jbw_blockItem;
typedef struct { uint32_t seed,live,count; int bx,by; jbw_blockItem item[JBW_BLOCK_ITEMS]; } jbw_blockEntry;
static jbw_blockEntry jbw_blockMemo[256];
static inline const jbw_blockEntry* jbw_blockItems(uint32_t seed,int bx,int by){
    uint32_t h=((uint32_t)bx*0x9E3779B9u)^((uint32_t)by*0x85EBCA6Bu);
    jbw_blockEntry *e=&jbw_blockMemo[(h^(h>>13))&255u];
    if(e->live&&e->bx==bx&&e->by==by&&e->seed==seed) return e;
    u4 block=jbw_field(seed,bx,by,JBW_STREAM_ITEMS);
    u4 wall =jbw_field(seed,bx,by,JBW_STREAM_WALLS);
    u4 eco  =jbw_field(seed,jbw_floorDiv(bx*JBW_BLOCK,JBW_ECO_BLOCK),jbw_floorDiv(by*JBW_BLOCK,JBW_ECO_BLOCK),JBW_STREAM_ECO);
    e->count=0u;
    for(uint32_t slot=0;slot<JBW_BLOCK_ITEMS;slot++){
        int px,py; jbw_candidate(block,bx,by,slot,&px,&py);
        int dup=0;
        for(uint32_t pr=0;pr<slot;pr++){int qx,qy;jbw_candidate(block,bx,by,pr,&qx,&qy);
            if(qx==px&&qy==py){dup=1;break;}}
        if(dup) continue;
        if(jbw_wallHit(wall,bx,by,px,py)) continue;
        jbw_blockItem *q=&e->item[e->count++];
        q->px=px; q->py=py; q->item=jbw_typeOf(block,eco,slot);
        q->scent[0]=JBW_SCENT[q->item][0]; q->scent[1]=JBW_SCENT[q->item][1]; q->scent[2]=JBW_SCENT[q->item][2];
    }
    e->seed=seed; e->bx=bx; e->by=by; e->live=1u;
    return e;
}

static inline uint32_t jbw_itemAt(uint32_t seed,int x,int y){
    int bx=jbw_floorDiv(x,JBW_BLOCK),by=jbw_floorDiv(y,JBW_BLOCK);
    if(jbw_wallHit(jbw_field(seed,bx,by,JBW_STREAM_WALLS),bx,by,x,y)) return JBW_WALL;
    const jbw_blockEntry *e=jbw_blockItems(seed,bx,by);
    for(uint32_t k=0;k<e->count;k++)
        if(e->item[k].px==x&&e->item[k].py==y) return e->item[k].item;
    return JBW_EMPTY;
}
static inline uint64_t jbw_key(int x,int y){return ((uint64_t)(uint32_t)x<<32)|(uint64_t)(uint32_t)y;}
static inline uint32_t* jbw_slotFor(uint32_t*h,uint32_t mask,int x,int y){
    uint64_t k=jbw_key(x,y), m=k;
    m^=m>>30; m*=0xbf58476d1ce4e5b9ull; m^=m>>27; m*=0x94d049bb133111ebull; m^=m>>31;
    for(uint32_t s=(uint32_t)m&mask,probe=0;;++probe,s=(s+1u)&mask){ uint32_t*e=h+3u*s;
        if(e[2]==0u||(e[0]==(uint32_t)(k>>32)&&e[1]==(uint32_t)k)) return e;
        if(probe>=mask){ fprintf(stderr,"jbw: destruction table full at %u slots\n",mask+1u); exit(1); } }
}
static inline uint32_t jbw_destroyedAt(const uint32_t*h,uint32_t mask,int x,int y){ return jbw_slotFor((uint32_t*)h,mask,x,y)[2]; }
static inline void jbw_destroy(uint32_t*h,uint32_t mask,int x,int y,uint32_t step){
    uint32_t*e=jbw_slotFor(h,mask,x,y); if(e[2]!=0u) return;
    uint64_t k=jbw_key(x,y); e[0]=(uint32_t)(k>>32); e[1]=(uint32_t)k; e[2]=step+1u;
}
static inline int jbw_blocks(uint32_t i){return i==JBW_WALL||i==JBW_TREE;}
static inline int jbw_collectible(uint32_t i){return i!=JBW_EMPTY&&!jbw_blocks(i);}
static inline float jbw_response(const float*t,uint32_t time,int dx,int dy){
    uint32_t x=(uint32_t)abs(dx),y=(uint32_t)abs(dy);
    if(x<y){uint32_t s=x;x=y;y=s;}
    return t[time*JBW_RESPONSE_CELLS + x*(x+1u)/2u + y];
}
static inline void jbw_buildResponse(float*r){
    size_t n=(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME;
    for(size_t i=0;i<n;i++) r[i]=0.0f;
    r[0]=1.0f;
    for(uint32_t time=1;time<JBW_DESTRUCTION_LIFETIME;time++){
        uint32_t prev=(time-1u)*JBW_RESPONSE_CELLS, cur=time*JBW_RESPONSE_CELLS;
        for(uint32_t i=0;i<JBW_RESPONSE_CELLS;i++) r[cur+i]=0.4f*r[prev+i];
        r[cur]+=1.0f;
        r[cur+JBW_RESPONSE_CELLS-1u]+=0.28f*r[prev+JBW_RESPONSE_CELLS-2u];
        for(int y=0;y<(int)JBW_RESPONSE_RADIUS-1;y++){
            r[cur+JBW_RESPONSE_CELLS-JBW_RESPONSE_RADIUS+(uint32_t)y]+=0.14f*(
                jbw_response(r,time-1u,(int)JBW_RESPONSE_RADIUS-2,y)
              + jbw_response(r,time-1u,(int)JBW_RESPONSE_RADIUS-1,y+1)
              + jbw_response(r,time-1u,(int)JBW_RESPONSE_RADIUS-1,y-1));
        }
        for(int x=0;x<(int)JBW_RESPONSE_RADIUS-1;x++)
            for(int y=0;y<=x;y++){
                uint32_t idx=(uint32_t)(x*(x+1)/2+y);
                r[cur+idx]+=0.14f*(
                    jbw_response(r,time-1u,x+1,y)+jbw_response(r,time-1u,x-1,y)
                  + jbw_response(r,time-1u,x,y+1)+jbw_response(r,time-1u,x,y-1));
            }
    }
}


static inline void jbw_scentAt(uint32_t seed,int x,int y,uint32_t step,
                               const uint32_t*hist,uint32_t mask,const float*resp,float*out){
    out[0]=out[1]=out[2]=0.0f;
    int reach=(int)JBW_RESPONSE_RADIUS-1;
    int bx0=jbw_floorDiv(x-reach,JBW_BLOCK),bx1=jbw_floorDiv(x+reach,JBW_BLOCK);
    int by0=jbw_floorDiv(y-reach,JBW_BLOCK),by1=jbw_floorDiv(y+reach,JBW_BLOCK);
    for(int by=by0;by<=by1;by++) for(int bx=bx0;bx<=bx1;bx++){
        const jbw_blockEntry *e=jbw_blockItems(seed,bx,by);
        for(uint32_t k=0;k<e->count;k++){
            const jbw_blockItem *q=&e->item[k];
            int dx=q->px-x,dy=q->py-y;
            if(abs(dx)>=(int)JBW_RESPONSE_RADIUS||abs(dy)>=(int)JBW_RESPONSE_RADIUS) continue;
            uint32_t del=jbw_collectible(q->item)?jbw_destroyedAt(hist,mask,q->px,q->py):0u;
            float value=jbw_response(resp,JBW_DESTRUCTION_LIFETIME-1u,dx,dy);
            if(del!=0u){
                uint32_t age=step-(del-1u);
                if(age>=JBW_DESTRUCTION_LIFETIME) value=0.0f;
                else value-=jbw_response(resp,age,dx,dy);
            }
            out[0]+=q->scent[0]*value; out[1]+=q->scent[1]*value; out[2]+=q->scent[2]*value;
        }
    }
}
#endif
