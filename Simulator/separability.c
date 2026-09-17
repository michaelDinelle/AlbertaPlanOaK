#include "jbw.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifndef OAK_CANDIDATES_PER_SLOT
#define OAK_CANDIDATES_PER_SLOT 7u
#endif
#define OAK_TILE_LAST (OAK_CANDIDATES_PER_SLOT-1u)
#define OAK_TILE_HALF (0.5f*(float)OAK_CANDIDATES_PER_SLOT)
static float clampf_(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}
static uint32_t wrapT(float f,uint32_t m){ if(!isfinite(f))return 0u; double q=fmod((double)f,(double)m); if(q<0)q+=m; return (uint32_t)q; }
static uint32_t tileClamp(float a,uint32_t z){ float zoom=(float)(1u<<(z<2u?z:2u));
    return (uint32_t)clampf_(floorf(a*zoom+OAK_TILE_HALF),0.0f,(float)OAK_TILE_LAST); }
static uint32_t tileWrap(float a,uint32_t z){ float zoom=ldexpf(1.0f,(int)z);
    return wrapT(floorf(a*zoom+OAK_TILE_HALF),OAK_CANDIDATES_PER_SLOT); }
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
#define NZ 7
int main(void){
    uint32_t seed=1;
    float *resp=malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    uint32_t HMASK=(1u<<20)-1u; uint32_t *hist=calloc(HMASK+1u,3u*sizeof(uint32_t));
    double mean=0.0, m2=0.0; long n=0;
    int px=0,py=0,hd=0;
    long seen[7]={0};
    unsigned char codeC[7][NZ][64]; memset(codeC,0,sizeof codeC);
    unsigned char codeW[7][NZ][64]; memset(codeW,0,sizeof codeW);
    for(long t=0;t<200000;t++){
        uint32_t ahead=jbw_itemAt(seed,px+FW[hd][0],py+FW[hd][1]);
        float e=jbw_energy(ahead);
        n++; double d=e-mean; mean+=d/n; m2+=d*(e-mean);
        double sd=sqrt(m2/n); if(sd<1e-12) sd=1e-12;
        float a=(float)((e-mean)/sd);
        if(ahead<7u&&n>150000){ seen[ahead]++;
            for(uint32_t z=0;z<NZ;z++){
                codeC[ahead][z][tileClamp(a,z)]=1;
                codeW[ahead][z][tileWrap(a,z)]=1; } }
        u4 rr=philox4x32(seed,2u,(u4){(uint32_t)t,9,0,0});
        uint32_t act=rr.x%3u;
        if(act==1u)hd=(hd+3)&3; else if(act==2u)hd=(hd+1)&3;
        else{int nx=px+FW[hd][0],ny=py+FW[hd][1];
             if(!jbw_blocks(jbw_itemAt(seed,nx,ny))){px=nx;py=ny;
                if(jbw_collectible(jbw_itemAt(seed,nx,ny))&&jbw_destroyedAt(hist,HMASK,nx,ny)==0u)
                    jbw_destroy(hist,HMASK,nx,ny,(uint32_t)t);} }
    }
    const char*NM[7]={"empty","banana","onion","jelly","wall","tree","truffle"};
    printf("obs[0] tile SET per item type, zoom 0..%d   (mean %.4f sd %.4f)\n",NZ-1,mean,sqrt(m2/n));
    printf("%-8s | %-30s | %-30s\n","type","CLAMPED  (tiles per zoom)","WRAPPED  (tiles per zoom)");
    for(int i=0;i<7;i++){ if(!seen[i])continue;
        printf("%-8s | ",NM[i]);
        for(int z=0;z<NZ;z++){ for(int k=0;k<(int)OAK_CANDIDATES_PER_SLOT;k++) if(codeC[i][z][k]) printf("%d",k);
                               printf(z==NZ-1?"":"."); }
        printf(" | ");
        for(int z=0;z<NZ;z++){ for(int k=0;k<(int)OAK_CANDIDATES_PER_SLOT;k++) if(codeW[i][z][k]) printf("%d",k);
                               printf(z==NZ-1?"":"."); }
        printf("\n"); }
    int cc=0,cw=0,pairs=0;
    for(int i=1;i<7;i++)for(int j=i+1;j<7;j++){ if(!seen[i]||!seen[j])continue; pairs++;
        int dc=0,dw=0;
        for(int z=0;z<NZ;z++){ int oc=0,ow=0;
            for(int k=0;k<(int)OAK_CANDIDATES_PER_SLOT;k++){ if(codeC[i][z][k]&&codeC[j][z][k])oc=1;
                                  if(codeW[i][z][k]&&codeW[j][z][k])ow=1; }
            if(!oc)dc=1; if(!ow)dw=1; }
        cc+=dc; cw+=dw;
        if(!dc||!dw) printf("   collide: %-7s vs %-7s   clamped=%s wrapped=%s\n",
                            NM[i],NM[j],dc?"sep":"COLLIDE",dw?"sep":"COLLIDE"); }
    printf("\nitem-type pairs separable at some zoom:  CLAMPED %d/%d    WRAPPED %d/%d\n",cc,pairs,cw,pairs);
    return 0;
}
