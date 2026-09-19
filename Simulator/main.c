#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include "jbw.h"
#include "agent.h"

#ifndef STEPS
#define STEPS 1000000
#endif
#define SEEDS 8
#define RUNGS 5

static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
static const long RUNG[RUNGS]={1000000L,4000000L,16000000L,64000000L,256000000L};
typedef struct { double rung[RUNGS]; long good,bad,fwd,mv; } Result;

static void runSeed(uint32_t seed, int policy, Result *out){
    size_t cap=(size_t)STEPS*JBW_BLOCK_ITEMS*2u/((size_t)JBW_BLOCK*JBW_BLOCK)+2u, slots=2u; while(slots<cap) slots<<=1;
    uint32_t hmask=(uint32_t)(slots-1u); uint32_t *hist=calloc(slots,3u*sizeof(uint32_t));
    AgentMemory *M = calloc(1,sizeof(AgentMemory));
    Agent a; a.observationCount=9u; a.envActionCount=3u; a.memory=M;
    int px=0,py=0,hd=0,moved=0,rk=0; long drought=0,record=0; double total=0.0;
    for (long t=0;t<STEPS;t++){
        uint32_t view[8], fwdItem=JBW_EMPTY, fwdDel=0u;
        for (int i=0;i<8;i++){
            int k=(2*hd+i)&7, qx=px+D8[k][0], qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy), del=jbw_collectible(it)?jbw_destroyedAt(hist,hmask,qx,qy):0u;
            if(!i){ fwdItem=it; fwdDel=del; }
            view[i] = del!=0u ? JBW_EMPTY : it;
        }
        float sign = ((t/100000L)&1)==0 ? 1.0f : -1.0f;
        uint32_t action;
        if (policy < 0){
            for (int i=0;i<8;i++) M->observation[i] = JBW_ENERGY[view[i]];
            M->observation[8]=(float)moved;
            constructFeatures(a);
            oakActionModelBiasScaleFree(a);
            u4 r0=philox4x32(seed,2u,(u4){(uint32_t)t,0,0,0});
            uint32_t w[3]={r0.x,r0.y,r0.z};
            action = oakSampleDiscrete(a, 3u, w);
            oakActionModelRemember(a, action);
        } else {
            uint32_t good = sign>0?JBW_JELLY:JBW_ONION, bad = sign>0?JBW_ONION:JBW_JELLY;
            u4 rr = philox4x32(seed,2u,(u4){(uint32_t)t,7,0,0});
            uint32_t ahead = view[0];
            if (policy == 0 || drought > record) action = rr.x%3u;
            else if (ahead == good) action = 0u;
            else if (ahead == bad || ahead == JBW_WALL || ahead == JBW_TREE) action = 1u+(rr.x&1u);
            else {
                int seen = -1;
                for (int i=1;i<8;i++) if (view[i]==good){ seen=i; break; }
                action = (seen==2) ? 2u : (seen==6) ? 1u
                       : (seen>=3 && seen<=5) ? ((seen<5)?2u:1u) : 0u;
            }
        }
        int tx=px+FW[hd][0], ty=py+FW[hd][1]; uint32_t it=fwdItem;
        float reward=0.0f; moved=0;
        if (action==1u) hd=(hd+3)&3;
        else if (action==2u) hd=(hd+1)&3;
        else if (!jbw_blocks(it)){
            px=tx; py=ty; moved=1;
            if (jbw_collectible(it) && fwdDel==0u){
                jbw_destroy(hist,hmask,tx,ty,(uint32_t)t);
                if (it==JBW_JELLY) reward=sign; else if (it==JBW_ONION) reward=-sign;
            }
        }
        if (reward!=0.0f){ if(drought>record) record=drought; drought=0; } else drought++;
        if (policy < 0) oakActionModelLearn(a, reward);
        if (action==0u) out->fwd++;
        out->mv += moved; total += reward;
        if (reward>0) out->good++; else if (reward<0) out->bad++;
        if (rk<RUNGS && t+1==RUNG[rk]){ out->rung[rk]=total/(double)RUNG[rk]; rk++; }
    }
}

int main(int argc, char **argv){
    int policy = argc>1 ? atoi(argv[1]) : -1; struct timespec c0,c1;
    Result *res = mmap(NULL, SEEDS*sizeof(Result), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANON, -1, 0); clock_gettime(CLOCK_MONOTONIC,&c0);
    for (int i=0;i<SEEDS;i++) if (fork()==0){ runSeed(101u+(uint32_t)i, policy, &res[i]); _exit(0); }
    while (wait(NULL) > 0) {} clock_gettime(CLOCK_MONOTONIC,&c1);
    long g=0,b=0,f=0,m=0; for(int i=0;i<SEEDS;i++){ g+=res[i].good; b+=res[i].bad; f+=res[i].fwd; m+=res[i].mv; }
    for (int k=0;k<RUNGS && RUNG[k]<=(long)STEPS;k++){
        double s=0,w=res[0].rung[k]; for(int i=0;i<SEEDS;i++){ s+=res[i].rung[k]; if(res[i].rung[k]<w) w=res[i].rung[k]; }
        printf("  %4ldM  mean %+.9f  worst %+.9f\n", RUNG[k]/1000000L, s/SEEDS, w);
    }
    double sec=(c1.tv_sec-c0.tv_sec)+1e-9*(c1.tv_nsec-c0.tv_nsec);
    printf("  eaten %ld good / %ld bad (%.1f%% correct)   %%forward %.1f%%   moves/step %.4f   %.2fM SPS/lane   %.1fM SPS\n", g,b,(g+b)?100.0*g/(g+b):0.0, 100.0*f/((double)STEPS*SEEDS), (double)m/((double)STEPS*SEEDS), STEPS/sec/1e6, (double)STEPS*SEEDS/sec/1e6);
    return 0;
}
