#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "jbw.h"
#include "agent.h"

#ifndef STEPS
#define STEPS 1000000
#endif
#define EPOCH 100000

static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};

int main(int argc, char **argv){
    uint32_t seed = (uint32_t)(argc>1?atoi(argv[1]):1);
    int policy = argc>2 ? atoi(argv[2]) : -1;
    float *resp = malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    size_t cap=(size_t)STEPS*JBW_BLOCK_ITEMS*2u/((size_t)JBW_BLOCK*JBW_BLOCK)+2u, slots=2u; while(slots<cap) slots<<=1;
    uint32_t hmask=(uint32_t)(slots-1u); uint32_t *hist=calloc(slots,3u*sizeof(uint32_t));
    AgentParameters *P = calloc(1,sizeof(AgentParameters));
    AgentMemory     *M = calloc(1,sizeof(AgentMemory));
    AgentIndices    *I = calloc(1,sizeof(AgentIndices));
    Agent a; a.observationCount=12u; a.envActionCount=3u; a.stepCount=0u; a.stateSlot=0u;
    a.parameters=P; a.memory=M; a.indices=I;
    oakInit(a, seed, 1.0f/3.0f);

    int px=0,py=0,hd=0,moved=0;
    double total=0.0; long good=0,bad=0,fwd=0,mv=0;
    double segR[8]={0}; long segN[8]={0};

    for (long t=0;t<STEPS;t++){
        uint32_t view[8], fwdItem=JBW_EMPTY, fwdDel=0u;
        for (int i=0;i<8;i++){
            int k=(2*hd+i)&7, qx=px+D8[k][0], qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy), del=jbw_collectible(it)?jbw_destroyedAt(hist,hmask,qx,qy):0u;
            if(!i){ fwdItem=it; fwdDel=del; }
            view[i] = del!=0u ? JBW_EMPTY : it;
        }
        float sc[3]; jbw_scentAt(seed,px,py,(uint32_t)t,hist,hmask,resp,sc);
        float sign = ((t/EPOCH)&1)==0 ? 1.0f : -1.0f;
        uint32_t action;

        if (policy < 0){
            for (int i=0;i<8;i++) M->observation[i] = JBW_ENERGY[view[i]];
            M->observation[8]=sc[0]; M->observation[9]=sc[1]; M->observation[10]=sc[2];
            M->observation[11]=(float)moved;
            constructFeatures(a);
            predictAndControl(a);
            a.stepCount += 1u;
            a.stateSlot ^= 1u;
            oakActionModelBiasScaleFree(a);
            uint32_t w[8];
            u4 r0=philox4x32(seed,2u,(u4){(uint32_t)t,0,0,0}), r1=philox4x32(seed,2u,(u4){(uint32_t)t,1,0,0});
            w[0]=r0.x;w[1]=r0.y;w[2]=r0.z;w[3]=r0.w;w[4]=r1.x;w[5]=r1.y;w[6]=r1.z;w[7]=r1.w;
            action = oakSampleDiscrete(a, 3u, w);
            oakActionModelRemember(a, action);
        } else {
            uint32_t good = sign>0?JBW_JELLY:JBW_ONION, bad = sign>0?JBW_ONION:JBW_JELLY;
            u4 rr = philox4x32(seed,2u,(u4){(uint32_t)t,7,0,0});
            uint32_t ahead = view[0];
            if (policy == 0) action = rr.x%3u;
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
        if (policy < 0) oakActionModelLearn(a, reward);

        if (action==0u) fwd++;
        mv += moved;
        total += reward;
        { int sg=(int)(t*8/STEPS); if(sg>7)sg=7; segR[sg]+=reward; segN[sg]++; }
        if (reward>0) good++; else if (reward<0) bad++;

        if (policy < 0) learnFromTransition(a, 1u - a.stateSlot);
    }

    printf("  by eighth    ");
    for(int i=0;i<8;i++) printf(" %+.6f", segN[i]?segR[i]/segN[i]:0.0);
    printf("\n  mean reward   %+.17g\n", total/(double)STEPS);
    printf("  eaten         %ld good / %ld bad  (%.1f%% correct)\n",
           good,bad,(good+bad)?100.0*good/(good+bad):0.0);
    printf("  %%forward      %.1f%%\n  moves/step    %.4f\n", 100.0*fwd/STEPS, (double)mv/STEPS);
    return 0;
}
