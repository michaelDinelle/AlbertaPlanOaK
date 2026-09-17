#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "jbw.h"
#include "agent.h"

#define HBITS 22
#define HMASK ((1u<<HBITS)-1u)
#ifndef STEPS
#define STEPS 1000000
#endif
#define EPOCH 100000
#define HALVES (2*(STEPS/EPOCH))

static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};

int main(int argc, char **argv){
    uint32_t seed = (uint32_t)(argc>1?atoi(argv[1]):1);
    int policy = argc>2 ? atoi(argv[2]) : -1;
    float *resp = malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    uint32_t *hist = calloc(HMASK+1u, 3u*sizeof(uint32_t));
    AgentParameters *P = calloc(1,sizeof(AgentParameters));
    AgentMemory     *M = calloc(1,sizeof(AgentMemory));
    AgentIndices    *I = calloc(1,sizeof(AgentIndices));
    Agent a; a.observationCount=12u; a.envActionCount=3u; a.stepCount=0u; a.stateSlot=0u;
    a.parameters=P; a.memory=M; a.indices=I;
    float alpha = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
    oakInit(a, seed, 1.0f/sqrtf((float)OAK_FEATURE_CAPACITY), alpha, logf(alpha), 1.0f/3.0f);

    int px=0,py=0,hd=0,moved=0;
    float pj=0.0f,po=0.0f;
    double total=0.0; long good=0,bad=0,fwd=0;
    long hg[HALVES]={0}, hb[HALVES]={0};
    double segR[8]={0}; long segN[8]={0};

    for (long t=0;t<STEPS;t++){
        uint32_t view[8];
        for (int i=0;i<8;i++){
            int k=(2*hd+i)&7, qx=px+D8[k][0], qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy);
            view[i] = (!jbw_blocks(it) && jbw_destroyedAt(hist,HMASK,qx,qy)!=0u) ? JBW_EMPTY : it;
        }
        float sc[3]; jbw_scentAt(seed,px,py,(uint32_t)t,hist,HMASK,resp,sc);
        float sign = ((t/EPOCH)&1)==0 ? 1.0f : -1.0f;
        uint32_t action;

        if (policy < 0){
            for (int i=0;i<8;i++) M->observation[i] = jbw_energy(view[i]);
            M->observation[8]=sc[0]; M->observation[9]=sc[1]; M->observation[10]=sc[2];
            M->observation[11]=(float)moved;
            constructFeatures(a);
            predictAndControl(a);
            a.stepCount += 1u;
            a.stateSlot ^= 1u;
            writePrimaryPolicy(a);
            oakActionModelBiasScaleFree(a);
            uint32_t w[8];
            u4 r0=philox4x32(seed,2u,(u4){(uint32_t)t,0,0,0});
            u4 r1=philox4x32(seed,2u,(u4){(uint32_t)t,1,0,0});
            w[0]=r0.x;w[1]=r0.y;w[2]=r0.z;w[3]=r0.w;w[4]=r1.x;w[5]=r1.y;w[6]=r1.z;w[7]=r1.w;
            action = oakSampleDiscrete(a, 3u, w);
            oakActionModelRemember(a, action);
        } else {
            float gs = sign>0?sc[2]:sc[1], bs = sign>0?sc[1]:sc[2];
            uint32_t GOODI = sign>0?JBW_JELLY:JBW_ONION, BADI = sign>0?JBW_ONION:JBW_JELLY;
            u4 rr=philox4x32(seed,2u,(u4){(uint32_t)t,7,0,0});
            switch(policy){
            case 0: action=0u; break;
            case 1: action=rr.x%3u; break;
            case 2: case 3: case 4: {
                uint32_t ah=view[0];
                if(ah==GOODI){action=0u;break;}
                if(ah==BADI||ah==JBW_WALL||ah==JBW_TREE){action=1u+(rr.x&1u);break;}
                int found=-1;
                for(int i=1;i<8;i++) if(view[i]==GOODI){found=i;break;}
                if(found==2){action=2u;break;}
                if(found==6){action=1u;break;}
                if(found==1||found==7){action=0u;break;}
                if(found>=3&&found<=5){action=(found<5)?2u:1u;break;}
                if(policy==2){action=0u;break;}
                if(policy==3){
                    if(!moved){action=0u;break;}
                    action=((gs-pj)-(bs-po)>=0.0f)?0u:(1u+(rr.x&1u));
                    break;
                }
                {   float best=-1e30f; uint32_t ba=0u; int any=0;
                    for(uint32_t A=0;A<3;A++){
                        int nh=hd,nx=px,ny=py;
                        if(A==1u)nh=(hd+3)&3; else if(A==2u)nh=(hd+1)&3;
                        else { int tx=px+FW[hd][0],ty=py+FW[hd][1];
                               if(jbw_blocks(jbw_itemAt(seed,tx,ty))) continue;
                               nx=tx;ny=ty; }
                        int lx=nx+FW[nh][0], ly=ny+FW[nh][1];
                        if(jbw_blocks(jbw_itemAt(seed,lx,ly))) continue;
                        float s2[3]; jbw_scentAt(seed,lx,ly,(uint32_t)t+1u,hist,HMASK,resp,s2);
                        float g2=sign>0?s2[2]:s2[1], b2=sign>0?s2[1]:s2[2];
                        float v=g2-b2;
                        if(!any||v>best){best=v;ba=A;any=1;}
                    }
                    action=any?ba:(1u+(rr.x&1u));
                } } break;
            case 5: action = jbw_blocks(view[0]) ? (1u+(rr.x&1u)) : 0u; break;
            case 6: action = (jbw_blocks(view[0])||view[0]==BADI) ? (1u+(rr.x&1u)) : 0u; break;
            case 7: { uint32_t ah=view[0];
                if(ah==BADI||ah==JBW_WALL||ah==JBW_TREE){action=1u+(rr.x&1u);break;}
                int found=-1;
                for(int i=1;i<8;i++) if(view[i]==GOODI){found=i;break;}
                if(found==2){action=2u;break;}
                if(found==6){action=1u;break;}
                action=0u; } break;
            default: action=0u;
            }
            pj=gs; po=bs;
        }

        int tx=px+FW[hd][0], ty=py+FW[hd][1];
        uint32_t it=jbw_itemAt(seed,tx,ty);
        float reward=0.0f; moved=0;
        if (action==1u) hd=(hd+3)&3;
        else if (action==2u) hd=(hd+1)&3;
        else if (!jbw_blocks(it)){
            px=tx; py=ty; moved=1;
            if (jbw_collectible(it) && jbw_destroyedAt(hist,HMASK,tx,ty)==0u)
                jbw_destroy(hist,HMASK,tx,ty,(uint32_t)t);
        }
        if (action==0u && moved && jbw_destroyedAt(hist,HMASK,px,py)==(uint32_t)t+1u){
            if (it==JBW_JELLY) reward=sign; else if (it==JBW_ONION) reward=-sign;
        }
        if (policy < 0){
            M->reward = reward;
            oakActionModelLearn(a, reward);
        }

        if (action==0u) fwd++;
        total += reward;
        { int sg=(int)(t*8/STEPS); if(sg>7)sg=7; segR[sg]+=reward; segN[sg]++; }
        long h = t/(STEPS/HALVES); if (h>=HALVES) h=HALVES-1;
        if (reward>0){ good++; hg[h]++; } else if (reward<0){ bad++; hb[h]++; }

        if (policy < 0){
            uint32_t previous = 1u - a.stateSlot;
            planAllOptions(a, previous);
            learnFromTransition(a, previous);
        }
    }

    printf("  by eighth    ");
    for(int i=0;i<8;i++) printf(" %+.6f", segN[i]?segR[i]/segN[i]:0.0);
    printf("\n  mean reward   %+.17g\n", total/(double)STEPS);
    printf("  eaten         %ld good / %ld bad  (%.1f%% correct)\n",
           good,bad,(good+bad)?100.0*good/(good+bad):0.0);
    printf("  %%forward      %.1f%%\n", 100.0*fwd/STEPS);
    double e=0,l=0; int en=0,ln=0;
    for (int i=0;i<HALVES;i+=2){
        long g1=hg[i],b1=hb[i],g2=hg[i+1],b2=hb[i+1];
        double c1=(g1+b1)?100.0*g1/(g1+b1):-1, c2=(g2+b2)?100.0*g2/(g2+b2):-1;
        if(c1>=0){e+=c1;en++;} if(c2>=0){l+=c2;ln++;}
    }
    printf("  adaptation    %.1f%% -> %.1f%%  (%+.1f pts)\n",
           en?e/en:0.0, ln?l/ln:0.0, (ln?l/ln:0.0)-(en?e/en:0.0));
    return 0;
}
