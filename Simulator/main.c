#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "jbw.h"
#include "agent.h"

void constructFeatures(Agent);
void predictAndControl(Agent);
void planAllOptions(Agent, uint32_t);
void learnFromTransition(Agent, const float*, const float*, uint32_t, uint32_t);
void writePrimaryPolicy(Agent);
void oakInit(Agent, uint32_t, float, float, float, float);
void oakSetStashDirty(int);
uint32_t oakSampleDiscrete(Agent, uint32_t, const uint32_t*);
void oakActionModelLearn(Agent, float);
void oakActionModelRemember(Agent, uint32_t);
void oakActionModelBiasScaleFree(Agent);
void oakStepSizeReport(Agent);
void oakDriftReport(Agent,int);
float oakValueNow(Agent);
float oakSlot0Q(Agent);
uint32_t oakSlot0Idx(Agent);
void oakModelQ(Agent,float*);

#define HBITS 22
#define HMASK ((1u<<HBITS)-1u)
#ifndef STEPS
#define STEPS 1000000
#endif
#define EPOCH 100000
#define HALVES (2*(STEPS/EPOCH))

static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
static const char *NAME[7]={"empty","banana","onion","jelly","wall","tree","truffle"};

int main(int argc, char **argv){
    uint32_t seed = (uint32_t)(argc>1?atoi(argv[1]):1);
    float *resp = malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    uint32_t *hist = calloc(HMASK+1u, 3u*sizeof(uint32_t));
    AgentParameters *P = calloc(1,sizeof(AgentParameters));
    AgentMemory     *M = calloc(1,sizeof(AgentMemory));
    AgentIndices    *I = calloc(1,sizeof(AgentIndices));
    Agent a; a.observationCount=12u; a.envActionCount=3u; a.stepCount=0u; a.stateSlot=0u;
    a.parameters=P; a.memory=M; a.indices=I;
#ifndef OAK_ALPHA_DENOM
#define OAK_ALPHA_DENOM (2u*OAK_PERCEPT_COUNT)
#endif
    float alpha = 1.0f/(float)(OAK_ALPHA_DENOM);
    oakInit(a, seed, 1.0f/sqrtf((float)OAK_FEATURE_CAPACITY), alpha, logf(alpha), 1.0f/3.0f);
    oakSetStashDirty(0);

    int px=0,py=0,hd=0,moved=0;
    float pendR=0.0f; int havePend=0;
    double total=0.0; long good=0,bad=0,fwd=0;
    long hg[HALVES]={0}, hb[HALVES]={0};
    double segR[8]={0}; long segN[8]={0};
    long ahead[7][2]={{0}};
    long blocked=0, wedgeRun=0, wedgeMax=0;
    double vSee=0,vAfter=0,qF=0,qT=0; long nSee=0,nAfter=0; int ateLast=0;
    double qFe=0; long nEmp=0;
    double gs[7]={0}, gq[7]={0}; long gn[7]={0};
    double s0q[7]={0}; long s0n[7]={0}; uint32_t s0i[7]={0};

    for (long t=0;t<STEPS;t++){
        for (int i=0;i<8;i++){
            int k=(2*hd+i)&7, qx=px+D8[k][0], qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy);
            int gone = !jbw_blocks(it) && jbw_destroyedAt(hist,HMASK,qx,qy)!=0u;
#ifdef OAK_NOVIEW
            M->observation[i] = 0.0f; (void)gone; (void)it;
#else
            M->observation[i] = gone?0.0f:jbw_energy(it);
#endif
        }
        float sc[3]; jbw_scentAt(seed,px,py,(uint32_t)t,hist,HMASK,resp,sc);
#ifdef OAK_NOSCENT
        M->observation[8]=0.0f; M->observation[9]=0.0f; M->observation[10]=0.0f; (void)sc;
#else
        M->observation[8]=sc[0]; M->observation[9]=sc[1]; M->observation[10]=sc[2];
#endif
        M->observation[11]=(float)moved;

        constructFeatures(a);
        predictAndControl(a);
        oakSetStashDirty(0);
        a.stepCount += 1u;
        a.stateSlot ^= 1u;
#ifdef OAK_ALIGN
        if (havePend) oakActionModelLearn(a, pendR);
#endif
        writePrimaryPolicy(a);
        oakActionModelBiasScaleFree(a);

        uint32_t w[8];
        u4 r0=philox4x32(seed,2u,(u4){(uint32_t)t,0,0,0});
        u4 r1=philox4x32(seed,2u,(u4){(uint32_t)t,1,0,0});
        w[0]=r0.x;w[1]=r0.y;w[2]=r0.z;w[3]=r0.w;w[4]=r1.x;w[5]=r1.y;w[6]=r1.z;w[7]=r1.w;
        uint32_t action = oakSampleDiscrete(a, 3u, w);
        oakActionModelRemember(a, action);

        if (ateLast){ vAfter += oakValueNow(a); nAfter++; ateLast=0; }
        int tx=px+FW[hd][0], ty=py+FW[hd][1];
        uint32_t it=jbw_itemAt(seed,tx,ty);
        if (it==0u){ float qq[3]; oakModelQ(a,qq); qFe += qq[0]; nEmp++; }
        if (it<7u && jbw_destroyedAt(hist,HMASK,tx,ty)==0u){
            ahead[it][action==0u?0:1]++;
            s0q[it]+=oakSlot0Q(a); s0n[it]++; s0i[it]=oakSlot0Idx(a);
            float lt = M->policyMeanRaw[1]>M->policyMeanRaw[2]?M->policyMeanRaw[1]:M->policyMeanRaw[2];

            if (it==JBW_JELLY){ float qq[3]; oakModelQ(a,qq);
                vSee += oakValueNow(a); qF += qq[0]; qT += (qq[1]>qq[2]?qq[1]:qq[2]); nSee++; }
            float g = M->policyMeanRaw[0]-lt;
#ifdef OAK_NOFLIP
            float sign = 1.0f;
#else
            float sign = ((t/EPOCH)&1)==0 ? 1.0f : -1.0f;
#endif
            g *= (it==JBW_ONION||it==JBW_JELLY) ? sign : 1.0f;
            gs[it]+=g; gq[it]+=(double)g*g; gn[it]++;
        }
        if (action==0u && jbw_blocks(it)){ blocked++; if(++wedgeRun>wedgeMax) wedgeMax=wedgeRun; }
        else wedgeRun=0;

        float reward=0.0f; moved=0;
        if (action==1u) hd=(hd+3)&3;
        else if (action==2u) hd=(hd+1)&3;
        else if (!jbw_blocks(it)){
            px=tx; py=ty; moved=1;
            if (jbw_collectible(it) && jbw_destroyedAt(hist,HMASK,tx,ty)==0u)
                jbw_destroy(hist,HMASK,tx,ty,(uint32_t)t);
        }
        if (action==0u && moved && jbw_destroyedAt(hist,HMASK,px,py)==(uint32_t)t+1u){
#ifdef OAK_NOFLIP
            float sign = 1.0f;
#else
            float sign = ((t/EPOCH)&1)==0 ? 1.0f : -1.0f;
#endif
            if (it==JBW_JELLY){ reward=sign; ateLast=1; } else if (it==JBW_ONION) reward=-sign;
        }
        M->reward = reward;
#ifdef OAK_ALIGN
        pendR = reward; havePend = 1;
#else
        oakActionModelLearn(a, reward);
#endif

        if (action==0u) fwd++;
        total += reward;
        { int sg=(int)(t*8/STEPS); if(sg>7)sg=7; segR[sg]+=reward; segN[sg]++;
          if(t%(STEPS/8)==0) oakDriftReport(a,sg); }
        long h = t/(STEPS/HALVES); if (h>=HALVES) h=HALVES-1;
        if (reward>0){ good++; hg[h]++; } else if (reward<0){ bad++; hb[h]++; }

        uint32_t previous = 1u - a.stateSlot;
        planAllOptions(a, previous);
        learnFromTransition(a, featureValues(a,previous), featureValues(a,1u-previous), previous, 3u);
    }

    oakStepSizeReport(a);
    printf("  V(see jelly) %.4f over %ld   V(just ate) %.4f over %ld   delta %.4f\n",
           nSee?vSee/nSee:0.0, nSee, nAfter?vAfter/nAfter:0.0, nAfter,
           (nSee?vSee/nSee:0.0)-(nAfter?vAfter/nAfter:0.0));
    printf("  Q(fwd | empty ahead) %.4f over %ld\n", nEmp?qFe/nEmp:0.0, nEmp);
    printf("  model Q(fwd) %.4f   Q(best turn) %.4f   diff %.4f   (want diff > 0)\n",
           nSee?qF/nSee:0.0, nSee?qT/nSee:0.0, nSee?(qF-qT)/nSee:0.0);
    printf("  by eighth    ");
    for(int i=0;i<8;i++) printf(" %+.6f", segN[i]?segR[i]/segN[i]:0.0);
    printf("\n");
    printf("  mean reward   %+.6f\n", total/(double)STEPS);
    printf("  eaten         %ld good / %ld bad  (%.1f%% correct)\n",
           good,bad,(good+bad)?100.0*good/(good+bad):0.0);
    printf("  %%forward      %.1f%%\n", 100.0*fwd/STEPS);
    printf("  P(forward|ahead):");
    for (int i=2;i<=5;i++){ long n=ahead[i][0]+ahead[i][1];
        printf("  %s %.0f%%(%ld)", NAME[i], n?100.0*ahead[i][0]/n:0.0, n); }
    printf("\n  logit(fwd)-logit(turn), polarity-corrected:\n");
    for (int i=2;i<=5;i++){ if(!gn[i])continue;
        double mu=gs[i]/gn[i], sd=sqrt(gq[i]/gn[i]-mu*mu);
        printf("    %-6s mean %+7.3f  sd %6.3f  |mean|/sd %.3f\n", NAME[i], mu, sd, sd>0?fabs(mu)/sd:0.0); }
    printf("  slot0 (obs[0] tile) contribution to Q(fwd), by item ahead:\n");
    for (int i=2;i<=5;i++) if(s0n[i]) printf("    %-6s feature#%-5u  contrib %+.4f\n", NAME[i], s0i[i], s0q[i]/s0n[i]);
    printf("  blocked %ld (%.1f%%)  longest wedge %ld\n", blocked, 100.0*blocked/STEPS, wedgeMax);
    double e=0,l=0; int en=0,ln=0;
    printf("  tracking     ");
    for (int i=0;i<HALVES;i+=2){
        long g1=hg[i],b1=hb[i],g2=hg[i+1],b2=hb[i+1];
        double c1=(g1+b1)?100.0*g1/(g1+b1):-1, c2=(g2+b2)?100.0*g2/(g2+b2):-1;
        if(c1>=0){e+=c1;en++;} if(c2>=0){l+=c2;ln++;}
        printf(" %.0f->%.0f",c1,c2);
    }
    printf("\n  adaptation    %.1f%% -> %.1f%%  (%+.1f pts)\n",
           en?e/en:0.0, ln?l/ln:0.0, (ln?l/ln:0.0)-(en?e/en:0.0));
    return 0;
}
