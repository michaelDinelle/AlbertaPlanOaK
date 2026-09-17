#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jbw.h"
#include "agent.h"
void constructFeatures(Agent);
void predictAndControl(Agent);
void planAllOptions(Agent, uint32_t);
void learnFromTransition(Agent, const float*, const float*, uint32_t, uint32_t);
void emitOptionActions(Agent, int);
void writePrimaryPolicy(Agent);
void updateUtilities(Agent);
void oakInit(Agent, uint32_t, float, float, float, float);
uint32_t oakSampleDiscrete(Agent, uint32_t, const uint32_t*);
void oakSetStashDirty(int);
void oakActionModelLearn(Agent,float);
void oakActionModelRemember(Agent,uint32_t);
void oakActionModelBias(Agent,float);
void oakActionModelBiasScaleFree(Agent);
float oakActionQForward(Agent);
void replaceWorstFeature(Agent);
void updateNursery(Agent);

#define HBITS 22
#define HMASK ((1u<<HBITS)-1u)
#ifndef STEPS
#define STEPS 200000
#endif
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};

static double probeV[7][10], probeA[7][10], probeQ[7][10]; static long probeN[7][10];
int main(int argc,char**argv){
    uint32_t seed = argc>1 ? (uint32_t)atoi(argv[1]) : 1024u;
    float *resp = malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    uint32_t *hist = calloc(3u*(HMASK+1u), sizeof(uint32_t));

    AgentParameters *P = calloc(1,sizeof(AgentParameters));
    AgentMemory     *M = calloc(1,sizeof(AgentMemory));
    AgentIndices    *I = calloc(1,sizeof(AgentIndices));
    Agent a; a.observationCount=12u; a.envActionCount=3u; a.stepCount=0u; a.stateSlot=0u;
    a.parameters=P; a.memory=M; a.indices=I;
    float alpha = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
    oakInit(a, seed, 1.0f/sqrtf((float)OAK_FEATURE_CAPACITY), alpha, logf(alpha), 1.0f/3.0f);
    oakSetStashDirty(0);

    int px=0,py=0,hd=0,moved=0;
#ifdef OAK_LINPROBE
    static float lw[OAK_FEATURE_CAPACITY]; static uint32_t lf[OAK_ACTIVE_CAPACITY]; static float lv[OAK_ACTIVE_CAPACITY];
    double lse=0.0, lsy=0.0; long lhit=0, lsign=0, lnz=0;
#endif
    long refuse[7][2]={{0}}; long wedge=0, wedgeMax=0, wedgeRun=0;
    uint32_t committed=3u; long commitRuns=0,commitSteps=0; float pendingReward=0.0f; int havePending=0;
    long steer[9][3]; memset(steer,0,sizeof steer);
    double tot=0; long G=0,B=0,fw=0;
    long epG[10]={0}, epB[10]={0};
    long halfG[20]={0}, halfB[20]={0};
    for (long t=0;t<STEPS;t++){

        for (int i=0;i<8;i++){
            int k=(2*hd+i)&7; int qx=px+D8[k][0], qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy);
            int absent = !jbw_blocks(it) && jbw_destroyedAt(hist,HMASK,qx,qy)!=0u;
            M->observation[i] = absent?0.0f:jbw_energy(it);
        }
        float sc[3]; jbw_scentAt(seed,px,py,(uint32_t)t,hist,HMASK,resp,sc);
        M->observation[8]=sc[0]; M->observation[9]=sc[1]; M->observation[10]=sc[2];
        M->observation[11]=(float)moved;

        {
            uint32_t ah=jbw_itemAt(seed,px+FW[hd][0],py+FW[hd][1]);
            if (jbw_destroyedAt(hist,HMASK,px+FW[hd][0],py+FW[hd][1])==0u && ah<7u){
                const float *cf=featureValues(a,a.stateSlot);
                const uint32_t *ca=activeFeatures(a,a.stateSlot);
                float v=0.0f;
                for (uint32_t q=0;q<OAK_ACTIVE_CAPACITY;++q) v+=P->valueWeight[ca[q]]*cf[ca[q]];
                int ep=(int)(t/(STEPS/10)); if(ep>9)ep=9;
                probeV[ah][ep]+=v; probeN[ah][ep]++;
                probeQ[ah][ep]+=oakActionQForward(a);
                float lf=M->policyMeanRaw[0], l1=M->policyMeanRaw[1], l2=M->policyMeanRaw[2];
                float turn = l1>l2?l1:l2;
                probeA[ah][ep]+=(lf-turn);
            }
        }
        constructFeatures(a);
        predictAndControl(a);
        oakSetStashDirty(0);
        replaceWorstFeature(a);
        a.stepCount += 1u;
        a.stateSlot ^= 1u;
#ifdef OAK_ALIGN
        if (havePending) oakActionModelLearn(a, pendingReward);
#endif
        emitOptionActions(a, 1);
        writePrimaryPolicy(a);
#ifdef OAK_SCALEFREE
        oakActionModelBiasScaleFree(a);
#elif defined(OAK_ACTION_MODEL)
        oakActionModelBias(a, OAK_ACTION_MODEL);
#endif
        uint32_t words[8];
        u4 r0=philox4x32(seed, 2u, (u4){(uint32_t)t,0,0,0});
        u4 r1=philox4x32(seed, 2u, (u4){(uint32_t)t,1,0,0});
        words[0]=r0.x;words[1]=r0.y;words[2]=r0.z;words[3]=r0.w;
        words[4]=r1.x;words[5]=r1.y;words[6]=r1.z;words[7]=r1.w;
        uint32_t action = oakSampleDiscrete(a, 3u, words);
#ifdef OAK_COMMIT
        {   float m=0.0f; for(int d=0;d<3;d++) m+=M->policyMeanRaw[d]; m/=3.0f;
            uint32_t am=0u; for(uint32_t d=1;d<3;d++) if(M->policyMeanRaw[d]>M->policyMeanRaw[am]) am=d;
            (void)m;
            if(committed<3u && committed==am){ action=committed; commitSteps++; }
            else { committed=am; action=am; commitRuns++; }
        }
#endif
#if defined(OAK_ACTION_MODEL) || defined(OAK_SCALEFREE)
        oakActionModelRemember(a, action);
#ifdef OAK_LINPROBE
        {   const uint32_t *ca=activeFeatures(a,a.stateSlot); const float *cv=featureValues(a,a.stateSlot);
            for(uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ lf[z]=ca[z]; lv[z]=cv[ca[z]]; } }
#endif
#endif

        {   uint32_t ia=jbw_itemAt(seed,px+FW[hd][0],py+FW[hd][1]);
            if(ia<7u && jbw_destroyedAt(hist,HMASK,px+FW[hd][0],py+FW[hd][1])==0u)
                refuse[ia][action==0u?0:1]++;
            if(action==0u && jbw_blocks(ia) && wedgeRun==2000){
                static int wedgeDump=0;
                if(wedgeDump<3){ wedgeDump++;
                  float mn=0; for(int d=0;d<3;d++) mn+=M->policyMeanRaw[d]; mn/=3;
                  float e[3],sm=0; for(int d=0;d<3;d++){e[d]=expf(M->policyMeanRaw[d]-mn);sm+=e[d];}
                  printf("  WEDGE t=%ld logits %+.3f %+.3f %+.3f -> P %.4f %.4f %.4f  V=%.4f rbar=%.6f\n",
                    t,M->policyMeanRaw[0],M->policyMeanRaw[1],M->policyMeanRaw[2],
                    e[0]/sm,e[1]/sm,e[2]/sm, oakActionQForward(a), M->averageReward); }
            }
            if(action==0u && jbw_blocks(ia)){ wedgeRun++; if(wedgeRun>wedgeMax)wedgeMax=wedgeRun; wedge++; }
            else wedgeRun=0;
        }
        {   float sg=((t/100000)&1)==0?1.0f:-1.0f;
            uint32_t GD=sg>0?JBW_JELLY:JBW_ONION;
            int gi=8;
            for(int i=0;i<8;i++){ int k=(2*hd+i)&7; int qx=px+D8[k][0],qy=py+D8[k][1];
                uint32_t it=jbw_itemAt(seed,qx,qy);
                if(it==GD && jbw_destroyedAt(hist,HMASK,qx,qy)==0u){gi=i;break;} }
            steer[gi][action]++;
        }

        float reward=0.0f; moved=0;
        if (action==1u) hd=(hd+3)&3;
        else if (action==2u) hd=(hd+1)&3;
        else {
            int nx=px+FW[hd][0], ny=py+FW[hd][1];
            uint32_t item=jbw_itemAt(seed,nx,ny);
            if (!jbw_blocks(item)){
                px=nx;py=ny;moved=1;
                if (jbw_collectible(item)&&jbw_destroyedAt(hist,HMASK,nx,ny)==0u)
                    jbw_destroy(hist,HMASK,nx,ny,(uint32_t)t);
            }
        }
        if (action==0u && moved){
            uint32_t item=jbw_itemAt(seed,px,py);
            if (jbw_destroyedAt(hist,HMASK,px,py)==(uint32_t)t+1u){
                float sign=((t/100000)&1)==0?1.0f:-1.0f;
                if (item==JBW_JELLY) reward=sign; else if (item==JBW_ONION) reward=-sign;
            }
        }
#ifdef OAK_LINPROBE
        if (action==0u){
            float p=0.0f,dn=1.0f;
            for(uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ p+=lw[lf[z]]*lv[z]; dn+=lv[z]*lv[z]; }
            float e=reward-p;
            if(t>20000){ lse+=(double)e*e; lsy+=(double)reward*reward; lhit++;
                if(reward!=0.0f){ lnz++; if((p>0)==(reward>0)) lsign++; } }
            float sc=1.0f/dn;
            for(uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z) lw[lf[z]]+=e*lv[z]*sc;
        }
#endif
        M->reward = reward;
#if defined(OAK_ACTION_MODEL) || defined(OAK_SCALEFREE)
#ifdef OAK_ALIGN
        pendingReward = reward; havePending = 1;
#else
        oakActionModelLearn(a, reward);
#endif
#endif
        if (action==0u) fw++;
        int ep=(int)(t/(STEPS/10)); if(ep>9)ep=9;
        if (reward>0){G++;epG[ep]++;} else if (reward<0){B++;epB[ep]++;}
        { int hh=(int)(t/(STEPS/20)); if(hh>19)hh=19;
          if(reward>0) halfG[hh]++; else if(reward<0) halfB[hh]++; }
        tot += reward;

        uint32_t previousSlot = 1u - a.stateSlot;
        planAllOptions(a, previousSlot);
        learnFromTransition(a, featureValues(a,previousSlot), featureValues(a,1u-previousSlot),
                            previousSlot, 3u);
        updateUtilities(a);
        updateNursery(a);
    }
    printf("REAL OaK AGENT (ported kernel) on BIT-EXACT JellyBeanWorld, seed %u, %d steps\n",seed,STEPS);
    printf("  mean reward   %+.6f\n", tot/(double)STEPS);
    printf("  eaten         %ld good / %ld bad  (%.1f%% correct)\n",G,B,(G+B)?100.0*G/(G+B):0.0);
    printf("  %%forward      %.1f%%\n", 100.0*fw/STEPS);
    {   const char*NM2[7]={"empty","banana","onion","jelly","wall","tree","truffle"};
        printf("  P(forward | item ahead):");
        for(int i=2;i<=5;i++){ long n=refuse[i][0]+refuse[i][1];
            printf("  %s %.0f%%(n=%ld)",NM2[i],n?100.0*refuse[i][0]/n:0.0,n); }
        printf("\n  blocked-forward steps %ld (%.1f%%), longest wedge %ld\n",wedge,100.0*wedge/STEPS,wedgeMax); }
    {   double ea=0,la=0; int en=0,ln=0;
        printf("  TRACKING  correct%% in 1st vs 2nd half of each polarity epoch\n    ");
        for(int e=0;e<10;e++){
            long g1=halfG[2*e],b1=halfB[2*e],g2=halfG[2*e+1],b2=halfB[2*e+1];
            double c1=(g1+b1)?100.0*g1/(g1+b1):-1, c2=(g2+b2)?100.0*g2/(g2+b2):-1;
            if(c1>=0){ea+=c1;en++;} if(c2>=0){la+=c2;ln++;}
            printf("%.0f->%.0f  ",c1,c2); }
        printf("\n    mean first half %.1f%%   mean second half %.1f%%   adaptation %+.1f pts\n",
               en?ea/en:0.0, ln?la/ln:0.0, (ln?la/ln:0.0)-(en?ea/en:0.0)); }
#ifdef OAK_LINPROBE
    printf("  LINEAR READOUT on agent features -> immediate reward of moving forward\n");
    printf("    n=%ld  residual/variance = %.3f  (0=perfect, 1=useless)   sign correct %ld/%ld = %.0f%%\n",
           lhit, lsy>0?lse/lsy:1.0, lsign, lnz, lnz?100.0*lsign/lnz:0.0);
#endif
#ifdef OAK_COMMIT
    printf("  commit        %ld runs, mean length %.1f\n",commitRuns,commitRuns?(double)(commitSteps+commitRuns)/commitRuns:0.0);
#endif
    {   const char*L[9]={"ahead","a-right","right","b-right","behind","b-left","left","a-left","none"};
        printf("  STEERING  P(action | nearest good item direction)   [want: ahead->fwd, right->turn-right(2), left->turn-left(1)]\n");
        for(int i=0;i<9;i++){ long n=steer[i][0]+steer[i][1]+steer[i][2]; if(!n)continue;
            printf("    %-8s n=%-7ld  fwd %.0f%%  Lturn %.0f%%  Rturn %.0f%%\n",
                   L[i],n,100.0*steer[i][0]/n,100.0*steer[i][1]/n,100.0*steer[i][2]/n); } }
    {
        const char*NM[7]={"empty","banana","onion","jelly","wall","tree","truffle"};
        printf("  V(state) by item ahead, per epoch  [correct sign alternates + - + - ...]\n");
        for (int i=2;i<=3;i++){ printf("    %-7s:",NM[i]);
            for (int e=0;e<10;e++) printf(" %+8.4f", probeN[i][e]?probeV[i][e]/probeN[i][e]:0.0);
            printf("\n"); }
        printf("  ACTION-MODEL Q(forward)-Q(turn) by item ahead  [want + for good, - for bad]\n");
        for (int i=2;i<=3;i++){ printf("    %-7s:",NM[i]);
            for (int e=0;e<10;e++) printf(" %+8.4f", probeN[i][e]?probeQ[i][e]/probeN[i][e]:0.0);
            printf("\n"); }
        { int ok3=0,n3=0; printf("    %-7s:","Q(J)-Q(O)");
          for (int e=0;e<10;e++){
            double qj=probeN[3][e]?probeQ[3][e]/probeN[3][e]:0.0;
            double qo=probeN[2][e]?probeQ[2][e]/probeN[2][e]:0.0;
            double d=qj-qo; int want=(e%2==0)?1:-1;
            if(probeN[3][e]&&probeN[2][e]){n3++; if((d>0)==(want>0)) ok3++;}
            printf(" %+8.4f", d); }
          printf("\n    action model correct sign on %d of %d epochs\n",ok3,n3); }
        printf("  forward-logit advantage by item ahead  [want: + for good, - for bad]\n");
        for (int i=2;i<=3;i++){ printf("    %-7s:",NM[i]);
            for (int e=0;e<10;e++) printf(" %+8.4f", probeN[i][e]?probeA[i][e]/probeN[i][e]:0.0);
            printf("\n"); }
        printf("    %-7s:","A(J)-A(O)");
        { int ok2=0,n2=0;
          for (int e=0;e<10;e++){
            double aj=probeN[3][e]?probeA[3][e]/probeN[3][e]:0.0;
            double ao=probeN[2][e]?probeA[2][e]/probeN[2][e]:0.0;
            double d=aj-ao; int want=(e%2==0)?1:-1;
            if(probeN[3][e]&&probeN[2][e]){n2++; if((d>0)==(want>0)) ok2++;}
            printf(" %+8.4f", d); }
          printf("\n    actor correct sign on %d of %d epochs\n",ok2,n2); }
        printf("    %-7s:","V(J)-V(O)");
        int ok=0,n=0;
        for (int e=0;e<10;e++){
            double vj=probeN[3][e]?probeV[3][e]/probeN[3][e]:0.0;
            double vo=probeN[2][e]?probeV[2][e]/probeN[2][e]:0.0;
            double d=vj-vo; int want=(e%2==0)?1:-1;
            if (probeN[3][e]&&probeN[2][e]){ n++; if((d>0)==(want>0)) ok++; }
            printf(" %+8.4f", d);
        }
        printf("\n    correct sign on %d of %d epochs\n",ok,n);
    }
    {
        int cok=0,aok=0,qok=0,n=0;
        for (int e=0;e<10;e++){
            if(!probeN[3][e]||!probeN[2][e]) continue; n++;
            int want=(e%2==0)?1:-1;
            double dv=probeV[3][e]/probeN[3][e]-probeV[2][e]/probeN[2][e];
            double dq=probeQ[3][e]/probeN[3][e]-probeQ[2][e]/probeN[2][e];
            double da=probeA[3][e]/probeN[3][e]-probeA[2][e]/probeN[2][e];
            if((dv>0)==(want>0)) cok++;
            if((dq>0)==(want>0)) qok++;
            if((da>0)==(want>0)) aok++;
        }
        printf("SCORE critic %d/%d  actionmodel %d/%d  actor %d/%d  n=%d\n",cok,n,qok,n,aok,n,n);
    }
    printf("  per-epoch correct%%: ");
    for (int e=0;e<10;e++){ long g=epG[e],b=epB[e];
        printf("%5.1f ", (g+b)? 100.0*g/(g+b) : 0.0); }
    printf("\n  per-epoch eaten  : ");
    for (int e=0;e<10;e++) printf("%5ld ", epG[e]+epB[e]);
    printf("\n");
    printf("  recorded OaK: -0.00002 mean, ~52%% correct   PPO +0.00005   ceiling +0.00368\n");
    return 0;
}
