#include "jbw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static const int D8[8][2]={{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};
static const int FW[4][2]={{0,-1},{1,0},{0,1},{-1,0}};
#ifndef STEPS
#define STEPS 1000000
#endif
int main(int argc,char**argv){
    uint32_t seed=(uint32_t)atoi(argv[1]);
    int pol=atoi(argv[2]);
    float *resp=malloc(sizeof(float)*(size_t)JBW_RESPONSE_CELLS*JBW_DESTRUCTION_LIFETIME);
    jbw_buildResponse(resp);
    uint32_t HMASK=(1u<<20)-1u; uint32_t *hist=calloc(HMASK+1u,3u*sizeof(uint32_t));
    int px=0,py=0,hd=0,moved=0;
    double tot=0; long G=0,B=0,fw=0;
    float pj=0.f,po=0.f;
    for(long t=0;t<STEPS;t++){
        float ob[8];
        for(int i=0;i<8;i++){
            int k=(2*hd+i)&7; int qx=px+D8[k][0],qy=py+D8[k][1];
            uint32_t it=jbw_itemAt(seed,qx,qy);
            int absent=!jbw_blocks(it)&&jbw_destroyedAt(hist,HMASK,qx,qy)!=0u;
            ob[i]=absent?0.0f:(float)it;
        }
        float sc[3]; jbw_scentAt(seed,px,py,(uint32_t)t,hist,HMASK,resp,sc);
        float sign=((t/100000)&1)==0?1.0f:-1.0f;
        float good = sign>0?sc[2]:sc[1];
        float bad  = sign>0?sc[1]:sc[2];
        uint32_t GOODI = sign>0?JBW_JELLY:JBW_ONION;
        uint32_t BADI  = sign>0?JBW_ONION:JBW_JELLY;
        uint32_t action;
        u4 rr=philox4x32(seed,2u,(u4){(uint32_t)t,7,0,0});
        switch(pol){
        case 0: action=0u; break;
        case 1: action=rr.x%3u; break;
        case 2: case 3: case 4: {
                  uint32_t ah=(uint32_t)ob[0];
                  if(ah==GOODI){action=0u;break;}
                  if(ah==BADI||ah==JBW_WALL||ah==JBW_TREE){action=1u+(rr.x&1u);break;}
                  int found=-1;
                  for(int i=1;i<8;i++) if((uint32_t)ob[i]==GOODI){found=i;break;}
                  if(found==2){action=2u;break;}
                  if(found==6){action=1u;break;}
                  if(found==1||found==7){action=0u;break;}
                  if(found>=3&&found<=5){action=(found<5)?2u:1u;break;}
                  if(pol==2){action=0u;break;}
                  if(pol==3){
                      if(!moved){action=0u;break;}
                      action=((good-pj)-(bad-po)>=0.0f)?0u:(1u+(rr.x&1u));
                      break;
                  }
                  {
                      float best=-1e30f; uint32_t ba=0u; int any=0;
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
                  }
                } break;
        case 5: { int tx=px+FW[hd][0],ty=py+FW[hd][1];
                  action = jbw_blocks(jbw_itemAt(seed,tx,ty)) ? (1u+(rr.x&1u)) : 0u; } break;
        case 6: { int tx=px+FW[hd][0],ty=py+FW[hd][1];
                  uint32_t it=jbw_itemAt(seed,tx,ty);
                  int bad = (it==BADI) && jbw_destroyedAt(hist,HMASK,tx,ty)==0u;
                  action = (jbw_blocks(it)||bad) ? (1u+(rr.x&1u)) : 0u; } break;
        case 7: { uint32_t ah=(uint32_t)ob[0];
                  if(ah==BADI||ah==JBW_WALL||ah==JBW_TREE){action=1u+(rr.x&1u);break;}
                  int found=-1;
                  for(int i=1;i<8;i++) if((uint32_t)ob[i]==GOODI){found=i;break;}
                  if(found==2){action=2u;break;}
                  if(found==6){action=1u;break;}
                  action=0u; } break;
        default: action=0u;
        }
        pj=good; po=bad;
        float reward=0.f; moved=0;
        if(action==1u) hd=(hd+3)&3;
        else if(action==2u) hd=(hd+1)&3;
        else { int nx=px+FW[hd][0],ny=py+FW[hd][1];
               uint32_t item=jbw_itemAt(seed,nx,ny);
               if(!jbw_blocks(item)){ px=nx;py=ny;moved=1;
                   if(jbw_collectible(item)&&jbw_destroyedAt(hist,HMASK,nx,ny)==0u)
                       jbw_destroy(hist,HMASK,nx,ny,(uint32_t)t); } }
        if(action==0u&&moved){ uint32_t item=jbw_itemAt(seed,px,py);
            if(jbw_destroyedAt(hist,HMASK,px,py)==(uint32_t)t+1u){
                if(item==JBW_JELLY) reward=sign; else if(item==JBW_ONION) reward=-sign; } }
        if(action==0u)fw++;
        if(reward>0){G++;} else if(reward<0){B++;}
        tot+=reward;
    }
    printf("%+.6f  eaten %ld/%ld (%.0f%% correct)  fwd %.0f%%\n",
        tot/(double)STEPS,G,B,(G+B)?100.0*G/(G+B):0.0,100.0*fw/STEPS);
    return 0;
}
