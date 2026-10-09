#!/bin/bash
# Verifies the encoders in rsp/neon_enc.h against GNU as (aarch64-linux-gnu-as + objdump).
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d)
cat > $T/t.cpp <<'CPP'
#include "rsp/neon_enc.h"
#include <stdio.h>
#include <initializer_list>
using namespace RSP::NeonEnc;
int main(){
  for (unsigned off : {0u, 16u, 800u, 832u, 4080u, 65520u}) { printf("ldr q3,[x19,#%u]\n%08x\n", off, ldr_q(3,19,off)); printf("str q7,[x19,#%u]\n%08x\n", off, str_q(7,19,off)); }
  for (unsigned off : {0u, 8u, 800u, 32760u}) { printf("ldr d3,[x19,#%u]\n%08x\n", off, ldr_d(3,19,off)); printf("str d5,[x19,#%u]\n%08x\n", off, str_d(5,19,off)); }
  printf("mov v1.16b,v0.16b\n%08x\n", mov16b(1,0));
  printf("mov v5.16b,v30.16b\n%08x\n", mov16b(5,30));
  printf("movi v0.4s,#0\n%08x\n", movi_zero(0));
  printf("movi v12.4s,#0\n%08x\n", movi_zero(12));
  printf("trn1 v1.8h,v1.8h,v1.8h\n%08x\n", trn1_8h(1,1,1));
  printf("trn2 v1.8h,v2.8h,v3.8h\n%08x\n", trn2_8h(1,2,3));
  printf("trn1 v1.4s,v1.4s,v1.4s\n%08x\n", trn1_4s(1,1,1));
  printf("trn2 v4.4s,v5.4s,v6.4s\n%08x\n", trn2_4s(4,5,6));
  for (unsigned i = 0; i < 8; i++) printf("dup v1.8h,v1.h[%u]\n%08x\n", i, dup_8h(1,1,i));
  printf("dup v9.8h,v31.h[5]\n%08x\n", dup_8h(9,31,5));
  printf("ldr d0,[x20,x9]\n%08x\n", ldr_d_r(0,20,9)); printf("str d3,[x20,x15]\n%08x\n", str_d_r(3,20,15));
  printf("ldr q1,[x20,x22]\n%08x\n", ldr_q_r(1,20,22)); printf("str q2,[x20,x28]\n%08x\n", str_q_r(2,20,28));
  printf("ldr w16,[x20,x10]\n%08x\n", ldr_w_r(16,20,10)); printf("str w16,[x20,x10]\n%08x\n", str_w_r(16,20,10));
  printf("ldrh w16,[x20,x10]\n%08x\n", ldrh_r(16,20,10)); printf("strh w16,[x20,x10]\n%08x\n", strh_r(16,20,10));
  printf("ldrb w16,[x20,x10]\n%08x\n", ldrb_r(16,20,10)); printf("strb w16,[x20,x10]\n%08x\n", strb_r(16,20,10));
  printf("ldr w16,[x19,#704]\n%08x\n", ldr_w(16,19,704)); printf("str w16,[x19,#4]\n%08x\n", str_w(16,19,4));
  printf("ldrh w16,[x19,#702]\n%08x\n", ldrh(16,19,702)); printf("strh w17,[x19,#194]\n%08x\n", strh(17,19,194));
  printf("ldrb w16,[x19,#703]\n%08x\n", ldrb(16,19,703)); printf("strb w17,[x19,#195]\n%08x\n", strb(17,19,195));
  printf("add x17,x19,#703\n%08x\n", add_imm(17,19,703)); printf("add x16,x20,#0\n%08x\n", add_imm(16,20,0));
  printf("ror w16,w16,#16\n%08x\n", ror16_w(16,16)); printf("ror w3,w5,#16\n%08x\n", ror16_w(3,5));
  printf("rev32 v0.4h,v0.4h\n%08x\n", rev32_4h(0,0)); printf("rev32 v2.8h,v3.8h\n%08x\n", rev32_8h(2,3)); printf("rev32 v1.8b,v4.8b\n%08x\n", rev32_8b(1,4));
  printf("shll v0.8h,v0.8b,#8\n%08x\n", shll_8h_8(0,0)); printf("ushll v2.8h,v5.8b,#7\n%08x\n", ushll_8h_7(2,5));
  printf("shrn v0.8b,v0.8h,#8\n%08x\n", shrn_8b(0,0,8)); printf("shrn v2.8b,v5.8h,#7\n%08x\n", shrn_8b(2,5,7));

  printf("rev32 v1.16b,v1.16b\n%08x\n", rev32_16b(1,1)); printf("rev16 v4.16b,v7.16b\n%08x\n", rev16_16b(4,7));
  printf("dup v3.16b,w9\n%08x\n", dup_16b_w(3,9)); printf("dup v3.16b,w28\n%08x\n", dup_16b_w(3,28));
  printf("add v2.16b,v2.16b,v3.16b\n%08x\n", add_16b(2,2,3)); printf("tbl v1.16b,{v1.16b},v2.16b\n%08x\n", tbl_16b(1,1,2));
  printf("cmhi v5.16b,v5.16b,v2.16b\n%08x\n", cmhi_16b(5,5,2)); printf("bsl v5.16b,v1.16b,v4.16b\n%08x\n", bsl_16b(5,1,4));
  printf("movi v5.16b,#16\n%08x\n", movi_16b(5,16)); printf("movi v5.16b,#255\n%08x\n", movi_16b(5,255)); printf("movi v6.16b,#33\n%08x\n", movi_16b(6,33));
  for (unsigned i = 0; i < 16; i += 2) printf("ext v1.16b,v0.16b,v0.16b,#%u\n%08x\n", i, ext_16b(1,0,0,i));

}
CPP
g++ -I. -o $T/t $T/t.cpp
$T/t > $T/out.txt
python3 - $T <<'PY'
import sys,subprocess
T=sys.argv[1]
lines=open(T+'/out.txt').read().split('\n')
asm=[];words=[]
for i in range(0,len(lines)-1,2):
    asm.append(lines[i]); words.append(int(lines[i+1],16))
open(T+'/a.s','w').write('\n'.join(asm)+'\n')
subprocess.check_call(['aarch64-linux-gnu-as',T+'/a.s','-o',T+'/a.o'])
obj=subprocess.check_output(['aarch64-linux-gnu-objdump','-d',T+'/a.o']).decode()
got=[int(l.split('\t')[1],16) for l in obj.split('\n') if '\t' in l and ':' in l.split('\t')[0]]
bad=0
for a,w,g in zip(asm,words,got):
    if w!=g: print('MISMATCH',a,'%08x'%w,'%08x'%g); bad+=1
print('encoders checked: %d, mismatches: %d'%(len(words),bad)); sys.exit(1 if bad else 0)
PY
rm -rf $T
