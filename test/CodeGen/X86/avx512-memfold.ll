; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -disable-peephole -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

define i8 @test_int_x86_avx512_mask_cmp_ss(<4 x float> %a, float* %b, i8 %mask) {
; CHECK-LABEL: test_int_x86_avx512_mask_cmp_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vcmpunordss (%rdi), %xmm0, %k0 {%k1}
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:    ## kill: %AL<def> %AL<kill> %EAX<kill>
; CHECK-NEXT:    retq
  %b.val = load float, float* %b
  %bv0 = insertelement <4 x float> undef, float %b.val, i32 0
  %bv1 = insertelement <4 x float> %bv0, float 0.000000e+00, i32 1
  %bv2 = insertelement <4 x float> %bv1, float 0.000000e+00, i32 2
  %bv  = insertelement <4 x float> %bv2, float 0.000000e+00, i32 3
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float> %a, <4 x float> %bv, i32 3, i8 %mask, i32 4)
  ret i8 %res2
}
declare i8 @llvm.x86.avx512.mask.cmp.ss(<4 x float>, <4 x float>, i32, i8, i32)

define <4 x float> @test_mask_max_ss(<4 x float> %a, float* %b, i8 %mask) {
; CHECK-LABEL: test_mask_max_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vmaxss (%rdi), %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %b.val = load float, float* %b
  %bv0 = insertelement <4 x float> undef, float %b.val, i32 0
  %bv1 = insertelement <4 x float> %bv0, float 0.000000e+00, i32 1
  %bv2 = insertelement <4 x float> %bv1, float 0.000000e+00, i32 2
  %bv  = insertelement <4 x float> %bv2, float 0.000000e+00, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>%a, <4 x float> %bv, <4 x float> zeroinitializer, i8 %mask, i32 4)
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.mask.max.ss.round(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_maskz_add_ss(<4 x float> %a, float* %b, i8 %mask) {
; CHECK-LABEL: test_maskz_add_ss:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vaddss (%rdi), %xmm0, %xmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %b.val = load float, float* %b
  %bv0 = insertelement <4 x float> undef, float %b.val, i32 0
  %bv1 = insertelement <4 x float> %bv0, float 0.000000e+00, i32 1
  %bv2 = insertelement <4 x float> %bv1, float 0.000000e+00, i32 2
  %bv  = insertelement <4 x float> %bv2, float 0.000000e+00, i32 3
  %res = call <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>%a, <4 x float> %bv, <4 x float> zeroinitializer, i8 %mask, i32 4)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.add.ss.round(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

declare <2 x double> @llvm.x86.avx512.mask.vfmadd.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32)

define <2 x double> @test_int_x86_avx512_mask_vfmadd_sd(<2 x double> %a, <2 x double> %b, double* %c, i8 %mask){
; CHECK-LABEL: test_int_x86_avx512_mask_vfmadd_sd:
; CHECK:       ## BB#0:
; CHECK-NEXT:    kmovw %esi, %k1
; CHECK-NEXT:    vfmadd213sd (%rdi), %xmm1, %xmm0 {%k1}
; CHECK-NEXT:    retq
  %c.val = load double, double* %c
  %cv0 = insertelement <2 x double> undef, double %c.val, i32 0
  %cv = insertelement <2 x double> %cv0, double 0.000000e+00, i32 1
  %res = call <2 x double> @llvm.x86.avx512.mask.vfmadd.sd(<2 x double> %a, <2 x double> %b, <2 x double> %cv, i8 %mask, i32 4)
  ret <2 x double> %res
}
