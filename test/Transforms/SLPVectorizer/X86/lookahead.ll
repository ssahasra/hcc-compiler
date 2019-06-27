; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -slp-vectorizer -S < %s -mtriple=x86_64-unknown-linux -mcpu=corei7-avx | FileCheck %s
;
; This file tests the look-ahead operand reordering heuristic.
;
;
; This checks that operand reordering will reorder the operands of the adds
; by taking into consideration the instructions beyond the immediate
; predecessors.
;
; A[0] B[0] C[0] D[0]  C[1] D[1] A[1] B[1]
;     \  /   \  /          \  /   \  /
;       -     -              -     -
;        \   /                \   /
;          +                    +
;          |                    |
;         S[0]                 S[1]
;
define void @lookahead_basic(double* %array) {
; CHECK-LABEL: @lookahead_basic(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[IDX0:%.*]] = getelementptr inbounds double, double* [[ARRAY:%.*]], i64 0
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 1
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 2
; CHECK-NEXT:    [[IDX3:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 3
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 4
; CHECK-NEXT:    [[IDX5:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 5
; CHECK-NEXT:    [[IDX6:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 6
; CHECK-NEXT:    [[IDX7:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 7
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* [[TMP0]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast double* [[IDX2]] to <2 x double>*
; CHECK-NEXT:    [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[TMP2]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast double* [[IDX4]] to <2 x double>*
; CHECK-NEXT:    [[TMP5:%.*]] = load <2 x double>, <2 x double>* [[TMP4]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast double* [[IDX6]] to <2 x double>*
; CHECK-NEXT:    [[TMP7:%.*]] = load <2 x double>, <2 x double>* [[TMP6]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = fsub fast <2 x double> [[TMP1]], [[TMP3]]
; CHECK-NEXT:    [[TMP9:%.*]] = fsub fast <2 x double> [[TMP5]], [[TMP7]]
; CHECK-NEXT:    [[TMP10:%.*]] = fadd fast <2 x double> [[TMP8]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP10]], <2 x double>* [[TMP11]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %idx0 = getelementptr inbounds double, double* %array, i64 0
  %idx1 = getelementptr inbounds double, double* %array, i64 1
  %idx2 = getelementptr inbounds double, double* %array, i64 2
  %idx3 = getelementptr inbounds double, double* %array, i64 3
  %idx4 = getelementptr inbounds double, double* %array, i64 4
  %idx5 = getelementptr inbounds double, double* %array, i64 5
  %idx6 = getelementptr inbounds double, double* %array, i64 6
  %idx7 = getelementptr inbounds double, double* %array, i64 7

  %A_0 = load double, double *%idx0, align 8
  %A_1 = load double, double *%idx1, align 8
  %B_0 = load double, double *%idx2, align 8
  %B_1 = load double, double *%idx3, align 8
  %C_0 = load double, double *%idx4, align 8
  %C_1 = load double, double *%idx5, align 8
  %D_0 = load double, double *%idx6, align 8
  %D_1 = load double, double *%idx7, align 8

  %subAB_0 = fsub fast double %A_0, %B_0
  %subCD_0 = fsub fast double %C_0, %D_0

  %subAB_1 = fsub fast double %A_1, %B_1
  %subCD_1 = fsub fast double %C_1, %D_1

  %addABCD_0 = fadd fast double %subAB_0, %subCD_0
  %addCDAB_1 = fadd fast double %subCD_1, %subAB_1

  store double %addABCD_0, double *%idx0, align 8
  store double %addCDAB_1, double *%idx1, align 8
  ret void
}


; Check whether the look-ahead operand reordering heuristic will avoid
; bundling the alt opcodes. The vectorized code should have no shuffles.
;
; A[0] B[0] A[0] B[0]  A[1] A[1] A[1] B[1]
;     \  /   \  /          \  /   \  /
;       +     -              -     +
;        \   /                \   /
;          +                    +
;          |                    |
;         S[0]                 S[1]
;
define void @lookahead_alt1(double* %array) {
; CHECK-LABEL: @lookahead_alt1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[IDX0:%.*]] = getelementptr inbounds double, double* [[ARRAY:%.*]], i64 0
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 1
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 2
; CHECK-NEXT:    [[IDX3:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 3
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 4
; CHECK-NEXT:    [[IDX5:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 5
; CHECK-NEXT:    [[IDX6:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 6
; CHECK-NEXT:    [[IDX7:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 7
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* [[TMP0]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast double* [[IDX2]] to <2 x double>*
; CHECK-NEXT:    [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[TMP2]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = fsub fast <2 x double> [[TMP1]], [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = fadd fast <2 x double> [[TMP1]], [[TMP3]]
; CHECK-NEXT:    [[TMP6:%.*]] = fadd fast <2 x double> [[TMP5]], [[TMP4]]
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP6]], <2 x double>* [[TMP7]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %idx0 = getelementptr inbounds double, double* %array, i64 0
  %idx1 = getelementptr inbounds double, double* %array, i64 1
  %idx2 = getelementptr inbounds double, double* %array, i64 2
  %idx3 = getelementptr inbounds double, double* %array, i64 3
  %idx4 = getelementptr inbounds double, double* %array, i64 4
  %idx5 = getelementptr inbounds double, double* %array, i64 5
  %idx6 = getelementptr inbounds double, double* %array, i64 6
  %idx7 = getelementptr inbounds double, double* %array, i64 7

  %A_0 = load double, double *%idx0, align 8
  %A_1 = load double, double *%idx1, align 8
  %B_0 = load double, double *%idx2, align 8
  %B_1 = load double, double *%idx3, align 8

  %addAB_0_L = fadd fast double %A_0, %B_0
  %subAB_0_R = fsub fast double %A_0, %B_0

  %subAB_1_L = fsub fast double %A_1, %B_1
  %addAB_1_R = fadd fast double %A_1, %B_1

  %addABCD_0 = fadd fast double %addAB_0_L, %subAB_0_R
  %addCDAB_1 = fadd fast double %subAB_1_L, %addAB_1_R

  store double %addABCD_0, double *%idx0, align 8
  store double %addCDAB_1, double *%idx1, align 8
  ret void
}


; This code should get vectorized all the way to the loads with shuffles for
; the alt opcodes.
;
; A[0] B[0] C[0] D[0]  C[1] D[1] A[1] B[1]
;     \  /   \  /          \  /   \  /
;       +     -              +     -
;        \   /                \   /
;          +                    +
;          |                    |
;         S[0]                 S[1]
;
define void @lookahead_alt2(double* %array) {
; CHECK-LABEL: @lookahead_alt2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[IDX0:%.*]] = getelementptr inbounds double, double* [[ARRAY:%.*]], i64 0
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 1
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 2
; CHECK-NEXT:    [[IDX3:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 3
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 4
; CHECK-NEXT:    [[IDX5:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 5
; CHECK-NEXT:    [[IDX6:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 6
; CHECK-NEXT:    [[IDX7:%.*]] = getelementptr inbounds double, double* [[ARRAY]], i64 7
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* [[TMP0]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast double* [[IDX2]] to <2 x double>*
; CHECK-NEXT:    [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[TMP2]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast double* [[IDX4]] to <2 x double>*
; CHECK-NEXT:    [[TMP5:%.*]] = load <2 x double>, <2 x double>* [[TMP4]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast double* [[IDX6]] to <2 x double>*
; CHECK-NEXT:    [[TMP7:%.*]] = load <2 x double>, <2 x double>* [[TMP6]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = fsub fast <2 x double> [[TMP5]], [[TMP7]]
; CHECK-NEXT:    [[TMP9:%.*]] = fadd fast <2 x double> [[TMP5]], [[TMP7]]
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <2 x double> [[TMP8]], <2 x double> [[TMP9]], <2 x i32> <i32 0, i32 3>
; CHECK-NEXT:    [[TMP11:%.*]] = fadd fast <2 x double> [[TMP1]], [[TMP3]]
; CHECK-NEXT:    [[TMP12:%.*]] = fsub fast <2 x double> [[TMP1]], [[TMP3]]
; CHECK-NEXT:    [[TMP13:%.*]] = shufflevector <2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x i32> <i32 0, i32 3>
; CHECK-NEXT:    [[TMP14:%.*]] = fadd fast <2 x double> [[TMP13]], [[TMP10]]
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast double* [[IDX0]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP14]], <2 x double>* [[TMP15]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %idx0 = getelementptr inbounds double, double* %array, i64 0
  %idx1 = getelementptr inbounds double, double* %array, i64 1
  %idx2 = getelementptr inbounds double, double* %array, i64 2
  %idx3 = getelementptr inbounds double, double* %array, i64 3
  %idx4 = getelementptr inbounds double, double* %array, i64 4
  %idx5 = getelementptr inbounds double, double* %array, i64 5
  %idx6 = getelementptr inbounds double, double* %array, i64 6
  %idx7 = getelementptr inbounds double, double* %array, i64 7

  %A_0 = load double, double *%idx0, align 8
  %A_1 = load double, double *%idx1, align 8
  %B_0 = load double, double *%idx2, align 8
  %B_1 = load double, double *%idx3, align 8
  %C_0 = load double, double *%idx4, align 8
  %C_1 = load double, double *%idx5, align 8
  %D_0 = load double, double *%idx6, align 8
  %D_1 = load double, double *%idx7, align 8

  %addAB_0 = fadd fast double %A_0, %B_0
  %subCD_0 = fsub fast double %C_0, %D_0

  %addCD_1 = fadd fast double %C_1, %D_1
  %subAB_1 = fsub fast double %A_1, %B_1

  %addABCD_0 = fadd fast double %addAB_0, %subCD_0
  %addCDAB_1 = fadd fast double %addCD_1, %subAB_1

  store double %addABCD_0, double *%idx0, align 8
  store double %addCDAB_1, double *%idx1, align 8
  ret void
}


;
; A[0] B[0] C[0] D[0]  A[1] B[2] A[2] B[1]
;     \  /   \  /       /  \  /   \  /
;       -     -        U     -     -
;        \   /                \   /
;          +                    +
;          |                    |
;         S[0]                 S[1]
;
; SLP should reorder the operands of the RHS add taking into consideration the cost of external uses.
; It is more profitable to reorder the operands of the RHS add, because A[1] has an external use.

define void @lookahead_external_uses(double* %A, double *%B, double *%C, double *%D, double *%S, double *%Ext1, double *%Ext2) {
; CHECK-LABEL: @lookahead_external_uses(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[IDXA0:%.*]] = getelementptr inbounds double, double* [[A:%.*]], i64 0
; CHECK-NEXT:    [[IDXB0:%.*]] = getelementptr inbounds double, double* [[B:%.*]], i64 0
; CHECK-NEXT:    [[IDXC0:%.*]] = getelementptr inbounds double, double* [[C:%.*]], i64 0
; CHECK-NEXT:    [[IDXD0:%.*]] = getelementptr inbounds double, double* [[D:%.*]], i64 0
; CHECK-NEXT:    [[IDXA1:%.*]] = getelementptr inbounds double, double* [[A]], i64 1
; CHECK-NEXT:    [[IDXB2:%.*]] = getelementptr inbounds double, double* [[B]], i64 2
; CHECK-NEXT:    [[IDXA2:%.*]] = getelementptr inbounds double, double* [[A]], i64 2
; CHECK-NEXT:    [[IDXB1:%.*]] = getelementptr inbounds double, double* [[B]], i64 1
; CHECK-NEXT:    [[A0:%.*]] = load double, double* [[IDXA0]], align 8
; CHECK-NEXT:    [[C0:%.*]] = load double, double* [[IDXC0]], align 8
; CHECK-NEXT:    [[D0:%.*]] = load double, double* [[IDXD0]], align 8
; CHECK-NEXT:    [[A1:%.*]] = load double, double* [[IDXA1]], align 8
; CHECK-NEXT:    [[B2:%.*]] = load double, double* [[IDXB2]], align 8
; CHECK-NEXT:    [[A2:%.*]] = load double, double* [[IDXA2]], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast double* [[IDXB0]] to <2 x double>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* [[TMP0]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x double> undef, double [[C0]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x double> [[TMP2]], double [[A1]], i32 1
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <2 x double> undef, double [[D0]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <2 x double> [[TMP4]], double [[B2]], i32 1
; CHECK-NEXT:    [[TMP6:%.*]] = fsub fast <2 x double> [[TMP3]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = insertelement <2 x double> undef, double [[A0]], i32 0
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <2 x double> [[TMP7]], double [[A2]], i32 1
; CHECK-NEXT:    [[TMP9:%.*]] = fsub fast <2 x double> [[TMP8]], [[TMP1]]
; CHECK-NEXT:    [[TMP10:%.*]] = fadd fast <2 x double> [[TMP9]], [[TMP6]]
; CHECK-NEXT:    [[IDXS0:%.*]] = getelementptr inbounds double, double* [[S:%.*]], i64 0
; CHECK-NEXT:    [[IDXS1:%.*]] = getelementptr inbounds double, double* [[S]], i64 1
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast double* [[IDXS0]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP10]], <2 x double>* [[TMP11]], align 8
; CHECK-NEXT:    store double [[A1]], double* [[EXT1:%.*]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %IdxA0 = getelementptr inbounds double, double* %A, i64 0
  %IdxB0 = getelementptr inbounds double, double* %B, i64 0
  %IdxC0 = getelementptr inbounds double, double* %C, i64 0
  %IdxD0 = getelementptr inbounds double, double* %D, i64 0

  %IdxA1 = getelementptr inbounds double, double* %A, i64 1
  %IdxB2 = getelementptr inbounds double, double* %B, i64 2
  %IdxA2 = getelementptr inbounds double, double* %A, i64 2
  %IdxB1 = getelementptr inbounds double, double* %B, i64 1

  %A0 = load double, double *%IdxA0, align 8
  %B0 = load double, double *%IdxB0, align 8
  %C0 = load double, double *%IdxC0, align 8
  %D0 = load double, double *%IdxD0, align 8

  %A1 = load double, double *%IdxA1, align 8
  %B2 = load double, double *%IdxB2, align 8
  %A2 = load double, double *%IdxA2, align 8
  %B1 = load double, double *%IdxB1, align 8

  %subA0B0 = fsub fast double %A0, %B0
  %subC0D0 = fsub fast double %C0, %D0

  %subA1B2 = fsub fast double %A1, %B2
  %subA2B1 = fsub fast double %A2, %B1

  %add0 = fadd fast double %subA0B0, %subC0D0
  %add1 = fadd fast double %subA1B2, %subA2B1

  %IdxS0 = getelementptr inbounds double, double* %S, i64 0
  %IdxS1 = getelementptr inbounds double, double* %S, i64 1

  store double %add0, double *%IdxS0, align 8
  store double %add1, double *%IdxS1, align 8

  ; External use
  store double %A1, double *%Ext1, align 8
  ret void
}


; This checks that the lookahead code does not crash when instructions with the same opcodes have different numbers of operands (in this case the calls).

%Class = type { i8 }
declare double @_ZN1i2ayEv(%Class*)
declare double @_ZN1i2axEv()

define void @lookahead_crash(double* %A, double *%S, %Class *%Arg0) {
; CHECK-LABEL: @lookahead_crash(
; CHECK-NEXT:    [[IDXA0:%.*]] = getelementptr inbounds double, double* [[A:%.*]], i64 0
; CHECK-NEXT:    [[IDXA1:%.*]] = getelementptr inbounds double, double* [[A]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast double* [[IDXA0]] to <2 x double>*
; CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, <2 x double>* [[TMP1]], align 8
; CHECK-NEXT:    [[C0:%.*]] = call double @_ZN1i2ayEv(%Class* [[ARG0:%.*]])
; CHECK-NEXT:    [[C1:%.*]] = call double @_ZN1i2axEv()
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x double> undef, double [[C0]], i32 0
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <2 x double> [[TMP3]], double [[C1]], i32 1
; CHECK-NEXT:    [[TMP5:%.*]] = fadd fast <2 x double> [[TMP2]], [[TMP4]]
; CHECK-NEXT:    [[IDXS0:%.*]] = getelementptr inbounds double, double* [[S:%.*]], i64 0
; CHECK-NEXT:    [[IDXS1:%.*]] = getelementptr inbounds double, double* [[S]], i64 1
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast double* [[IDXS0]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP5]], <2 x double>* [[TMP6]], align 8
; CHECK-NEXT:    ret void
;
  %IdxA0 = getelementptr inbounds double, double* %A, i64 0
  %IdxA1 = getelementptr inbounds double, double* %A, i64 1

  %A0 = load double, double *%IdxA0, align 8
  %A1 = load double, double *%IdxA1, align 8

  %C0 = call double @_ZN1i2ayEv(%Class *%Arg0)
  %C1 = call double @_ZN1i2axEv()

  %add0 = fadd fast double %A0, %C0
  %add1 = fadd fast double %A1, %C1

  %IdxS0 = getelementptr inbounds double, double* %S, i64 0
  %IdxS1 = getelementptr inbounds double, double* %S, i64 1
  store double %add0, double *%IdxS0, align 8
  store double %add1, double *%IdxS1, align 8
  ret void
}
