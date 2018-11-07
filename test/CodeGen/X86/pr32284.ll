; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -fast-isel-sink-local-values -O0 -mtriple=x86_64-unknown -mcpu=skx -o - %s | FileCheck %s --check-prefix=X86-O0
; RUN: llc -fast-isel-sink-local-values     -mtriple=x86_64-unknown -mcpu=skx -o - %s | FileCheck %s --check-prefix=X64
; RUN: llc -fast-isel-sink-local-values -O0 -mtriple=i686-unknown   -mcpu=skx -o - %s | FileCheck %s --check-prefix=686-O0
; RUN: llc -fast-isel-sink-local-values     -mtriple=i686-unknown   -mcpu=skx -o - %s | FileCheck %s --check-prefix=686

@c = external constant i8, align 1

define void @foo() {
; X86-O0-LABEL: foo:
; X86-O0:       # %bb.0: # %entry
; X86-O0-NEXT:    xorl %eax, %eax
; X86-O0-NEXT:    movl %eax, %ecx
; X86-O0-NEXT:    xorl %eax, %eax
; X86-O0-NEXT:    movzbl c, %edx
; X86-O0-NEXT:    subl %edx, %eax
; X86-O0-NEXT:    movslq %eax, %rsi
; X86-O0-NEXT:    subq %rsi, %rcx
; X86-O0-NEXT:    movb %cl, %dil
; X86-O0-NEXT:    cmpb $0, %dil
; X86-O0-NEXT:    setne %dil
; X86-O0-NEXT:    andb $1, %dil
; X86-O0-NEXT:    movb %dil, -{{[0-9]+}}(%rsp)
; X86-O0-NEXT:    cmpb $0, c
; X86-O0-NEXT:    setne %dil
; X86-O0-NEXT:    xorb $-1, %dil
; X86-O0-NEXT:    xorb $-1, %dil
; X86-O0-NEXT:    andb $1, %dil
; X86-O0-NEXT:    movzbl %dil, %eax
; X86-O0-NEXT:    movzbl c, %edx
; X86-O0-NEXT:    cmpl %edx, %eax
; X86-O0-NEXT:    setle %dil
; X86-O0-NEXT:    andb $1, %dil
; X86-O0-NEXT:    movzbl %dil, %eax
; X86-O0-NEXT:    movl %eax, -{{[0-9]+}}(%rsp)
; X86-O0-NEXT:    retq
;
; X64-LABEL: foo:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movzbl {{.*}}(%rip), %eax
; X64-NEXT:    xorl %ecx, %ecx
; X64-NEXT:    testl %eax, %eax
; X64-NEXT:    setne %cl
; X64-NEXT:    testb %al, %al
; X64-NEXT:    setne -{{[0-9]+}}(%rsp)
; X64-NEXT:    xorl %edx, %edx
; X64-NEXT:    cmpl %eax, %ecx
; X64-NEXT:    setle %dl
; X64-NEXT:    movl %edx, -{{[0-9]+}}(%rsp)
; X64-NEXT:    retq
;
; 686-O0-LABEL: foo:
; 686-O0:       # %bb.0: # %entry
; 686-O0-NEXT:    subl $8, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 12
; 686-O0-NEXT:    movb c, %al
; 686-O0-NEXT:    cmpb $0, %al
; 686-O0-NEXT:    setne %al
; 686-O0-NEXT:    andb $1, %al
; 686-O0-NEXT:    movb %al, {{[0-9]+}}(%esp)
; 686-O0-NEXT:    cmpb $0, c
; 686-O0-NEXT:    setne %al
; 686-O0-NEXT:    xorb $-1, %al
; 686-O0-NEXT:    xorb $-1, %al
; 686-O0-NEXT:    andb $1, %al
; 686-O0-NEXT:    movzbl %al, %ecx
; 686-O0-NEXT:    movzbl c, %edx
; 686-O0-NEXT:    cmpl %edx, %ecx
; 686-O0-NEXT:    setle %al
; 686-O0-NEXT:    andb $1, %al
; 686-O0-NEXT:    movzbl %al, %ecx
; 686-O0-NEXT:    movl %ecx, (%esp)
; 686-O0-NEXT:    addl $8, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 4
; 686-O0-NEXT:    retl
;
; 686-LABEL: foo:
; 686:       # %bb.0: # %entry
; 686-NEXT:    subl $8, %esp
; 686-NEXT:    .cfi_def_cfa_offset 12
; 686-NEXT:    movzbl c, %eax
; 686-NEXT:    xorl %ecx, %ecx
; 686-NEXT:    testl %eax, %eax
; 686-NEXT:    setne %cl
; 686-NEXT:    testb %al, %al
; 686-NEXT:    setne {{[0-9]+}}(%esp)
; 686-NEXT:    xorl %edx, %edx
; 686-NEXT:    cmpl %eax, %ecx
; 686-NEXT:    setle %dl
; 686-NEXT:    movl %edx, {{[0-9]+}}(%esp)
; 686-NEXT:    addl $8, %esp
; 686-NEXT:    .cfi_def_cfa_offset 4
; 686-NEXT:    retl
entry:
  %a = alloca i8, align 1
  %b = alloca i32, align 4
  %0 = load i8, i8* @c, align 1
  %conv = zext i8 %0 to i32
  %sub = sub nsw i32 0, %conv
  %conv1 = sext i32 %sub to i64
  %sub2 = sub nsw i64 0, %conv1
  %conv3 = trunc i64 %sub2 to i8
  %tobool = icmp ne i8 %conv3, 0
  %frombool = zext i1 %tobool to i8
  store i8 %frombool, i8* %a, align 1
  %1 = load i8, i8* @c, align 1
  %tobool4 = icmp ne i8 %1, 0
  %lnot = xor i1 %tobool4, true
  %lnot5 = xor i1 %lnot, true
  %conv6 = zext i1 %lnot5 to i32
  %2 = load i8, i8* @c, align 1
  %conv7 = zext i8 %2 to i32
  %cmp = icmp sle i32 %conv6, %conv7
  %conv8 = zext i1 %cmp to i32
  store i32 %conv8, i32* %b, align 4
  ret void
}

@var_5 = external global i32, align 4
@var_57 = external global i64, align 8
@_ZN8struct_210member_2_0E = external global i64, align 8

define void @f1() {
; X86-O0-LABEL: f1:
; X86-O0:       # %bb.0: # %entry
; X86-O0-NEXT:    movslq var_5, %rax
; X86-O0-NEXT:    movabsq $8381627093, %rcx # imm = 0x1F3957AD5
; X86-O0-NEXT:    addq %rcx, %rax
; X86-O0-NEXT:    cmpq $0, %rax
; X86-O0-NEXT:    setne %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movb %dl, -{{[0-9]+}}(%rsp)
; X86-O0-NEXT:    movl var_5, %esi
; X86-O0-NEXT:    xorl $-1, %esi
; X86-O0-NEXT:    cmpl $0, %esi
; X86-O0-NEXT:    setne %dl
; X86-O0-NEXT:    xorb $-1, %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movzbl %dl, %esi
; X86-O0-NEXT:    movl %esi, %eax
; X86-O0-NEXT:    movslq var_5, %rcx
; X86-O0-NEXT:    addq $7093, %rcx # imm = 0x1BB5
; X86-O0-NEXT:    cmpq %rcx, %rax
; X86-O0-NEXT:    setg %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movzbl %dl, %esi
; X86-O0-NEXT:    movl %esi, %eax
; X86-O0-NEXT:    movq %rax, var_57
; X86-O0-NEXT:    movl var_5, %esi
; X86-O0-NEXT:    xorl $-1, %esi
; X86-O0-NEXT:    cmpl $0, %esi
; X86-O0-NEXT:    setne %dl
; X86-O0-NEXT:    xorb $-1, %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movzbl %dl, %esi
; X86-O0-NEXT:    movl %esi, %eax
; X86-O0-NEXT:    movq %rax, _ZN8struct_210member_2_0E
; X86-O0-NEXT:    retq
;
; X64-LABEL: f1:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movslq {{.*}}(%rip), %rax
; X64-NEXT:    movabsq $-8381627093, %rcx # imm = 0xFFFFFFFE0C6A852B
; X64-NEXT:    cmpq %rcx, %rax
; X64-NEXT:    setne -{{[0-9]+}}(%rsp)
; X64-NEXT:    xorl %ecx, %ecx
; X64-NEXT:    cmpq $-1, %rax
; X64-NEXT:    sete %cl
; X64-NEXT:    xorl %edx, %edx
; X64-NEXT:    cmpl $-1, %eax
; X64-NEXT:    sete %dl
; X64-NEXT:    addq $7093, %rax # imm = 0x1BB5
; X64-NEXT:    xorl %esi, %esi
; X64-NEXT:    cmpq %rax, %rdx
; X64-NEXT:    setg %sil
; X64-NEXT:    movq %rsi, {{.*}}(%rip)
; X64-NEXT:    movq %rcx, {{.*}}(%rip)
; X64-NEXT:    retq
;
; 686-O0-LABEL: f1:
; 686-O0:       # %bb.0: # %entry
; 686-O0-NEXT:    pushl %ebp
; 686-O0-NEXT:    .cfi_def_cfa_offset 8
; 686-O0-NEXT:    pushl %ebx
; 686-O0-NEXT:    .cfi_def_cfa_offset 12
; 686-O0-NEXT:    pushl %edi
; 686-O0-NEXT:    .cfi_def_cfa_offset 16
; 686-O0-NEXT:    pushl %esi
; 686-O0-NEXT:    .cfi_def_cfa_offset 20
; 686-O0-NEXT:    subl $24, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 44
; 686-O0-NEXT:    .cfi_offset %esi, -20
; 686-O0-NEXT:    .cfi_offset %edi, -16
; 686-O0-NEXT:    .cfi_offset %ebx, -12
; 686-O0-NEXT:    .cfi_offset %ebp, -8
; 686-O0-NEXT:    movl var_5, %eax
; 686-O0-NEXT:    movl %eax, %ecx
; 686-O0-NEXT:    sarl $31, %ecx
; 686-O0-NEXT:    xorl $208307499, %eax # imm = 0xC6A852B
; 686-O0-NEXT:    xorl $-2, %ecx
; 686-O0-NEXT:    orl %ecx, %eax
; 686-O0-NEXT:    setne {{[0-9]+}}(%esp)
; 686-O0-NEXT:    movl var_5, %ecx
; 686-O0-NEXT:    movl %ecx, %edx
; 686-O0-NEXT:    sarl $31, %edx
; 686-O0-NEXT:    movl %ecx, %esi
; 686-O0-NEXT:    subl $-1, %esi
; 686-O0-NEXT:    sete %bl
; 686-O0-NEXT:    movzbl %bl, %edi
; 686-O0-NEXT:    addl $7093, %ecx # imm = 0x1BB5
; 686-O0-NEXT:    adcl $0, %edx
; 686-O0-NEXT:    subl %edi, %ecx
; 686-O0-NEXT:    sbbl $0, %edx
; 686-O0-NEXT:    setl %bl
; 686-O0-NEXT:    movzbl %bl, %edi
; 686-O0-NEXT:    movl %edi, var_57
; 686-O0-NEXT:    movl $0, var_57+4
; 686-O0-NEXT:    movl var_5, %edi
; 686-O0-NEXT:    subl $-1, %edi
; 686-O0-NEXT:    sete %bl
; 686-O0-NEXT:    movzbl %bl, %ebp
; 686-O0-NEXT:    movl %ebp, _ZN8struct_210member_2_0E
; 686-O0-NEXT:    movl $0, _ZN8struct_210member_2_0E+4
; 686-O0-NEXT:    movl %eax, {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Spill
; 686-O0-NEXT:    movl %esi, {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Spill
; 686-O0-NEXT:    movl %ecx, {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Spill
; 686-O0-NEXT:    movl %edx, {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Spill
; 686-O0-NEXT:    movl %edi, (%esp) # 4-byte Spill
; 686-O0-NEXT:    addl $24, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 20
; 686-O0-NEXT:    popl %esi
; 686-O0-NEXT:    .cfi_def_cfa_offset 16
; 686-O0-NEXT:    popl %edi
; 686-O0-NEXT:    .cfi_def_cfa_offset 12
; 686-O0-NEXT:    popl %ebx
; 686-O0-NEXT:    .cfi_def_cfa_offset 8
; 686-O0-NEXT:    popl %ebp
; 686-O0-NEXT:    .cfi_def_cfa_offset 4
; 686-O0-NEXT:    retl
;
; 686-LABEL: f1:
; 686:       # %bb.0: # %entry
; 686-NEXT:    pushl %esi
; 686-NEXT:    .cfi_def_cfa_offset 8
; 686-NEXT:    subl $1, %esp
; 686-NEXT:    .cfi_def_cfa_offset 9
; 686-NEXT:    .cfi_offset %esi, -8
; 686-NEXT:    movl var_5, %edx
; 686-NEXT:    movl %edx, %eax
; 686-NEXT:    xorl $208307499, %eax # imm = 0xC6A852B
; 686-NEXT:    movl %edx, %esi
; 686-NEXT:    sarl $31, %esi
; 686-NEXT:    movl %esi, %ecx
; 686-NEXT:    xorl $-2, %ecx
; 686-NEXT:    orl %eax, %ecx
; 686-NEXT:    setne (%esp)
; 686-NEXT:    movl %edx, %ecx
; 686-NEXT:    andl %esi, %ecx
; 686-NEXT:    xorl %eax, %eax
; 686-NEXT:    cmpl $-1, %ecx
; 686-NEXT:    sete %al
; 686-NEXT:    xorl %ecx, %ecx
; 686-NEXT:    cmpl $-1, %edx
; 686-NEXT:    sete %cl
; 686-NEXT:    addl $7093, %edx # imm = 0x1BB5
; 686-NEXT:    adcl $0, %esi
; 686-NEXT:    cmpl %ecx, %edx
; 686-NEXT:    sbbl $0, %esi
; 686-NEXT:    setl %cl
; 686-NEXT:    movzbl %cl, %ecx
; 686-NEXT:    movl %ecx, var_57
; 686-NEXT:    movl $0, var_57+4
; 686-NEXT:    movl %eax, _ZN8struct_210member_2_0E
; 686-NEXT:    movl $0, _ZN8struct_210member_2_0E+4
; 686-NEXT:    addl $1, %esp
; 686-NEXT:    .cfi_def_cfa_offset 8
; 686-NEXT:    popl %esi
; 686-NEXT:    .cfi_def_cfa_offset 4
; 686-NEXT:    retl
entry:
  %a = alloca i8, align 1
  %0 = load i32, i32* @var_5, align 4
  %conv = sext i32 %0 to i64
  %add = add nsw i64 %conv, 8381627093
  %tobool = icmp ne i64 %add, 0
  %frombool = zext i1 %tobool to i8
  store i8 %frombool, i8* %a, align 1
  %1 = load i32, i32* @var_5, align 4
  %neg = xor i32 %1, -1
  %tobool1 = icmp ne i32 %neg, 0
  %lnot = xor i1 %tobool1, true
  %conv2 = zext i1 %lnot to i64
  %2 = load i32, i32* @var_5, align 4
  %conv3 = sext i32 %2 to i64
  %add4 = add nsw i64 %conv3, 7093
  %cmp = icmp sgt i64 %conv2, %add4
  %conv5 = zext i1 %cmp to i64
  store i64 %conv5, i64* @var_57, align 8
  %3 = load i32, i32* @var_5, align 4
  %neg6 = xor i32 %3, -1
  %tobool7 = icmp ne i32 %neg6, 0
  %lnot8 = xor i1 %tobool7, true
  %conv9 = zext i1 %lnot8 to i64
  store i64 %conv9, i64* @_ZN8struct_210member_2_0E, align 8
  ret void
}


@var_7 = external global i8, align 1

define void @f2() {
; X86-O0-LABEL: f2:
; X86-O0:       # %bb.0: # %entry
; X86-O0-NEXT:    movzbl var_7, %eax
; X86-O0-NEXT:    cmpb $0, var_7
; X86-O0-NEXT:    setne %cl
; X86-O0-NEXT:    xorb $-1, %cl
; X86-O0-NEXT:    andb $1, %cl
; X86-O0-NEXT:    movzbl %cl, %edx
; X86-O0-NEXT:    xorl %edx, %eax
; X86-O0-NEXT:    movw %ax, %si
; X86-O0-NEXT:    movw %si, -{{[0-9]+}}(%rsp)
; X86-O0-NEXT:    movzbl var_7, %eax
; X86-O0-NEXT:    movw %ax, %si
; X86-O0-NEXT:    cmpw $0, %si
; X86-O0-NEXT:    setne %cl
; X86-O0-NEXT:    xorb $-1, %cl
; X86-O0-NEXT:    andb $1, %cl
; X86-O0-NEXT:    movzbl %cl, %eax
; X86-O0-NEXT:    movzbl var_7, %edx
; X86-O0-NEXT:    cmpl %edx, %eax
; X86-O0-NEXT:    sete %cl
; X86-O0-NEXT:    andb $1, %cl
; X86-O0-NEXT:    movzbl %cl, %eax
; X86-O0-NEXT:    movw %ax, %si
; X86-O0-NEXT:    # implicit-def: $rdi
; X86-O0-NEXT:    movw %si, (%rdi)
; X86-O0-NEXT:    retq
;
; X64-LABEL: f2:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movzbl {{.*}}(%rip), %eax
; X64-NEXT:    xorl %ecx, %ecx
; X64-NEXT:    testl %eax, %eax
; X64-NEXT:    sete %cl
; X64-NEXT:    xorl %eax, %ecx
; X64-NEXT:    movw %cx, -{{[0-9]+}}(%rsp)
; X64-NEXT:    xorl %ecx, %ecx
; X64-NEXT:    testb %al, %al
; X64-NEXT:    sete %cl
; X64-NEXT:    xorl %edx, %edx
; X64-NEXT:    cmpl %eax, %ecx
; X64-NEXT:    sete %dl
; X64-NEXT:    movw %dx, (%rax)
; X64-NEXT:    retq
;
; 686-O0-LABEL: f2:
; 686-O0:       # %bb.0: # %entry
; 686-O0-NEXT:    pushl %esi
; 686-O0-NEXT:    .cfi_def_cfa_offset 8
; 686-O0-NEXT:    subl $2, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 10
; 686-O0-NEXT:    .cfi_offset %esi, -8
; 686-O0-NEXT:    movzbl var_7, %eax
; 686-O0-NEXT:    cmpb $0, var_7
; 686-O0-NEXT:    setne %cl
; 686-O0-NEXT:    xorb $-1, %cl
; 686-O0-NEXT:    andb $1, %cl
; 686-O0-NEXT:    movzbl %cl, %edx
; 686-O0-NEXT:    xorl %edx, %eax
; 686-O0-NEXT:    movw %ax, %si
; 686-O0-NEXT:    movw %si, (%esp)
; 686-O0-NEXT:    movzbl var_7, %eax
; 686-O0-NEXT:    movw %ax, %si
; 686-O0-NEXT:    cmpw $0, %si
; 686-O0-NEXT:    setne %cl
; 686-O0-NEXT:    xorb $-1, %cl
; 686-O0-NEXT:    andb $1, %cl
; 686-O0-NEXT:    movzbl %cl, %eax
; 686-O0-NEXT:    movzbl var_7, %edx
; 686-O0-NEXT:    cmpl %edx, %eax
; 686-O0-NEXT:    sete %cl
; 686-O0-NEXT:    andb $1, %cl
; 686-O0-NEXT:    movzbl %cl, %eax
; 686-O0-NEXT:    movw %ax, %si
; 686-O0-NEXT:    # implicit-def: $eax
; 686-O0-NEXT:    movw %si, (%eax)
; 686-O0-NEXT:    addl $2, %esp
; 686-O0-NEXT:    .cfi_def_cfa_offset 8
; 686-O0-NEXT:    popl %esi
; 686-O0-NEXT:    .cfi_def_cfa_offset 4
; 686-O0-NEXT:    retl
;
; 686-LABEL: f2:
; 686:       # %bb.0: # %entry
; 686-NEXT:    subl $2, %esp
; 686-NEXT:    .cfi_def_cfa_offset 6
; 686-NEXT:    movzbl var_7, %eax
; 686-NEXT:    xorl %ecx, %ecx
; 686-NEXT:    testl %eax, %eax
; 686-NEXT:    sete %cl
; 686-NEXT:    xorl %eax, %ecx
; 686-NEXT:    movw %cx, (%esp)
; 686-NEXT:    xorl %ecx, %ecx
; 686-NEXT:    testb %al, %al
; 686-NEXT:    sete %cl
; 686-NEXT:    xorl %edx, %edx
; 686-NEXT:    cmpl %eax, %ecx
; 686-NEXT:    sete %dl
; 686-NEXT:    movw %dx, (%eax)
; 686-NEXT:    addl $2, %esp
; 686-NEXT:    .cfi_def_cfa_offset 4
; 686-NEXT:    retl
entry:
  %a = alloca i16, align 2
  %0 = load i8, i8* @var_7, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* @var_7, align 1
  %tobool = icmp ne i8 %1, 0
  %lnot = xor i1 %tobool, true
  %conv1 = zext i1 %lnot to i32
  %xor = xor i32 %conv, %conv1
  %conv2 = trunc i32 %xor to i16
  store i16 %conv2, i16* %a, align 2
  %2 = load i8, i8* @var_7, align 1
  %conv3 = zext i8 %2 to i16
  %tobool4 = icmp ne i16 %conv3, 0
  %lnot5 = xor i1 %tobool4, true
  %conv6 = zext i1 %lnot5 to i32
  %3 = load i8, i8* @var_7, align 1
  %conv7 = zext i8 %3 to i32
  %cmp = icmp eq i32 %conv6, %conv7
  %conv8 = zext i1 %cmp to i32
  %conv9 = trunc i32 %conv8 to i16
  store i16 %conv9, i16* undef, align 2
  ret void
}


@var_13 = external global i32, align 4
@var_16 = external global i32, align 4
@var_46 = external global i32, align 4

define void @f3() #0 {
; X86-O0-LABEL: f3:
; X86-O0:       # %bb.0: # %entry
; X86-O0-NEXT:    movl var_13, %eax
; X86-O0-NEXT:    xorl $-1, %eax
; X86-O0-NEXT:    movl %eax, %eax
; X86-O0-NEXT:    movl %eax, %ecx
; X86-O0-NEXT:    cmpl $0, var_13
; X86-O0-NEXT:    setne %dl
; X86-O0-NEXT:    xorb $-1, %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movzbl %dl, %eax
; X86-O0-NEXT:    movl %eax, %esi
; X86-O0-NEXT:    movl var_13, %eax
; X86-O0-NEXT:    xorl $-1, %eax
; X86-O0-NEXT:    xorl var_16, %eax
; X86-O0-NEXT:    movl %eax, %eax
; X86-O0-NEXT:    movl %eax, %edi
; X86-O0-NEXT:    andq %rdi, %rsi
; X86-O0-NEXT:    orq %rsi, %rcx
; X86-O0-NEXT:    movq %rcx, -{{[0-9]+}}(%rsp)
; X86-O0-NEXT:    movl var_13, %eax
; X86-O0-NEXT:    xorl $-1, %eax
; X86-O0-NEXT:    movl %eax, %eax
; X86-O0-NEXT:    movl %eax, %ecx
; X86-O0-NEXT:    cmpl $0, var_13
; X86-O0-NEXT:    setne %dl
; X86-O0-NEXT:    xorb $-1, %dl
; X86-O0-NEXT:    andb $1, %dl
; X86-O0-NEXT:    movzbl %dl, %eax
; X86-O0-NEXT:    movl %eax, %esi
; X86-O0-NEXT:    andq $0, %rsi
; X86-O0-NEXT:    orq %rsi, %rcx
; X86-O0-NEXT:    movl %ecx, %eax
; X86-O0-NEXT:    movl %eax, var_46
; X86-O0-NEXT:    retq
;
; X64-LABEL: f3:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movl {{.*}}(%rip), %eax
; X64-NEXT:    xorl %ecx, %ecx
; X64-NEXT:    testl %eax, %eax
; X64-NEXT:    notl %eax
; X64-NEXT:    sete %cl
; X64-NEXT:    movl {{.*}}(%rip), %edx
; X64-NEXT:    xorl %eax, %edx
; X64-NEXT:    andl %edx, %ecx
; X64-NEXT:    orl %eax, %ecx
; X64-NEXT:    movq %rcx, -{{[0-9]+}}(%rsp)
; X64-NEXT:    movl %eax, {{.*}}(%rip)
; X64-NEXT:    retq
;
; 686-O0-LABEL: f3:
; 686-O0:       # %bb.0: # %entry
; 686-O0-NEXT:    pushl %ebp
; 686-O0-NEXT:    .cfi_def_cfa_offset 8
; 686-O0-NEXT:    .cfi_offset %ebp, -8
; 686-O0-NEXT:    movl %esp, %ebp
; 686-O0-NEXT:    .cfi_def_cfa_register %ebp
; 686-O0-NEXT:    pushl %edi
; 686-O0-NEXT:    pushl %esi
; 686-O0-NEXT:    andl $-8, %esp
; 686-O0-NEXT:    subl $8, %esp
; 686-O0-NEXT:    .cfi_offset %esi, -16
; 686-O0-NEXT:    .cfi_offset %edi, -12
; 686-O0-NEXT:    movl var_13, %eax
; 686-O0-NEXT:    movl %eax, %ecx
; 686-O0-NEXT:    notl %ecx
; 686-O0-NEXT:    testl %eax, %eax
; 686-O0-NEXT:    sete %dl
; 686-O0-NEXT:    movzbl %dl, %eax
; 686-O0-NEXT:    movl var_16, %esi
; 686-O0-NEXT:    movl %ecx, %edi
; 686-O0-NEXT:    xorl %esi, %edi
; 686-O0-NEXT:    andl %edi, %eax
; 686-O0-NEXT:    orl %eax, %ecx
; 686-O0-NEXT:    movl %ecx, (%esp)
; 686-O0-NEXT:    movl $0, {{[0-9]+}}(%esp)
; 686-O0-NEXT:    movl var_13, %eax
; 686-O0-NEXT:    notl %eax
; 686-O0-NEXT:    movl %eax, var_46
; 686-O0-NEXT:    leal -8(%ebp), %esp
; 686-O0-NEXT:    popl %esi
; 686-O0-NEXT:    popl %edi
; 686-O0-NEXT:    popl %ebp
; 686-O0-NEXT:    .cfi_def_cfa %esp, 4
; 686-O0-NEXT:    retl
;
; 686-LABEL: f3:
; 686:       # %bb.0: # %entry
; 686-NEXT:    pushl %ebp
; 686-NEXT:    .cfi_def_cfa_offset 8
; 686-NEXT:    .cfi_offset %ebp, -8
; 686-NEXT:    movl %esp, %ebp
; 686-NEXT:    .cfi_def_cfa_register %ebp
; 686-NEXT:    andl $-8, %esp
; 686-NEXT:    subl $8, %esp
; 686-NEXT:    movl var_13, %ecx
; 686-NEXT:    xorl %eax, %eax
; 686-NEXT:    testl %ecx, %ecx
; 686-NEXT:    notl %ecx
; 686-NEXT:    sete %al
; 686-NEXT:    movl var_16, %edx
; 686-NEXT:    xorl %ecx, %edx
; 686-NEXT:    andl %eax, %edx
; 686-NEXT:    orl %ecx, %edx
; 686-NEXT:    movl %edx, (%esp)
; 686-NEXT:    movl $0, {{[0-9]+}}(%esp)
; 686-NEXT:    movl %ecx, var_46
; 686-NEXT:    movl %ebp, %esp
; 686-NEXT:    popl %ebp
; 686-NEXT:    .cfi_def_cfa %esp, 4
; 686-NEXT:    retl
entry:
  %a = alloca i64, align 8
  %0 = load i32, i32* @var_13, align 4
  %neg = xor i32 %0, -1
  %conv = zext i32 %neg to i64
  %1 = load i32, i32* @var_13, align 4
  %tobool = icmp ne i32 %1, 0
  %lnot = xor i1 %tobool, true
  %conv1 = zext i1 %lnot to i64
  %2 = load i32, i32* @var_13, align 4
  %neg2 = xor i32 %2, -1
  %3 = load i32, i32* @var_16, align 4
  %xor = xor i32 %neg2, %3
  %conv3 = zext i32 %xor to i64
  %and = and i64 %conv1, %conv3
  %or = or i64 %conv, %and
  store i64 %or, i64* %a, align 8
  %4 = load i32, i32* @var_13, align 4
  %neg4 = xor i32 %4, -1
  %conv5 = zext i32 %neg4 to i64
  %5 = load i32, i32* @var_13, align 4
  %tobool6 = icmp ne i32 %5, 0
  %lnot7 = xor i1 %tobool6, true
  %conv8 = zext i1 %lnot7 to i64
  %and9 = and i64 %conv8, 0
  %or10 = or i64 %conv5, %and9
  %conv11 = trunc i64 %or10 to i32
  store i32 %conv11, i32* @var_46, align 4
  ret void
}

