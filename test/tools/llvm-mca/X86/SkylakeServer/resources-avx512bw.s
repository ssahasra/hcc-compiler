# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512 -instruction-tables < %s | FileCheck %s

vpabsb            %zmm16, %zmm19
vpabsb            (%rax), %zmm19
vpabsb            %zmm16, %zmm19 {k1}
vpabsb            (%rax), %zmm19 {k1}
vpabsb            %zmm16, %zmm19 {z}{k1}
vpabsb            (%rax), %zmm19 {z}{k1}

vpabsw            %zmm16, %zmm19
vpabsw            (%rax), %zmm19
vpabsw            %zmm16, %zmm19 {k1}
vpabsw            (%rax), %zmm19 {k1}
vpabsw            %zmm16, %zmm19 {z}{k1}
vpabsw            (%rax), %zmm19 {z}{k1}

vpaddb            %zmm16, %zmm17, %zmm19
vpaddb            (%rax), %zmm17, %zmm19
vpaddb            %zmm16, %zmm17, %zmm19 {k1}
vpaddb            (%rax), %zmm17, %zmm19 {k1}
vpaddb            %zmm16, %zmm17, %zmm19 {z}{k1}
vpaddb            (%rax), %zmm17, %zmm19 {z}{k1}

vpaddw            %zmm16, %zmm17, %zmm19
vpaddw            (%rax), %zmm17, %zmm19
vpaddw            %zmm16, %zmm17, %zmm19 {k1}
vpaddw            (%rax), %zmm17, %zmm19 {k1}
vpaddw            %zmm16, %zmm17, %zmm19 {z}{k1}
vpaddw            (%rax), %zmm17, %zmm19 {z}{k1}

vpsubb            %zmm16, %zmm17, %zmm19
vpsubb            (%rax), %zmm17, %zmm19
vpsubb            %zmm16, %zmm17, %zmm19 {k1}
vpsubb            (%rax), %zmm17, %zmm19 {k1}
vpsubb            %zmm16, %zmm17, %zmm19 {z}{k1}
vpsubb            (%rax), %zmm17, %zmm19 {z}{k1}

vpsubw            %zmm16, %zmm17, %zmm19
vpsubw            (%rax), %zmm17, %zmm19
vpsubw            %zmm16, %zmm17, %zmm19 {k1}
vpsubw            (%rax), %zmm17, %zmm19 {k1}
vpsubw            %zmm16, %zmm17, %zmm19 {z}{k1}
vpsubw            (%rax), %zmm17, %zmm19 {z}{k1}

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects (U)

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT:  1      1     1.00                        vpabsb	%zmm16, %zmm19
# CHECK-NEXT:  2      8     1.00    *                   vpabsb	(%rax), %zmm19
# CHECK-NEXT:  1      1     1.00                        vpabsb	%zmm16, %zmm19 {%k1}
# CHECK-NEXT:  2      8     1.00    *                   vpabsb	(%rax), %zmm19 {%k1}
# CHECK-NEXT:  1      1     1.00                        vpabsb	%zmm16, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     1.00    *                   vpabsb	(%rax), %zmm19 {%k1} {z}
# CHECK-NEXT:  1      1     1.00                        vpabsw	%zmm16, %zmm19
# CHECK-NEXT:  2      8     1.00    *                   vpabsw	(%rax), %zmm19
# CHECK-NEXT:  1      1     1.00                        vpabsw	%zmm16, %zmm19 {%k1}
# CHECK-NEXT:  2      8     1.00    *                   vpabsw	(%rax), %zmm19 {%k1}
# CHECK-NEXT:  1      1     1.00                        vpabsw	%zmm16, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     1.00    *                   vpabsw	(%rax), %zmm19 {%k1} {z}
# CHECK-NEXT:  1      1     0.33                        vpaddb	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  2      8     0.50    *                   vpaddb	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  1      1     0.33                        vpaddb	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  2      8     0.50    *                   vpaddb	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  1      1     0.33                        vpaddb	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     0.50    *                   vpaddb	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  1      1     0.33                        vpaddw	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  2      8     0.50    *                   vpaddw	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  1      1     0.33                        vpaddw	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  2      8     0.50    *                   vpaddw	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  1      1     0.33                        vpaddw	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     0.50    *                   vpaddw	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  1      1     0.33                        vpsubb	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  2      8     0.50    *                   vpsubb	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  1      1     0.33                        vpsubb	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  2      8     0.50    *                   vpsubb	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  1      1     0.33                        vpsubb	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     0.50    *                   vpsubb	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  1      1     0.33                        vpsubw	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  2      8     0.50    *                   vpsubw	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  1      1     0.33                        vpsubw	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  2      8     0.50    *                   vpsubw	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  1      1     0.33                        vpsubw	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  2      8     0.50    *                   vpsubw	(%rax), %zmm17, %zmm19 {%k1} {z}

# CHECK:      Resources:
# CHECK-NEXT: [0]   - SKXDivider
# CHECK-NEXT: [1]   - SKXFPDivider
# CHECK-NEXT: [2]   - SKXPort0
# CHECK-NEXT: [3]   - SKXPort1
# CHECK-NEXT: [4]   - SKXPort2
# CHECK-NEXT: [5]   - SKXPort3
# CHECK-NEXT: [6]   - SKXPort4
# CHECK-NEXT: [7]   - SKXPort5
# CHECK-NEXT: [8]   - SKXPort6
# CHECK-NEXT: [9]   - SKXPort7

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]
# CHECK-NEXT:  -      -     20.00  8.00   9.00   9.00    -     8.00    -      -

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    Instructions:
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsb	%zmm16, %zmm19
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsb	(%rax), %zmm19
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsb	%zmm16, %zmm19 {%k1}
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsb	(%rax), %zmm19 {%k1}
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsb	%zmm16, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsb	(%rax), %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsw	%zmm16, %zmm19
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsw	(%rax), %zmm19
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsw	%zmm16, %zmm19 {%k1}
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsw	(%rax), %zmm19 {%k1}
# CHECK-NEXT:  -      -     1.00    -      -      -      -      -      -      -     vpabsw	%zmm16, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     1.00    -     0.50   0.50    -      -      -      -     vpabsw	(%rax), %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddb	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddb	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddb	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddb	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddb	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddb	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddw	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddw	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddw	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddw	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpaddw	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpaddw	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubb	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubb	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubb	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubb	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubb	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubb	(%rax), %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubw	%zmm16, %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubw	(%rax), %zmm17, %zmm19
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubw	%zmm16, %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubw	(%rax), %zmm17, %zmm19 {%k1}
# CHECK-NEXT:  -      -     0.33   0.33    -      -      -     0.33    -      -     vpsubw	%zmm16, %zmm17, %zmm19 {%k1} {z}
# CHECK-NEXT:  -      -     0.33   0.33   0.50   0.50    -     0.33    -      -     vpsubw	(%rax), %zmm17, %zmm19 {%k1} {z}
