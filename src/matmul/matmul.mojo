from algorithm import vectorize, parallelize
from memory.memory import _malloc
from sys import has_avx512f, num_performance_cores
import benchmark
from testing import assert_equal
from sys.info import simdwidthof, sizeof
from memory import stack_allocation
from utils.index import StaticIntTuple
from collections import InlineArray
import random


@always_inline
fn roundup(a: Int, b: Int) -> Int:
    return ((a + b - 1) // b) * b


@always_inline
fn rounddown(a: Int, b: Int) -> Int:
    return (a // b) * b


# math.sqrt doesn't work at compile time
fn intsqrt[n: Int]() -> Int:
    @parameter
    if n == 0:
        return 0
    var x = n
    var y = (x + 1) // 2
    while y < x:
        x = y
        y = (n // x + x) // 2
    return x


@value
@register_passable("trivial")
struct Layout:
    var shape: StaticIntTuple[2]
    var strides: StaticIntTuple[2]

    fn __init__(inout self, shape: (Int, Int), strides: (Int, Int)):
        self.shape = StaticIntTuple[2](shape[0], shape[1])
        self.strides = StaticIntTuple[2](strides[0], strides[1])

    fn __init__(inout self, shape: (Int, Int)):
        self.strides = StaticIntTuple[2](shape[1], 1)
        self.shape = StaticIntTuple[2](shape[0], shape[1])

    @always_inline("nodebug")
    fn __call__(self, i: Int, j: Int) -> Int:
        return i * self.strides[0] + j * self.strides[1]

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self.shape[0] * self.shape[1]

    @always_inline("nodebug")
    fn format_to(self, inout writer: Formatter):
        writer.write(self.shape, ":", self.strides, "\n")


struct Matrix[Type: DType]:
    var data: UnsafePointer[Scalar[Type]]
    var layout: Layout

    fn __init__(inout self, shape: (Int, Int)):
        self.data = UnsafePointer[Scalar[Type]].alloc(shape[0] * shape[1])
        self.layout = Layout(shape)

    @always_inline("nodebug")
    fn __init__(
        inout self, data: UnsafePointer[Scalar[Type]], owned layout: Layout
    ):
        self.data = UnsafePointer[Scalar[Type]](data)
        self.layout = layout

    @always_inline("nodebug")
    fn __init__(
        inout self, data: UnsafePointer[Scalar[Type]], shape: (Int, Int)
    ):
        self.data = data
        self.layout = Layout(shape)

    @always_inline("nodebug")
    fn __getitem__(
        ref [_]self, i: Int, j: Int
    ) -> ref [__lifetime_of(self)] Scalar[Type]:
        var offset = self.layout(i, j)
        return (self.data + offset)[]

    @always_inline("nodebug")
    fn slice(self, i: Int, j: Int, ir: Int, jr: Int) -> Self:
        var shape = (ir, jr)
        var strides = (self.layout.strides[0], self.layout.strides[1])
        var offset = self.layout(i, j)
        return Matrix(self.data + offset, Layout(shape, strides))

    @always_inline("nodebug")
    fn shape[dim: Int](self) -> Int:
        return self.layout.shape[dim]

    @always_inline("nodebug")
    fn stride[dim: Int](self) -> Int:
        return self.layout.strides[dim]

    fn rand(inout self):
        random.rand(self.data, self.layout.size())

    @always_inline("nodebug")
    fn load[width: Int, *, dim: Int](self, i: Int, j: Int) -> SIMD[Type, width]:
        var offset = self.layout(i, j)
        var ptr = self.data + offset

        @parameter
        if dim == 0:
            return ptr.strided_load[width=width](self.layout.strides[0])
        else:
            return ptr.load[Type, width]()

    @always_inline("nodebug")
    fn store[
        width: Int, *, dim: Int
    ](self, value: SIMD[Type, width], i: Int, j: Int):
        var offset = self.layout(i, j)
        var ptr = self.data + offset

        @parameter
        if dim == 0:
            ptr.strided_store[width=width](value, self.layout.strides[0])
        else:
            ptr.store[Type, width](value)

    fn format_to(self, inout writer: Formatter):
        writer.write(
            "Matrix: ",
            str(self.data),
            ", Layout: ",
            self.layout,
            "\n",
        )
        for i in range(self.layout.shape[0]):
            for j in range(self.layout.shape[1]):
                writer.write(self[i, j], " ")
            writer.write("\n")


@always_inline
fn pack_A[
    Type: DType, //, mc: Int, mr: Int
](Ac_buffer: UnsafePointer[Scalar[Type]], Ac: Matrix[Type]) -> Matrix[Type]:
    @parameter
    fn pack_panel(idx: Int):
        var i = idx * mr
        # for i in range(0, Ac.shape[0](), mr):
        var dst_ptr = Ac_buffer + i * Ac.shape[1]()
        var src_ptr = Ac.data + i * Ac.stride[0]()
        for _ in range(Ac.shape[1]()):

            @parameter
            fn pack_col[width: Int](l: Int):
                (dst_ptr + l).store[Type, width](
                    (src_ptr + l * Ac.stride[0]()).strided_load[
                        width=width
                    ](Ac.stride[0]()),
                )

            vectorize[pack_col, simdwidthof[Type]()](min(Ac.shape[0]() - i, mr))

            for l in range(min(Ac.shape[0]() - i, mr), mr):
                dst_ptr[l] = Scalar[Type](0)

            dst_ptr = dst_ptr + mr
            src_ptr = src_ptr + 1

    parallelize[pack_panel](
        (Ac.shape[0]() + mr - 1) // mr, num_performance_cores()
    )

    var Ac_layout = Layout(
        (roundup(Ac.shape[0](), mr), Ac.shape[1]()), (1, mr)
    )  # NOTE: The stride is a lie and only used for slicing
    return Matrix(Ac_buffer, Ac_layout)


@always_inline
fn pack_B[
    Type: DType, //, kc: Int, nr: Int
](Bc_buffer: UnsafePointer[Scalar[Type]], Bc: Matrix[Type]) -> Matrix[Type]:
    var dst_ptr = Bc_buffer
    for i in range(0, Bc.shape[1](), nr):
        var src_ptr = Bc.data + i
        for _ in range(Bc.shape[0]()):

            @parameter
            fn pack_row[width: Int](l: Int):
                (dst_ptr + l).store[Type, width][
                    alignment = sizeof[Type]() * simdwidthof[Type]()
                ](
                    (src_ptr + l).load[Type, width](),
                )

            vectorize[
                pack_row,
                simdwidthof[Type](),
                unroll_factor = nr // simdwidthof[Type](),
            ](min(Bc.shape[1]() - i, nr))

            for l in range(min(Bc.shape[1]() - i, nr), nr):
                dst_ptr[l] = Scalar[Type](0)

            dst_ptr = dst_ptr + nr
            src_ptr = src_ptr + Bc.stride[0]()

    var Bc_layout = Layout(
        (Bc.shape[0](), roundup(Bc.shape[1](), nr)), (nr, 1)
    )  # NOTE: The stride is a lie and only used for slicing
    return Matrix[Type](Bc_buffer, Bc_layout)


@always_inline
fn matmul_impl[
    Type: DType, //,
    mc: Int,
    nc: Int,
    kc: Int,
    mr: Int,
    nr: Int,
](inout C: Matrix[Type], A: Matrix[Type], B: Matrix[Type]):
    var Ac_buffer = _malloc[Scalar[Type], alignment=64](
        mc * kc * sizeof[Type]()
    )

    var M = C.shape[0]()
    var N = C.shape[1]()
    var K = A.shape[1]()

    for i in range(0, A.shape[0](), mc):
        var Cb = C.slice(i, 0, min(M - i, mc), N)
        for p in range(0, A.shape[1](), kc):
            var Ac = pack_A[mc, mr](
                Ac_buffer, A.slice(i, p, min(M - i, mc), min(K - p, kc))
            )

            var Bb = B.slice(p, 0, min(K - p, kc), N)
            loop_n[nc, kc, mr, nr](Cb, Ac, Bb)

    Ac_buffer.free()


@always_inline
fn loop_n[
    Type: DType, //,
    nc: Int,
    kc: Int,
    mr: Int,
    nr: Int,
](inout C: Matrix[Type], A: Matrix[Type], B: Matrix[Type]):
    var max_threads = num_performance_cores()
    var nc_per_thread = nc if nc * max_threads <= B.shape[1]() else rounddown(
        B.shape[1]() // max_threads, nr
    )
    var balanced_part = rounddown(B.shape[1](), nc_per_thread)

    var remainder = B.shape[1]() - balanced_part
    var remainder_per_thread = rounddown(remainder // max_threads, nr)
    remainder_per_thread = max(remainder_per_thread, nr)

    var items_remainder = (
        remainder + remainder_per_thread - 1
    ) // remainder_per_thread

    @parameter
    fn parallelize_balanced_part(idx: Int):
        var Bc_buffer = UnsafePointer[Scalar[Type]](
            _malloc[Scalar[Type], alignment=64](
                kc * nc_per_thread * sizeof[Type]()
            )
        )

        var j = idx * nc_per_thread
        var Bc = pack_B[kc, nr](
            Bc_buffer,
            B.slice(0, j, B.shape[0](), min(B.shape[1]() - j, nc_per_thread)),
        )
        var Cc = C.slice(
            0, j, C.shape[0](), min(C.shape[1]() - j, nc_per_thread)
        )
        macro_kernel[mr, nr](Cc, A, Bc)
        Bc_buffer.free()

    parallelize[parallelize_balanced_part](
        balanced_part // nc_per_thread, balanced_part // nc_per_thread
    )

    @parameter
    fn parallelize_remainder(idx: Int):
        var Bc_buffer = UnsafePointer[Scalar[Type]](
            _malloc[Scalar[Type], alignment=64](
                kc * remainder_per_thread * sizeof[Type]()
            )
        )
        var j = balanced_part + idx * remainder_per_thread
        var Bc = pack_B[kc, nr](
            Bc_buffer,
            B.slice(
                0, j, B.shape[0](), min(B.shape[1]() - j, remainder_per_thread)
            ),
        )
        var Cc = C.slice(
            0, j, C.shape[0](), min(C.shape[1]() - j, remainder_per_thread)
        )
        macro_kernel[mr, nr](Cc, A, Bc)
        Bc_buffer.free()

    parallelize[parallelize_remainder](items_remainder, items_remainder)

    _ = balanced_part
    _ = remainder_per_thread
    _ = nc_per_thread


@always_inline
fn macro_kernel[
    Type: DType, //,
    mr: Int,
    nr: Int,
](inout Cc: Matrix[Type], Ac: Matrix[Type], Bc: Matrix[Type]):
    @parameter
    fn parallelize_ir(idx: Int):
        var ir = idx * mr
        var Ar = Matrix(Ac.data + ir * Ac.shape[1](), (mr, Ac.shape[1]()))
        for jr in range(0, Bc.shape[1](), nr):
            var Cr = Cc.slice(
                ir,
                jr,
                min(Cc.shape[0]() - ir, mr),
                min(Cc.shape[1]() - jr, nr),
            )
            var Br = Matrix(
                Bc.data + jr * Bc.shape[0](),
                (Bc.shape[0](), nr),
            )
            if Cr.shape[0]() == mr and Cr.shape[1]() == nr:
                micro_kernel[mr, nr, False](Cr, Ar, Br)
            else:
                micro_kernel[mr, nr, True](Cr, Ar, Br)

    parallelize[parallelize_ir]((Ac.shape[0]() + mr - 1) // mr, 2)


@always_inline
fn micro_kernel[
    Type: DType, //, mr: Int, nr: Int, padding: Bool
](inout Cr: Matrix[Type], Ar: Matrix[Type], Br: Matrix[Type]):
    alias simd_width = simdwidthof[Type]()
    constrained[nr % simd_width == 0, "nr must be multiple of simd_width"]()

    var Ar_ptr = Ar.data
    var Br_ptr = Br.data
    var Cr_ptr = Cr.data

    var ar: SIMD[Type, simd_width]
    var br = InlineArray[SIMD[Type, simd_width], nr // simd_width](
        SIMD[Type, simd_width](0)
    )
    var cr_ptr = stack_allocation[mr * nr, Scalar[Type], alignment=64]()

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.shape[0]():

                @parameter
                fn load_col[width: Int](j: Int):
                    (cr_ptr + (i * nr + j)).store[Type, width](
                        (Cr_ptr + (i * Cr.stride[0]() + j)).load[Type, width](),
                    )

                vectorize[load_col, simd_width](Cr.shape[1]())
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                (cr_ptr + i * nr + j).store[Type, simd_width](
                    (Cr_ptr + (i * Cr.stride[0]() + j)).load[Type, simd_width](),
                )

    for _ in range(Ar.shape[1]()):

        @parameter
        for j in range(0, nr, simd_width):
            br[j // simd_width] = (Br_ptr + j).load[
                Type, simd_width, alignment = sizeof[Type]() * simdwidthof[Type]()
            ]()

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                ar = SIMD[Type, size=simd_width](Ar_ptr[])
                cr_ptr.store[Type, simd_width](
                    ar.fma(
                        br[j // simd_width],
                        cr_ptr.load[Type, simd_width](),
                    ),
                )
                cr_ptr += simd_width
            Ar_ptr += 1

        Br_ptr += nr
        cr_ptr += -mr * nr

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.shape[0]():

                @parameter
                fn store_row[width: Int](j: Int):
                    (Cr_ptr + (i * Cr.stride[0]() + j)).store[Type, width](
                        (cr_ptr + (i * nr + j)).load[Type, width](),
                    )

                vectorize[store_row, simd_width](Cr.shape[1]())
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                (Cr_ptr + (i * Cr.stride[0]() + j)).store[Type, simd_width](
                    (cr_ptr + (i * nr + j)).load[Type, simd_width](),
                )


fn matmul_params[Type: DType]() -> StaticIntTuple[5]:
    alias mc = 8192 // sizeof[Type]()  # fix this for simplicity
    alias N = simdwidthof[Type]()
    alias L1_ASSOCIATIVITY = 12
    alias L1_CACHE_SIZE = 48 * 1024
    alias L2_ASSOCIATIVITY = 16
    alias L2_CACHE_SIZE = 2 * 1024 * 1024

    alias Vectors = 32 if has_avx512f() else 16

    @parameter
    fn compute_kc[mr: Int, nr: Int]() -> Int:
        alias CBr = int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        return (CBr * L1_CACHE_SIZE) // (nr * sizeof[Type]() * L1_ASSOCIATIVITY)

    @parameter
    fn compute_params[C: Int]() -> StaticIntTuple[5]:
        alias p = C // (intsqrt[C]() + 1)
        alias mr = C // p - 1
        alias nr = p * N
        alias CBr = int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        alias kc = compute_kc[mr, nr]()
        alias nc = (L2_ASSOCIATIVITY - 1) * L2_CACHE_SIZE // (
            kc * sizeof[Type]() * L2_ASSOCIATIVITY
        ) - mr
        return StaticIntTuple[5](mc, nc, kc, mr, nr)

    @parameter
    if Type.is_floating_point():
        alias TempVectors = 1
        return compute_params[Vectors - TempVectors]()
    else:

        @parameter
        if Type is DType.int64:

            @parameter
            if has_avx512f():
                alias TempVectors = 2
                return compute_params[Vectors - TempVectors]()
            else:
                alias TempVectors = 3
                return compute_params[Vectors - TempVectors]()
        else:
            alias TempVectors = 2
            return compute_params[Vectors - TempVectors]()


fn matmul[
    Type: DType, //,
    m: Int,
    n: Int,
    k: Int,
](inout C: Matrix[Type], A: Matrix[Type], B: Matrix[Type]):
    alias params = matmul_params[Type]()
    alias mc = params[0]
    alias nc = params[1]
    alias kc = params[2]
    alias mr = params[3]
    alias nr = params[4]
    alias resized_mc = roundup(min(mc, m), mr)
    alias resized_nc = roundup(min(nc, n), nr)
    matmul_impl[resized_mc, resized_nc, kc, mr, nr](C, A, B)
