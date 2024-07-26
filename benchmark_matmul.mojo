import benchmark
from python import Python
from matmul import matmul, Matrix
from max.graph import Graph, TensorType
from max.engine import Model, InferenceSession
from max import graph
from max.tensor import Tensor, TensorShape
from testing import assert_equal


alias MIN_SIZE = 200
alias MAX_SIZE = 4000
alias NUMPTS = 50
alias Type = DType.float32
alias NUM_ITER = 30
alias WARMUP = 20
import time


@always_inline
fn matmul_mkn[
    Type: DType, //,
    M: Int,
    N: Int,
    K: Int,
](inout c: Matrix[Type], a: Matrix[Type], b: Matrix[Type]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                c[m, n] += a[m, k] * b[k, n]


fn gflops(size: Int, time: Float64) -> Int:
    var flops = 2 * size * size * size
    return int(flops / (time * 1e9))


def to_python_list(arr: List[Int]) -> PythonObject:
    py_list = PythonObject([])
    for i in arr:
        py_list.append(i[])
    return py_list


def benchmark_matmul() -> Tuple[List[Int], List[Int]]:
    print("Benchmarking matmul.mojo")
    var mean_flops = List[Int](capacity=NUMPTS)
    var max_flops = List[Int](capacity=NUMPTS)

    for _ in range(WARMUP):
        var A = Matrix[Type]((MAX_SIZE, MAX_SIZE))
        var B = Matrix[Type]((MAX_SIZE, MAX_SIZE))
        var C = Matrix[Type]((MAX_SIZE, MAX_SIZE))
        matmul.matmul[MAX_SIZE, MAX_SIZE, MAX_SIZE](C, A, B)
        A.data.free()
        B.data.free()
        C.data.free()

    @parameter
    for size in range(
        MIN_SIZE, MAX_SIZE, (MAX_SIZE - MIN_SIZE) // (NUMPTS - 1)
    ):
        var avg_time = Float64(0.0)
        var min_time = Float64.MAX


        # test correctness
        #var A = Matrix[Type]((size, size))
        #var B = Matrix[Type]((size, size))
        #var C = Matrix[Type]((size, size))
        #var C_correct = Matrix[Type]((size, size))

        # A.rand()
        # B.rand()
        # memset_zero(C.data, size*size)
        # memset_zero(C_correct.data, size*size)

        # matmul.matmul[size, size, size](C, A, B)
        # matmul_mkn[size, size, size](C_correct, A, B)
        # for i in range(size*size):
        #     try:
        #         assert_equal(C.data[i], C_correct.data[i])
        #     except:
        #         print("❌ Matmul Incorrect for size:", size)
        #         print("C:", C.data[i], "C_correct:", C_correct.data[i])
        #         raise Error("Matmul Incorrect")

        # A.data.free()
        # B.data.free()
        # C.data.free()
        # C_correct.data.free()

        for _ in range(NUM_ITER):
            var A = Matrix[Type]((size, size))
            var B = Matrix[Type]((size, size))

            A.rand()
            B.rand()

            var start = time.perf_counter()
            var C = Matrix[Type]((size, size))
            matmul.matmul[size, size, size](C, A, B)
            var end = time.perf_counter()

            var exec_time = end - start
            min_time = min(min_time, exec_time)
            avg_time += exec_time

            benchmark.keep(C.data)
            benchmark.keep(A.data)
            benchmark.keep(B.data)

            C.data.free()
            A.data.free()
            B.data.free()

        avg_time /= NUM_ITER

        print(
            "Size:",
            size,
            "Max:",
            gflops(size, min_time),
            "Mean:",
            gflops(size, avg_time),
        )

        mean_flops.append(gflops(size, avg_time))
        max_flops.append(gflops(size, min_time))

    return mean_flops, max_flops


def benchmark_max() -> Tuple[List[Int], List[Int]]:
    print("Benchmarking Max Engine")
    var mean_flops = List[Int](capacity=NUMPTS)
    var max_flops = List[Int](capacity=NUMPTS)

    def build_model(size: Int) -> Model:
        var graph = Graph(
            in_types=List[graph.Type](
                TensorType(Type, size, size), TensorType(Type, size, size)
            )
        )
        var out = graph[0] @ graph[1]
        graph.output(out)
        var session = InferenceSession()
        var model = session.load(graph)
        return model

    var warm_up_model = build_model(MAX_SIZE)
    for _ in range(WARMUP):
        var A = Tensor[Type].rand(TensorShape(MAX_SIZE, MAX_SIZE))
        var B = Tensor[Type].rand(TensorShape(MAX_SIZE, MAX_SIZE))
        var ret = warm_up_model.execute("input0", A^, "input1", B^)

    for size in range(
        MIN_SIZE, MAX_SIZE, (MAX_SIZE - MIN_SIZE) // (NUMPTS - 1)
    ):
        var model = build_model(size)

        var avg_time = Float64(0.0)
        var min_time = Float64.MAX

        for _ in range(NUM_ITER):
            var A = Tensor[Type].rand(TensorShape(size, size))
            var B = Tensor[Type].rand(TensorShape(size, size))
            var start = time.perf_counter()
            var ret = model.execute("input0", A, "input1", B)
            var end = time.perf_counter()
            var exec_time = end - start
            min_time = min(min_time, exec_time)
            avg_time += exec_time

        avg_time /= NUM_ITER

        print(
            "Size:",
            size,
            "Max:",
            gflops(size, min_time),
            "Mean:",
            gflops(size, avg_time),
        )

        mean_flops.append(gflops(size, avg_time))
        max_flops.append(gflops(size, min_time))

    return mean_flops, max_flops


def benchmark_numpy() -> Tuple[List[Int], List[Int]]:
    print("Benchmarking numpy")
    var np = Python.import_module("numpy")
    var mean_flops = List[Int](capacity=NUMPTS)
    var max_flops = List[Int](capacity=NUMPTS)

    for _ in range(WARMUP):
        var A = np.random.randn(MAX_SIZE, MAX_SIZE).astype(np.float32)
        var B = np.random.randn(MAX_SIZE, MAX_SIZE).astype(np.float32)
        var C = np.matmul(A, B)

    @parameter
    for size in range(
        MIN_SIZE, MAX_SIZE, (MAX_SIZE - MIN_SIZE) // (NUMPTS - 1)
    ):
        var avg_time = Float64(0.0)
        var min_time = Float64.MAX

        for _ in range(NUM_ITER):
            var A = np.random.randn(size, size).astype(np.float32)
            var B = np.random.randn(size, size).astype(np.float32)
            var start = time.perf_counter()
            var C = np.matmul(A, B)
            var end = time.perf_counter()
            var exec_time = end - start
            min_time = min(min_time, exec_time)
            avg_time += exec_time

        avg_time /= NUM_ITER

        print(
            "Size:",
            size,
            "Peak:",
            gflops(size, min_time),
            "Mean:",
            gflops(size, avg_time),
        )

        mean_flops.append(gflops(size, avg_time))
        max_flops.append(gflops(size, min_time))

    return mean_flops, max_flops


def main():
    sizes = PythonObject([])
    for size in range(
        MIN_SIZE, MAX_SIZE, (MAX_SIZE - MIN_SIZE) // (NUMPTS - 1)
    ):
        sizes.append(size)

    plt = Python.import_module("matplotlib.pyplot")
    fig_ax = plt.subplots(figsize=(10, 8))

    matmul_flops = benchmark_matmul()
    plt.plot(
        sizes, to_python_list(matmul_flops[0]), "-*", label="matmul.mojo Mean"
    )
    plt.plot(
        sizes, to_python_list(matmul_flops[1]),  "-*", label="matmul.mojo Peak"
    )

    numpy_flops = benchmark_numpy()
    plt.plot(
        sizes,
        to_python_list(numpy_flops[0]),
         "-*", 
        label="numpy(OpenBLAS) Mean",
    )
    plt.plot(
        sizes,
        to_python_list(numpy_flops[1]),
         "-*", 
        label="numpy(OpenBLAS) Peak",
    )

    max_flops = benchmark_max()
    plt.plot(sizes, to_python_list(max_flops[0]),  "-*", label="MAX Engine Mean")
    plt.plot(sizes, to_python_list(max_flops[1]),  "-*", label="MAX Engine Peak")

    fig_ax[1].set_xlabel("Matrix Size")
    fig_ax[1].set_ylabel("GFLOP/s")
    fig_ax[1].legend(fontsize=12, loc="lower right")
    fig_ax[1].grid()
    fig_ax[1].set_title("Matrix Multiplication Benchmark")
    plt.show()
    fig_ax[0].savefig("benchmark_results.svg")
