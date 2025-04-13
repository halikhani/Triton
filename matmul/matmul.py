"""
- automatic peformance tuning
- PID reordering for improved SRAM sharing between PIDs
- multi-dimensional pointer arithmetic
- data types, high precision accumulation 
- triton interpreter for improved debugging
A @ B = C 
(M, K) @ (K, N) = (M, N)

1.

for m in range(0, M):
    for n in range(0, N):
        a_vec = A[m, :]
        b_vec = B[:, n]
        C[m, n] = dot_prod(a_vec, b_vec)

2. (block-wise matmul)
(each iteration has its own PID)

for m in range(0, M, BLOCK_SIZE_M):
    for n in range(0, N, BLOCK_SIZE_N):
        acc = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a_block = A[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K]
            b_block = A[m: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N]
            acc += matmul(a_block, b_block) (implemented as tl.dot(a_block, b_block))
        
        C[m: m + BLOCK_SIZE_M, n: n + BLOCK_SIZE_N] = acc
         
"""
import torch
import triton
import triton.language as tl 

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

# step 3
import os
# os.environ["TRITON_INTERPRET"] = "1" # a triton env simulator in numpy so you can use print statements in the code, useful for debugging, ...
# see if above part works, if not comment it
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   1) a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   2) an auto-tuning *key* whose change in values will trigger a new evaluation of all the provided configs, meaning
#       that any time either M, N, or K changes with a new input, Triton will check which config is best all over again
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K,
    stride_a_M, stride_a_K,
    stride_b_K, stride_b_N,
    stride_c_M, stride_c_N,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """
    M = N = K = 8
    BLOCK_SIZE_M/N/K = 2
    PIDs:
    [0,   1,  2,  3]
    [4,   5,  6,  7]
    [8,   9, 10, 11]
    [12, 13, 14, 15]
    
    PID 0 for this example:
        A           @       B           =       C
    [x, x, x, x]        [x, _, _, _]        [0, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    
        A           @       B
    [--------->]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [\|/, _, _, _]

    chunks of data (input matrix blocks) used by each PID:

    PID = 0
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 1
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    PID = 2
    [x, x, x, x]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    PID = 3
    [x, x, x, x]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]

    think about how we can group PIDs on an SM to reduce memory load ? (e.g., grouping PIDs: 0, 1, 4, 5) (for more, search row-major vs grouped ordering)
    PID = 4
    [_, _, _, _]        [x, _, _, _]
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 5
    [_, _, _, _]        [_, x, _, _]
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]

    so, how about this ordering:
    [0,  2, |  4,  6]
    [1,  3, |  5,  7]
    --------|--------
    [8, 10, | 12, 14]
    [9, 11, | 13, 15] 
    (The size of these groups is defined by our "GROUP_SIZE" meta-parameter.)

    """

    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    # (PID % num_PID_in_group) puts the current program id into the context of a group
    # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
    # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
    # (... // group_size_adj) removes the row component to get us onto the correct column

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N
    """
    [:, None] turns [m1 m2 m3] into [[m1] [m2] [m3]]
    [None, :] turns [n1 n2 n3] into [[n1 n2 n3]]

    combining them gives the matrix
    [[m1n1, m1n2, m1n3],
     [m2n1, m2n2, m2n3],
     [m3n1, m3n2, m3n3]] 
    """
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < (K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    accumulator = accumulator.to(tl.float16)

    c_offsets = offsets_M[:, None] * stride_c_M + offsets_N[None, :] * stride_c_N 
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


# step 2. 
def matmul(a, b):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]
    a, b = a.to(torch.float16), b.to(torch.float16)

    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_M']), # number of total blocks needed
    ) # (16,)
    # cdiv(x, y) = (x + (y-1))//y
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# step 1. defining the test function 

def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)

    assert type(size) == tuple and len(size) == 2
    
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print('PASSED!')


configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # we can increase multiple dimensions simultaneously while benchmarking
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
        # 3 = number of memory operations (2 read + 1 write)
        # M * N * K = number of elements per memory op
        # 1e-12 converts flops to Teraflops
        # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # always run unit-tests
    test_matmul_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)