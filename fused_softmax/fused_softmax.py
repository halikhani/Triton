"""
  - Reducing mem read/write by fusing
  - GPU spec and notes on architecture
  - Defining meta-params with heuristics and gpu-specific attrs
  - masking

"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
# print(torch.cuda.current_device())

# step 1

def naive_softmax(x):
    # assume input size (M, N)
    # reads MN elements
    x_max = x.max(dim=1)[0] # shape (M)

    # read MN + M elements, subtraction is MN flops, write MN elements
    z = x - x_max[:, None] # shape (M, N) - shape (M, 1) = shape (M, N)

    # read MN elements, write MN elements
    numerator = torch.exp(z)

    # read MN elems, MN flops, then write M elems
    denominator = numerator.sum(dim=1) # shape (M, N) -> shape (M)

    # read MN + M elems, div MN flops, write MN elems 
    out = numerator/denominator[:, None] # shape (M, N) / shape (M, 1) = shape (M, N)


    # total: 8 MN + 4M mem operations
    return out


# step 4
@triton.jit
def _softmax_kernel(input_ptr, output_ptr,
                    input_row_stride, output_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    num_stages: tl.constexpr
                     ):
    # shape (M, N)
    # BLOCK_SIZE = next power of two bigger than N
    
    PID = tl.program_id(0)
    row_step = tl.num_programs(0)
    # e.g., if 4 programs, then row_step = 4
    # if n_rows = 6, then:
    # pid 0 would get row 0
    # pid 1 row 1
    # pid 2 row 2
    # pid 3 row 3
    # after pids are done with their first assigned rows: 
    # pid 0 += row_step (it gets row 4)
    # pid 1 += row_step (it gets row 5)

    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) # shape of BLOCK_SIZE (roughly n_cols)

        row_minus_max = row - tl.max(row, axis=0)  # only 1 axis here so row - a number, shape BLOCK_SIZE - (1) -> BLOCK_SIZE
        numinator = tl.exp(row_minus_max) # shape BLOCK_SIZE
        denominator = tl.sum(numinator, axis=0) # SHAPE 1
        softmax_output = numinator / denominator 

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)




# step 2: test softmax kernel

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) == tuple and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=DEVICE)
    
    z_tri = softmax(x) # triton function to be implemented
    z_ref = torch.softmax(x, axis=1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print('PASSED!')

# step 3: properties of GPU
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x):
    assert x.ndim == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(x, y, # this warmup depends on the attributes of the input and output
                                    x.stride(0), y.stride(0), # see below
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))

    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_prgoram = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    # NUM_REGS = 65536
    # n_regs_per_program = 32
    # WARP_SIZE, num_warps = 32, 8
    # so each program needs 32 * 32 * 8 regs => 65536/(32*32*8) = 8 programs per SM

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_prgoram

    programs_per_sm = min(reg_occupancy, sram_occupancy)

    num_programs = min(NUM_SM * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)

    kernel[grid](
        x, y, 
        x.stride(0), y.stride(0),
        n_rows, n_cols
    )

    return y


# step 5
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096} # values for function arguments not in x_names
    )
)

def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms*1e-3)
    return gbps(ms)


if __name__ == "__main__":
    # unit test 
    test_softmax_kernel(size=(1823, 781))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
    
    
