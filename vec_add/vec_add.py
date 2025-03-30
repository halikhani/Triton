import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
# print(DEVICE)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load data from DRAM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    output = x + y

    # write data back to DRAM
    tl.store(output_ptr + offsets, output) 


def add(x, y):
    # preallocate output
    output = torch.empty_like(x)
    #check tensors are on the same device
    assert x.device == DEVICE and y.device == DEVICE

    # defining our launch grid
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), ) #(4,)
    # cdiv(m, n) =  (m + n - 1)//n
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    return output


def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    # create test data
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    # run triton kernel & python equivalent
    z_tri = add(x, y)
    z_ref = x + y 

    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals= [2**i for i in range(12, 20, 1 )],
        x_log=True,
        line_arg='provider',
        line_vals= ['triton', 'torch'],
        line_names= ['Triton', 'Torch'],
        styles= [('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='Vector-add_perf',
        args={}
    )
)

def benchmark(size, provider):
    # create input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True )

