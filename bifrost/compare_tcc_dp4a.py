import numpy as np
import bifrost as bf
from bifrost.libbifrost import _bf
from btcc import Btcc

def compute_xcorr_cpu(d):
    dc = d.astype('float32').view('complex64')
    dc = dc.transpose((0,1,3,4,2)).copy()
    xcorr_cpu = np.einsum('...i,...j', dc,  np.conj(dc)).view('float32').sum(axis=-4)
    return xcorr_cpu

def test_xcorr(H, N, F, T):
    # Create complex data
    reset = 1

    # Create complex data
    xcorr = np.zeros((H, F, N, N*2), dtype='float32')
    _N = N // 2
    xcorr_tcc = np.zeros((F, _N*(_N+1)//2*4*2), dtype='float32')

    xcorr_bf = bf.ndarray(xcorr, dtype='f32', space='cuda').view('cf32')
    xcorr_tcc_bf = bf.ndarray(xcorr_tcc.view('complex64'), dtype='cf32', space='cuda')

    tcc = Btcc()
    nbits_tcc = 16
    npol_tcc = 2 ## TODO: Only seems to work on Npol=2
    tcc.init(nbits_tcc, H*T, F, N // npol_tcc, npol_tcc)

    for ii in range(8):
        print("---- Iteration %i ----" % ii)
        print("Generate test vectors...")
        # dp4a takes (Heap, Freq, Nant, Time, cplx)
        d = np.random.randint(32, size=(H, F, N, T, 2), dtype='int8')

        # TCC takes (H*T, F, N, 2)
        d_tcc = d.transpose(0, 3, 1, 2, 4).reshape((H*T, F, N, 2)).astype('float16')

        print("Copying data to GPU...")
        d_gpu       = bf.ndarray(d, dtype='i8', space='cuda')
        d_tcc_gpu   = bf.ndarray(d_tcc, dtype='f16', space='cuda').view('cf16').squeeze()

        #print(d)
        print("Data shape: ", d_gpu.shape)
        print("Data shape, TCC: ", d_tcc_gpu.shape)
        print("Xcorr GPU shape:", xcorr_bf.shape)
        print("Xcorr TCC GPU shape:", xcorr_tcc_bf.shape)

        print("Computing Xcorr on CPU...")
        xcorr_cpu = compute_xcorr_cpu(d)



        print("Run xcorr_lite...")
        _bf.XcorrLite(d_gpu.as_BFarray(), xcorr_bf.as_BFarray(), np.int32(reset))
        dump = True
        tcc.execute(d_tcc_gpu, xcorr_tcc_bf, dump)

        print("Copy result from GPU...")
        xcorr_gpu = np.array(xcorr_bf.copy('system'))
        xcorr_tcc_gpu = np.array(xcorr_tcc_bf.copy('system'))
        
        print("Comparing CPU to GPU...")
        #assert np.allclose(xcorr_gpu.squeeze(), xcorr_cpu.squeeze())
        print(xcorr_gpu[:, :, 0, 0].sum(axis=0))
        print(xcorr_tcc_gpu[:, 0])

def test_multi_accumulate(H, N, F, T, n_cycles):
    reset = 1
    # Create complex data
    xcorr = np.zeros((H, F, N, N*2), dtype='float32')
    xcorr_bf = bf.ndarray(xcorr, dtype='f32', space='cuda')

    d = np.random.randint(64, size=(H, F, N, T, 2), dtype='int8')

    print("Computing Xcorr on CPU...")
    xcorr_cpu = compute_xcorr_cpu(d)

    print("Running xcorr_lite...")
    d_gpu     = bf.ndarray(d, dtype='i8', space='cuda')
    _bf.XcorrLite(d_gpu.as_BFarray(), xcorr_bf.as_BFarray(), np.int32(reset))
    xcorr_gpu = np.array(xcorr_bf.copy('system'))

    print("Testing first integration cycle...")    
    assert np.allclose(xcorr_gpu.squeeze(), xcorr_cpu.squeeze())
    
    print("Running loop ...")
    for ii in range(1, n_cycles):
        print("Run xcorr_lite...")
        reset = 0
        _bf.XcorrLite(d_gpu.as_BFarray(), xcorr_bf.as_BFarray(), np.int32(reset))

        print("Copy result from GPU...")
        xcorr_gpu = np.array(xcorr_bf.copy('system'))
        xcorr_cpu += compute_xcorr_cpu(d)

        print("Testing integration cycle %i /%i..." % (ii+1, n_cycles))    
        assert np.allclose(xcorr_gpu.squeeze(), xcorr_cpu.squeeze())

if __name__ == "__main__":
    test_xcorr(H=2, N=12, F=2, T=1024)
    #test_xcorr(H=7, N=32, F=1, T=1024)
    #test_xcorr(H=7, N=12, F=1, T=1024)
    #test_xcorr(H=7, N=12, F=17, T=1024)
    
    #test_multi_accumulate(H=3, N=12, F=1, T=1024, n_cycles=4)

