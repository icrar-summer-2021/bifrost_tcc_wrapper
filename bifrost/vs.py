#  Author: Liam Ryan (ICRAR/Pawsey Summer Intern)
# Contact: liamdryan01@gmail.com
# Created: 05/01/2022
# Purpose: Benchmark the correlation performance of xcorr_lite (xcorr)
#     and Bifrost's wrapper for Romein's (2021) Tensor-Core 
#     Correlator (btcc/tcc)
#
# Program Summary:
#   1. Generates input data in xcorr_lite form:
#       - shape=(heap, nchan, nant*npol, ntime, 2)
#       - dtype='i8' (Bifrost)
#      Multiple options for test data
#       - Random int8 0->64
#       - Repeating int8 0->64 (0 1 2 ... 63 64 0 1 2 ... 63 64 ...)
#       - Manual entry (for debugging)
#   2. Copies and transforms input data across to BTCC form: 
#       - shape=(ntime, nchan, nant*npol)
#       - dtype='cf16' (Bifrost)
#   3. Execute both correlation operations, timed using time.perf_counter()
#       - tcc.execute(tcc_input, tcc_output, True)
#       - _bf.XcorrLite(xcorr_input.as_BFarray(), xcorr_output.as_BFarray(), np.int32(reset)))
#   4. Both correlator outputs transformed into a common form - 
#      FULL correlation matrix (for comparison)
#       - shape=(nchan, nant, nant, npol, npol)
#       - dtype='complex64' (NumPy)
#
# NOTE: Whilst performance was heavily considered in creation of the code this
#   is in no way a working correlator, several performance limiting operations
#   are included for comparison and benchmarking purposes, many optimisations
#   are likely to be found for an actual working BTCC-based correlator

import sys
import time
import numpy as np
import bifrost as bf
from btcc import Btcc
from bifrost.libbifrost import _bf

nbits = 16 # number of bits MUST be 16 for current ICRAR hardware (Tesla V100)

def test_correlate(nant, nchan, ntime, npol):
    """
    Generates input data and times the correlation of 
    Bifrost's wrapped versions of TCC and xcorr_lite.
    Returns: tcc_time_ms, xcorr_time_ms
    """

    print("\nParameters")
    print("\tnant  =", nant)
    print("\tnpol  =", npol)
    print("\tnchan =", nchan)
    print("\tntime =", ntime, "\n")

    # initialise the tensor core correlator
    tcc = Btcc()
    tcc.init(nbits, ntime, nchan, nant, npol)

    # ---------------------------------------------------------------------
    # ---------------- Create test vectors in "xcorr form" ----------------
    # ---------------------------------------------------------------------

    # NOTE: xcorr_input.shape = (heap, nchan, nant*npol, ntime, 2)
    
    # Random int8 0->64
    # gen_xcorr_re = (np.random.rand(1, nchan, nant*npol, ntime)*64).astype('int8')

    # Repeating int8 0->64 0 1 2 3 ... 62 63 64 0 1 2 3 ... 62 63 64 ...
    # Just trust me
    # gen_xcorr_re = np.mod(np.arange(nchan*nant*npol*ntime).reshape(1,nchan,nant*npol, ntime), np.full((1, nchan, nant*npol, ntime), 64)).astype('int8')
    
    # Manual Entry
    gen_xcorr_re = np.zeros(shape=(1, nchan, nant*npol, ntime)).astype('int8')
    gen_xcorr_re[0,0,0,0] = 1
    gen_xcorr_re[0,0,1,0] = 2
    gen_xcorr_re[0,0,2,0] = 3
    gen_xcorr_re[0,0,3,0] = 5

    # Create imaginary components, all 0
    gen_xcorr_im = np.zeros_like(gen_xcorr_re)
    gen_xcorr_im[0,0,0,0] = 0
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    # create input and output arrays for xcorr_lite
    xcorr_input = bf.ndarray(   shape=(1, nchan, nant*npol, ntime, 2), 
                                dtype='i8', 
                                space='system')
    xcorr_input[:,:,:,:,0] = gen_xcorr_re # set real components of xcorr_input
    xcorr_input[:,:,:,:,1] = gen_xcorr_im # set imag components of xcorr_input
    xcorr_input = xcorr_input.copy(space='cuda')
    # NOTE: there seems to be an issue with Bifrost, can't set real
    #   and imag components in 'cuda' space, must set in 'system'
    #   space first and then copy over to 'cuda'?
    xcorr_output = bf.ndarray(shape=(1, nchan, nant*npol, nant*npol*2), # the *2 is to for the storage of complex data
                            dtype='f32',
                            space='cuda')

    # reformat generated input data for tcc 
    # required shape: (ntime, nchan, nant*npol)
    gen_tcc_re = np.moveaxis(np.squeeze(np.copy(gen_xcorr_re)), [0,1,2], [1,2,0]).astype('float16')
    gen_tcc_im = np.moveaxis(np.squeeze(np.copy(gen_xcorr_im)), [0,1,2], [1,2,0]).astype('float16')
    tcc_input = bf.ndarray( shape=gen_tcc_re.shape,
                            dtype='cf16',
                            space='system')
    tcc_input['re'] = gen_tcc_re
    tcc_input['im'] = gen_tcc_im
    tcc_input = tcc_input.copy(space='cuda')

    tcc_output = bf.ndarray(shape=(nchan, nant*(nant+1)//2*npol*npol),
                            dtype='cf32',
                            space='cuda')
    # nant*(nant+1)//2*npol*npol comes from cross-corrs AND auto-corrs
    
    # execute TCC
    tcc_start = time.perf_counter()
    tcc.execute(tcc_input, tcc_output, True)
    tcc_end = time.perf_counter()
    tcc_time_ms = (tcc_end - tcc_start) * 1000
    print("btcc time (ms):  " + str(tcc_time_ms) + "\n")

    # copy output data from GPU to CPU
    tcc_output_cpu = tcc_output.copy(space='system')
    
    # reshape into (nchan, nant(nant+1)/2, npol, npol)
    tcc_shaped = tcc_output_cpu.reshape(nchan, -1, npol, npol)

    # reshape into correlation matrix upper triangle (nchan, nant, nant, npol, npol)
    tcc_comp = np.zeros((nchan, nant, nant, npol, npol)).astype('complex64')
    a, b = np.triu_indices(nant)
    tcc_comp[:, a, b] = tcc_shaped
    
    # reflect upper triangle onto lower triangle with conjugate for full correlation matrix
    a, b = np.tril_indices(nant, -1)
    tcc_comp[:, a, b] = np.conjugate(tcc_comp.swapaxes(1,2).swapaxes(3,4)[:, a, b])

    # execute xcorr_lite correlation
    reset = 1 # TODO: found out what this does...
    xcorr_start = time.perf_counter()
    _bf.XcorrLite(xcorr_input.as_BFarray(), xcorr_output.as_BFarray(), np.int32(reset))
    xcorr_end = time.perf_counter()
    xcorr_time_ms = (xcorr_end - xcorr_start) * 1000
    print("xcorr time:  " + str(xcorr_time_ms))

    # copy xcorr output to CPU and reformat into (nchan, nant, nant, npol, npol)
    xcorr_output_cpu = np.array(xcorr_output.copy('system'))
    xcorr_comp = np.squeeze(xcorr_output_cpu).view('complex64')
    xcorr_comp = xcorr_comp.reshape(nchan, nant, npol, -1, npol).swapaxes(2,3).reshape(nchan, nant, nant, npol, npol)


    # -----------------------------------------------
    # ------------------ DEBUGGING ------------------
    # -----------------------------------------------
    print("BTCC Input Shape")
    print("\t" + str(tcc_input.shape))
    print("\t(ntime, nchan, nant*npol)")
    print("BTCC Output Shape")
    print("\t" + str(tcc_output.shape))
    print("\t(nchan, nant*(nant+1)//2*npol*npol)")

    print("XCORR Input Shape")
    print("\t" + str(xcorr_input.shape))
    print("\t(H, nchan, nant*npol, ntime, 2)")
    print("XCORR Output Shape")
    print("\t" + str(xcorr_output.shape))
    print("\t(H, nchan, nant*npol, nant*npol*2)")

    print("tcc_input")
    print(tcc_input)
    print("xcorr_input")
    print(xcorr_input)

    print("tcc_output")
    print(tcc_output_cpu)
    print("xcorr_output")
    print(xcorr_output_cpu.view('complex64'))

    print("tcc_comp")
    print(tcc_comp)
    print("xcorr_comp")
    print(xcorr_comp)

    # print("tcc_comp[0]")
    # print(tcc_comp[0])
    # print("xcorr_comp[0]")
    # print(xcorr_comp[0])
    # -----------------------------------------------
    # -----------------------------------------------
    # -----------------------------------------------

    print("Results: {}/{} correct".format(np.isclose(tcc_comp, xcorr_comp, rtol=1e-2).sum(), tcc_comp.size))

    # delete the tcc object, this was required at some point w/ multiple iterations?
    del tcc
    
    return tcc_time_ms, xcorr_time_ms


if __name__ == '__main__':
    # default parameters
    nchan = 4
    ntime = 128
    npol = 2
    nant = 8

    if (len(sys.argv) != 4):
        test_correlate(nant, nchan, ntime, npol)
    else:
        min_nant = int(sys.argv[1])
        step_nant = int(sys.argv[2])
        max_nant = int(sys.argv[3])
    
        # write results as csv to "timing_results/" subdirectory
        f = open("timing_results/results_N" + str(min_nant) + "-" + str(step_nant) + "-" 
                + str(max_nant) + "_T" + str(ntime) + "_F" + str(nchan) + ".csv", "w")
        
        for nant in range(min_nant, max_nant + 1, step_nant):
            print("Testing...")

            tcc_time_ms, xcorr_time_ms = test_correlate(nant, nchan, ntime, npol)

            f.write(str(nant) + "," + str(tcc_time_ms) + "," + str(xcorr_time_ms) + "\n")
        f.close()
