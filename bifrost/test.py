from bifrost import ndarray
from btcc import Btcc
import time
import numpy as np

bits_per_sample = 16
ntime_per_gulp = 32
nchan = 128
nstand = 256
npol = 2

tcc = Btcc()
tcc.init(bits_per_sample,
              ntime_per_gulp,
              nchan,
              nstand,
              npol)

dshape = (ntime_per_gulp, nchan, nstand*npol)
d = np.ones(shape=dshape, dtype='float16')
d = ndarray(d, dtype='cf16')

input_data = ndarray(d,
                          dtype='cf16',
                          space='cuda')
print(input_data)

output_data = ndarray(shape=(nchan, nstand*(nstand+1)//2*npol*npol),
                           dtype='cf32',
                           space='cuda')
dump = True

tcc_start = time.perf_counter()
tcc.execute(input_data, output_data, dump)
tcc_end = time.perf_counter()
tcc_time = tcc_end - tcc_start

output_data = output_data.copy(space='system')
print(output_data)

print("Time: ", tcc_time * 1000,  "ms")
