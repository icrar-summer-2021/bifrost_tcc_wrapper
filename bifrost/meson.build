# Bifrost libtcc wrapper build file
# export PKG_CONFIG_PATH=/group/director2183/dancpr/software/centos7.6/apps/cascadelake/gcc/8.3.0/bifrost/0.0.2/lib/pkgconfig:$PKG_CONFIG_PATH
# export PATH=/group/director2183/dancpr/src/:$PATH
# export PATH=/home/dancpr/.local/bin/:$PATH

project('btcc', 'cpp', 'cuda', version: '0.0.2',
         default_options : ['cpp_std=c++11'])

# CUDA setup
cuda = import('unstable-cuda')
cuda_dep = dependency('cuda', version : '>=11', modules : ['cudart', 'nvrtc'])

# Dependency: bifrost
bf_dep = dependency('bifrost')

# Dependency: TCC
tcc_inc = include_directories(
    '/group/director2183/dancpr/src/tensor-core-correlator/',
    '/group/director2183/dancpr/src/tensor-core-correlator/util',
    '/group/director2183/dancpr/src/tensor-core-correlator/libtcc',
    )

cc = meson.get_compiler('cpp')
tcc_lib = cc.find_library('libtcc',
               dirs : ['/group/director2183/dancpr/src/tensor-core-correlator/libtcc'], required: true)

# CUDA flags
#nvflags = cuda.nvcc_arch_flags('11.1', 'Volta')
#message(cuda.min_driver_version('11.1'))

lib = library('btcc', 'btcc.cu',  version : '0.0.2', soversion : '0', 
             dependencies: [bf_dep, cuda_dep, tcc_lib], include_directories: tcc_inc)


