source pacer_env.sh
module load bifrost/0.0.3
export PKG_CONFIG_PATH=/group/director2183/dancpr/software/centos7.6/apps/cascadelake/gcc/8.3.0/bifrost/0.0.2/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=/group/director2183/dancpr/src/:$PATH
export PATH=/group/director2183/dancpr/.local/bin/:$PATH
export PYTHONPATH=/group/director2183/dancpr/.local/lib/python3.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/group/director2183/dancpr/src/bifrost_tcc_wrapper/bifrost/libtcc:/group/director2183/dancpr/src/bifrost_tcc_wrapper/bifrost/build:$LD_LIBRARY_PATH
