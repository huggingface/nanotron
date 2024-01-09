export CUDA_HOME="/usr/local/cuda-12.1"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.1/bin:$PATH"

git clone git@github.com:NVIDIA/apex.git
cd apex

# this is the commit where fast layer norm and fused layer norm aren't broken
git checkout bc4be41c6fdb889db84b9f61f35440f82a057948

# install fused layer norm
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# install fast layer norm
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--global-option=--fast_layer_norm" ./
