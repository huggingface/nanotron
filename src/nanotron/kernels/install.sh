git clone git@github.com:NVIDIA/apex.git
cd apex
# this is the commit where fast layer norm and fused layer norm aren't broken
# source: https://github.com/NVIDIA/apex/issues/1594#issuecomment-1636078623
git checkout 6943fd26e04c59327de32592cf5af68be8f5c44e
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --global-option="--fast_layer_norm" ./
