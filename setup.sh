python -m pip install -r requirements.txt

# Might need to delete respective directories inside submodules for this to properly clone the submodules.
git submodule update --init qdpy
python -m pip install -e submodules/qdpy
# python -m pip install --upgrade minerl  # As per instructions in minerl docs.

# Install torch

# For CPU-only:
conda install pytorch torchvision torchaudio -c pytorch

# For most GPUs:
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# For 3090:
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch