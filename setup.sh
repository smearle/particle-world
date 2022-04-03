python -m pip install -r requirements.txt

# Probably need to delete respective directories inside submodules for this to properly clone the submodules.
git submodule update --init qdpy pytorch-neat
python -m pip install -e submodules/qdpy
python -m pip install --upgrade minerl  # As per instructions in minerl docs.

# for torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for 3090
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch