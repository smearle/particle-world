python -m pip install -r requirements.txt

# Probably need to delete respective directories inside submodules for this to properly clone the submodules.
# python -m pip install --upgrade minerl  # As per instructions in minerl docs.

# for torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for 3090
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch