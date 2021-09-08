set -euo pipefail

conda install mamba -c conda-forge

mamba create -n srunet python=3.9 pip
conda activate srunet
mamba install pytorch torchvision cudatoolkit=10.2 -c pytorch numpy matplotlib
git clone git@github.com:ivan94fi/fast-sr-unet.git
pip install git+https://github.com/ivan94fi/PerceptualSimilarity.git comet_ml gpustat torchinfo tqdm
