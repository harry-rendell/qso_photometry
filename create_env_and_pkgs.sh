source ~/.bashrc_old
yes | conda create -n astro python=3.9.16
conda activate astro
yes | conda update -n base -c defaults conda
yes | conda install numpy pandas matplotlib scipy seaborn jupyter
yes | conda install -c conda-forge jupyterlab
yes | conda install -c conda-forge jupytext
yes | conda install -c conda-forge easydict
yes | conda install -c conda-forge astropy
yes | conda install -c conda-forge celerite
yes | conda update -n base -c defaults conda
