source ~/.bashrc_old
yes | conda update -n base -c defaults conda
yes | conda create -n astro python=3.9.16
conda activate astro
yes | conda install numpy pandas matplotlib scipy seaborn jupyter
yes | conda install -c conda-forge jupyterlab jupytext easydict astropy celerite jupyterlab-git jupyter_nbextensions_configurator venn scikit-learn autograd
yes | conda install -c astropy emcee
yes | conda update -n base -c defaults conda