# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append('../')
from module.analysis.plot_filters import plot_filters
import matplotlib
font = {'size'   : 17}
matplotlib.rc('font', **font)

# + active=""
# ### PanSTARRS info ###
# Title: The Pan-STARRS1 Photometric System 
# Authors: Tonry J.L., Stubbs C.W., Lykke K.R., Doherty P., Shivvers I.S., 
#          Burgett W.S., Chambers K.C., Hodapp K.W., Kaiser N., Kudritzki R.-P., 
#          Magnier E.A., Morgan J.S., Price P.A., Wainscoat R.J. 
# Table: Pan-STARRS1 Bandpasses
# ================================================================================
# Byte-by-byte Description of file: apj425122t3_mrt.txt
# --------------------------------------------------------------------------------
#    Bytes Format Units Label  Explanations
# --------------------------------------------------------------------------------
#    1-  4 I4     nm    Wave   Wavelength 
#    6- 10 F5.3   m2    Open   The open bandpass capture cross-section (1)
#   12- 16 F5.3   m2    gp1    The gp1 bandpass capture cross-section (1)
#   18- 22 F5.3   m2    rp1    The rp1 bandpass capture cross-section (1)
#   24- 28 F5.3   m2    ip1    The ip1 bandpass capture cross-section (1)
#   30- 34 F5.3   m2    zp1    The zp1 bandpass capture cross-section (1)
#   36- 40 F5.3   m2    yp1    The yp1 bandpass capture cross-section (1)
#   42- 46 F5.3   m2    wp1    The wp1 bandpass capture cross-section (1)
#   48- 52 F5.3   ---   Aero   Aerosol scattering transmission (2)
#   54- 58 F5.3   ---   Ray    Rayleigh scattering transmission (2)
#   60- 64 F5.3   ---   Mol    Molecular absorption transmission (2)
# -

fig, ax, sdss, ps, ztf = plot_filters(xlim=[3.5e3,9.5e3])

fig.savefig('/disk1/hrb/python/analysis/plots/filters.pdf', bbox_inches='tight')


