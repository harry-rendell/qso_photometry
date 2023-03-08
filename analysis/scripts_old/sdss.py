from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import matplotlib.pyplot as plt

path = '/disk1/hrb/Python/DR14Q.fits'

plt.style.use(astropy_mpl_style)


fits.info(path)
image_data = fits.open(path)
data = image_data[1].data

u = data['PSFFLUX']
t = data['MJD']
name = data['SDSS_NAME']