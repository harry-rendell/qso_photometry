import pandas as pd
from astropy.table import Table
from astropy.io import ascii

def convert_to_ipac(dataframe, save_as):
	t = Table.from_pandas(dataframe, index=True)
	ascii.write(t, save_as, format='ipac')