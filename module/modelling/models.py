import numpy as np
# Note, all these are defined in a way such that the first argument is the
#   lowest order term, and the last argument is the highest order term.

def power_law(x,a,b):
    """
    Basic power law, y = ax^b
    """
    return a*x**b

def power_law_offset(x, a, b, c):
    """
    Power law with linear offset, y = a + bx^c
    """
    return a + b*x**c

def linear(x, a, b):
    """
    Linear function, y = a + bx
    """
    return a + b*x

def bkn_pow_smooth(x, A, x_b, a_1, a_2, delta=1):
    """
    https://gist.github.com/cgobat/12595d4e242576d4d84b1b682476317d
    """
    a_1 *= -1
    a_2 *= -1
    return A*(x/x_b)**(-a_1) * (0.5*(1+(x/x_b)**(1/delta)))**((a_1-a_2)*delta)

def bkn_pow(xvals,breaks,alphas):
    """
    https://gist.github.com/cgobat/12595d4e242576d4d84b1b682476317d
    """
    try:
        if len(breaks) != len(alphas) - 1:
            raise ValueError("Dimensional mismatch. There should be one more alpha than there are breaks.")
    except TypeError:
        raise TypeError("Breaks and alphas should be array-like.")
    if any(breaks < np.min(xvals)) or any(breaks > np.max(xvals)):
        raise ValueError("One or more break points fall outside given x bounds.")
    
    breakpoints = [np.min(xvals)] + breaks + [np.max(xvals)] # create a list of all the bounding x-values
    chunks = [np.array([x for x in xvals if x >= breakpoints[i] and x <= breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
    
    all_y = []
    
    #alpha = pd.cut(pd.Series(xvals),breakpoints,labels=alphas,include_lowest=True).to_numpy()

    for idx,xchunk in enumerate(chunks):
        yvals = xchunk**alphas[idx]
        all_y.append(yvals) # add this piece to the output
    
    for i in range(1,len(all_y)):
        all_y[i] *= np.abs(all_y[i-1][-1]/all_y[i][0]) # scale the beginning of each piece to the end of the last so it is continuous
    
    return(np.array([y for ychunk in all_y for y in ychunk])) # return flattened list
