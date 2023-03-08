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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class main():
    def __init__(self, index, column):
        self.index = sub(index)
        self.column = column
        self.x = np.linspace(0,6.28)
        self.y = np.sin(self.x)
        
    def plot(self):
        self.p = plotting()        
        self.p.scatter(self)


class plotting():
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1, figsize=(3,3))
        
    def scatter(self, other):
        self.ax.scatter(other.x, other.y)


class sub():
    def __init__(self, values):
        self.values = values
    
    def __repr__(self):    
        return str(self.values.shape)
    
    def __len__(self):
        return len(self.values)
    
    def __add__(self, new_entries):
        return np.append(self.values, new_entries)
    
    def mod(self,x):
        return np.mod(self.values, x)
        


# Example of using classes as attributes of another class. Here, main() has an attribute self.index which is the class sub(). This allows us to do things like m.index.values, analagous to pd.DataFrame.Index

m = main(np.arange(10), np.arange(10,20))
m.index.values

# ## dunder methods

# > we can use __ repr __ to print the shape of the index just by calling m.index (without this, calling m.index would return < function name >. We could make this more complex by printing a summary of the data in m.index.

m.index

# > we can use __ len __ to make len(m.index) return something useful too.

len(m.index)

# ## plotting

# Below are two methods of plotting. Either we create a plotting class and pass values to that, or we call the plotting class from within main() and call that. Which is better?

p = plotting()
p.scatter(m)

m.plot()
