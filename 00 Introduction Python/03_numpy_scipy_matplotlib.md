---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Importing packages and introduction to Numpy/Scipy/Matplotlib
Python capabilities are extended with many useful packages. A few of the packages that we'll use in this class are:

- [NumPy](https://www.numpy.org/) for array-computing
- [SciPy](https://www.scipy.org/) for numerical routines
- [Matplotlib](https://www.matplotlib.org/) for data visualization

You can import a package with ```import package as pk``` where ```package``` is the package name and ```pk``` is the abbreviated name.


***Task:*** Import the ```numpy``` package as ```np``` (shortened name) and get help on the ```np.linspace``` function.

```{code-cell} ipython3
:tags: [solution]

import numpy as np
help(np.linspace)
```

***Task:*** Use ```np.linspace``` to define 20 equally spaced values between $0$ and $2\,\pi$. Name the variable ```x``` and use the built-in ```np.pi``` constant for $\pi$.

```{code-cell} ipython3
:tags: [solution]

x = np.linspace(0,2*np.pi,20)
print(x)
```

***Task:*** Use ```np.sin``` to calculate a new variable ```y``` as $y=2\,\sin{\left(\frac{x}{2}\right)}$.

```{code-cell} ipython3
:tags: [solution]

y = 2*np.sin(x/2)
print(y)
```

***Task:*** Import the ```matplotlib.pyplot``` package as ```plt``` (shortened name) and show the help on the ```plt.plot``` function.

```{code-cell} ipython3
:tags: [solution]

import matplotlib.pyplot as plt
help(plt.plot)
```

***Task:*** Create a plot of ```x``` and ```y``` with the ```plt.plot``` function. Add an x-label with ```plt.xlabel``` and a y-label with ```plt.ylabel```.

```{code-cell} ipython3
:tags: [solution]

plt.plot(x,y,label='y=2*sin(x/2)')
plt.xlabel('x')
plt.ylabel('y')

z = np.cos(x)
plt.plot(x,z,label='cos(x)')

plt.title('Trigonometry Functions')
plt.legend()
```
