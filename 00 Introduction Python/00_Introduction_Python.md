---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to Python

## Why Python?
- Simplicity: Easy to code, easy to read.
- Availability: Free and open source and compatible with all major operating systems.
- "Batteries included": Robust standard library provides wide range of modules to add functionality to the Python application.
- Excellent third party libraries:

***The Python ecosystem:***

```{figure} ./images/python_ecosystem.png
:alt: Python Ecosystem
:width: 500px
:align: center
```

- In this course we will be using:
    - [NumPy](https://www.numpy.org/) for array-computing
    - [SciPy](https://www.scipy.org/) for numerical routines
    - [Matplotlib](https://www.matplotlib.org/) for data visualization

- Widespread Adoption in Science and Education:


```{figure} ./images/stackoverflow-growth.png
:alt: Python Growth
:width: 500px
:align: center
```

### Tutorial objectives
This tutorial covers the following topics:
- Installation of Python
- `help`, `print`, `type`
- Variables
- Operators
- Basic Functions
- Import of packages
- Introduction to numpy
- Introduction to matplotlib

+++

___

+++

## Python Installation
- In this course, we will be using the Conda Python distribution. Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies completely seprate from any system Python installations and withoud needing administrative rights.
- Download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)
- Installation remarks:
    - Accept License Agreement
    - Select *Install Just Me*
    - Select Destination Folder: use recommended
    - Do *not* add Anaconda to your PATH
    - Click install
    - To install additional packages, open the anaconda prompt from the windows menu and enter:
        ```
        conda install numpy scipy matplotlib jupyterlab
        ```
- The tutorial and exercises will be presented using jupyter notebooks. Jupyterlab is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.
    - Markdown cells are used for formatted text
    - Code cells contain executable code

***Task:*** Start Jupyterlab Server by entering ``` jupyter-lab ``` in the anaconda prompt, create a new notebook and a new cell and convert the type to 'Markdown'.

+++

## Further Watching
You can find plenty of great resources for (Python) programming online.
- [Corey Schafer](https://www.youtube.com/user/schafer5) is focused on general programming aspectes, mainly in Python, but also has good introductions to *Bash*, *Git*, *SQL*, and many more.
- [APMonitor](https://www.youtube.com/user/APMonitorCom) has a lot of engineering examples in *Python*, *Matlab* or on *paper*. In fact, some of this course's exercises are based heavily on his work.
- [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) makes great videos series on a broad range of (physical) math topics such as *linear algebra*, *calculus*, and *differential equations*.
