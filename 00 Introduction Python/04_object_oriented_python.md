---
jupytext:
  formats: ipynb,md:myst
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

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Object Oriented Programming

+++

## Introduction to Object-Oriented Programming (OOP) for Modeling

Last year, we received a few questions and noticed some confusion around object-oriented programming. So, I want to take this opportunity to introduce some techniques and syntax we’ll use going forward. This will help everyone follow along more smoothly.

### Basics of Variables in Python

In Python, we can store information in variables. For instance, I could define the length of a column as a floating-point number and add a comment to indicate that it's measured in meters. Comments in Python begin with a pound symbol, which tells the interpreter to ignore everything after it. Then, I can output the column length to see its value.

Now, if I want to expand on this model of a column, I might add another variable, like radius, to store the column’s radius in meters. Python automatically recognizes the data type if I assign it a number. I can then calculate the column’s volume using length, radius, and pi.

However, if I want a second column, I have to duplicate all this code—defining a new length, radius, and volume, which becomes tedious. Constantly updating variable names, repeating code, and managing values manually is not efficient.

### Reducing Code Repetition

To reduce code repetition, I could use functions. A function allows us to encapsulate repetitive tasks, making our code shorter and easier to manage. But even with functions, manually handling individual values for each column remains cumbersome, especially when we add more parameters like cross-sectional area or porosity. Imagine managing several columns with unique properties—having a separate variable for each becomes overwhelming.

### Solution: Object-Oriented Programming

This is where object-oriented programming (OOP) comes in. OOP allows us to create a class, or a blueprint, that groups all relevant information about an entity—in this case, a column. We can then create instances of this class, each representing a distinct column with its own attributes. Think of it like DNA and proteins: the class is like the DNA, which defines how to create the object, and the object is the end result that we can work with directly.

To define a class in Python, we start by giving it an initialization function. This function, called "init," specifies the attributes each instance needs, such as length and radius. These attributes are stored within the object itself, using a standard placeholder in Python called "self." Once defined, I can create columns with specific lengths and radii. Each column instance is self-contained, so changing the length of one column doesn’t affect the others.

Another advantage of classes is that we can add functions, or "methods," to perform actions directly within each object. For instance, we could create a method that calculates volume, using the length and radius stored in that instance. This way, each column can calculate its volume independently without needing a separate variable for volume.

In future sessions, we’ll be creating column objects with various properties, such as porosity and adsorption methods, and possibly even event objects for more complex modeling.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
tags: [solution]
---

```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
tags: [solution]
---

```

```{code-cell} ipython3

```
