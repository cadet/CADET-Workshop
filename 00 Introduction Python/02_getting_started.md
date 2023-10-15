---
jupytext:
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

## Getting started with Python

### Hello World!
The simplest program in Python consists of a line that tells the computer a command. Traditionally, the first program of every programmer in every new language prints "Hello World!"

+++

***Task:*** Change ```'Hello world!'``` to a different text message.

```{code-cell} ipython3
:tags: [solution]
print('Hello world!')
```

### Help me!
The most important thing to learn in programming is finding help. The easiest way is to to use the ```help()``` function. If you get stuck, search for answers online in the online documentation of [Python](https://docs.python.org/3/) or in forums such as [Stackoverflow](https://stackoverflow.com/questions/tagged/python). Chances are, other programmers will have had the same problem as you have!

+++

***Task:*** Show the help documentation for the ```max``` function instead of the ```print``` function. Based on the help, use the ```max``` function to find the highest value of two numbers: ```5``` and ```2```.

```{code-cell} ipython3
:tags: [solution]
help(print)
```

### Python as a calculator

Python can be used just like a calculator. Enter an expression and the interpreter returns the result. The syntax is simple: the operators ```+```, ```-```, ```*```, and ```/``` act as you would expect. Round brackets ```( )``` can be used to group expressions.

+++

***Task:*** Play around with the interpreter and enter some equations!

```{code-cell} ipython3
:tags: [solution]
1 + 1
```

```{code-cell} ipython3
:tags: [solution]
(50 * 5 - 6) / 3
```

### Operators

We've already seen some operators. Operators are used to transform, compare, join, substract, etc. Below is a list of operators in descending order of precedence. When there are no parenthesis to guide the precedence, this order will be assumed.

| Operator | Description |
| --- | --- |
| ** | Exponent |
| *, /, //, % | Multiplication, Division, Floor division, Modulus |
| +, - | Addition, Subtraction |
| <=, <, >, >= | Comparison |
| =, %=, /=, //=, -=, +=, *=, **= | Assignment |
| is, is not | Identity |
| in, not in | Membership |
| not, or, and | Logical |

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
# Area of unit circle. <-- Note the '#', it makes everything after that a comment, i.e. it will NOT be executed
3.1415926/4 * 2**2
```

***Example:***

```{code-cell} ipython3
:tags: [solution]
1 == 0
```

***Task:*** Calculate the volume of the unit sphere.

```{code-cell} ipython3
:tags: [solution]

```

***Task:*** Determine whether $35^2$ is greater than $2^{10}$.

```{code-cell} ipython3
:tags: [solution]

```

### Variables

Variables are reserved memory locations to store values. By assigning different data types to variables, you can store integers, decimals or characters in these variables. Python variables do not need explicit declaration to reserve memory space. The declaration happens automatically when you assign a value to a variable using the equal sign (```=```).

The operand to the left of the ```=``` operator is the name of the variable and the operand to the right of the ```=``` operator is the value stored in the variable. A variable name must begin with a letter or underscore but not with a number or other special characters. A variable name must not have a space and lowercase or uppercase are permitted.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
answer = 42
print(answer)
```

***Task:*** Correct the following errors in the variable names and print their values.

```{code-cell} ipython3
:tags: [solution]
1diameter = 1
length! = 10
circle_area = 3.1415926/4* diameter**2
#bar = 4
```

***Task:*** Write an equation for the volume of a cylinder using predefined variables.

```{code-cell} ipython3
:tags: [solution]

```

### Object Orientation/Types
Object-oriented Programming (OOP), is a programming paradigm which provides means of structuring programs so that properties and behaviors are bundled into individual objects. For instance, an object could represent a person with a name property, age, address, etc., with behaviors like walking, talking, breathing, and running. Or an email with properties like recipient list, subject, body, etc., and behaviors like adding attachments and sending.

A deeper introduction to OOP is out of scope for this course. However, it is important to know that in Python *everything* is an object. This means, it is of a certain *type* and every type brings with it certain behaviour. Python has five standard data types and we've already met some (subclasses) of them:
- Numbers
- String
- List
- Tuple
- Dictionary

The type can be determined using the ```type``` function.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
foo = 2.0
print(type(foo))
```

***Task:*** Print the variable value and type for ```answer```, and ```file_name```.

```{code-cell} ipython3
:tags: [solution]
answer = 42
file_name = "readme.md"
```

Almost everything in Python has *attributes* and *methods*.

+++

***Example:*** Complex numbers are an extension of the familiar real number system in which all numbers are expressed as a sum of a real part and an imaginary part. Imaginary numbers are real multiples of the imaginary unit (the square root of -1), written with $j$ in Python. You can access the real and imaginary parts of complex numbers using ```real``` and ```imganiary``` parts.

```{code-cell} ipython3
:tags: [solution]
c = complex(3,2)
print(type(c))
print(c.real)
print(c.imag)
```

***Task:*** The method ```upper()``` returns a copy of the string in which all case-based characters have been uppercased. Use this method to capitalize a string variable.

```{code-cell} ipython3
:tags: [solution]

```

### Containers
Lists are the most versatile compound data type for grouping together values in Python. A list contains items separated by commas and enclosed within square brackets (```[]```). The values stored in a list can be accessed using the slice operator (```[index]```, ```[start:end]```) with indexes starting at ***0*** in the beginning of the list and working their way to end ***-1***.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
my_list = ['abcd', 786 , 2.23, 'john', 70.2]
print(my_list)
print(my_list[0])
```

***Task:*** Print elements starting from 3rd element.

```{code-cell} ipython3
:tags: [solution]

```

***Task:*** Append ```99``` to the list using the ```append()``` method.

```{code-cell} ipython3
:tags: [solution]

```

### Dictionaries

Dictionaries in Python are unordered collections of key-value pairs enclosed within curly brackets (`{}`). Unlike lists, which are indexed by numbers, dictionaries are indexed by
keys, which can be either numbers or strings.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
my_dict = {'name': 'Cadet', 'version': '4.1.0', 'nUsers': 500}
print(my_dict)
print(my_dict['name'])
my_dict['nDownloads'] = 2000
print(my_dict['nDownloads'])
```

Dictionaries, like lists, can be nested.

```{code-cell} ipython3
:tags: [solution]
my_dict = {'name': 'Cadet', 'version': '4.1.0', 'nUsers': 500}
my_dict['stats'] = { 'github_stars': 1000, 'downloads': 2000, 'issues': 3}
print(my_dict)
```

#### Addict

Since the syntax to traverse deeply nested dictionaries can be quite tedious, we can use the package `addict` to simplify it to the dot notation.

```{code-cell} ipython3
:tags: [solution]
from addict import Dict

my_new_dict = Dict(my_dict)
print(my_new_dict.stats)
my_new_dict.stats.pull_requests = 10
print(my_new_dict.stats)
```

### Python syntax

In Python code blocks are structured by indentation level. It is is a language requirement not a matter of style. This principle makes it easier to read because it eliminates most of braces ```{ }``` and ```end``` statements which makes Python code much more readable. All statements with the same distance from left belong to the same block of code, i.e. the statements within a block line up vertically. If a block has to be more deeply nested, it is simply indented further to the right.

Loops and Conditional statements end with a colon ```:```. The same is true for functions and other structures introducing blocks.

+++

### Conditions

Conditional statements are used to direct the flow of the program to different commands based on whether a statement is ```True``` or ```False```. A boolean (```True``` or ```False```) is used to direct the flow with an ```if```, ```elif``` (else if), or ```else``` parts to the statement.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
x = 4
if x < 3:
    print('less than 3')
elif x<4:
    print('between 3 and 4')
else:
    print('greater than 4')
```

***Task:*** Write a conditional statement that checks whether the string ```'spam'``` is in ```menu```.
**Hint:** Check the operators list for membership statements.

```{code-cell} ipython3
:tags: [solution]
menu = ['eggs','bacon']
```

### Loops
Often, we don't know (or care) how many elements are in a container (e.g. list). We just want to perform an operation on every element of the container. The ```for``` statement in Python differs a bit from what you may be used to in C or Pascal. Python’s ```for``` statement iterates over the items of any sequence (e.g. a list or a string), in the order that they appear in the sequence.

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
list_of_primes = [2, 3, 5, 7]
for element in list_of_primes:
    print(element)
```

***Task:*** Create a list with strings, iterate over all elements and print the string and the length of the string using the ```len``` function.

```{code-cell} ipython3
:tags: [solution]

```

## Functions
A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing. There are built-in functions like ```print()```, ```help()```, etc. but it is possible to create your own functions. These functions are called user-defined functions.
A function definition describes what is to be calculated once the function is called. The keyword ```def``` introduces a function definition. It must be followed by the function name and the parenthesized list of formal parameters. The statements that form the body of the function start at the next line, and must be indented.

***Note***, the function is not evaluated when defined!

+++

***Example:***

```{code-cell} ipython3
:tags: [solution]
def circle_area(diameter):
    return 3.1415926/4 * diameter**2

area = circle_area(3)
print(area)
```

***Task:*** Define a function that returns the volume of a cylinder as a function of diameter and length.

```{code-cell} ipython3
:tags: [solution]

```
