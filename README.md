# Timple

Timple offers extended functionality for plotting timedelta values with 
Matplotlib.


## Installation
Installation via pip is recommended:
``pip install timple``


##  Overview

Matplotlib generally supports plotting of timedelta values but only as 
numeric values and only for limited data types. It does not natively 
have locators and formatters to create fancy plot ticks.

This package provides the necessary locators and formatters to create 
axis ticks in intervals of minutes, hours, ... 
Ticks can be formatted into a more readable time format. For example, 
something like ``185`` seconds can be turned into ``3:05`` minute:seconds
representation.

Example plot:

![image of example plot](docs/_static/intro_example.svg)

The full documentation can be found here: https://theoehrly.github.io/Timple/

#### Features

- Formatters and Locators for timedelta
- Matplotlib patching, so that everything can happen with minimal effort
- Supports ``numpy.timedelta64``, ``datetime.timedelta``, ``pandas.Timedelta``
  
Additionally

- Support for ``pandas.NaT`` when plotting timedelta and optionally when 
  plotting date values too.
  

## Usage

Minimal example: import and enable Timple. This will patch Matplotlib and
register Timple's timedelta converter. By default, Timple's automatic 
locators and formatters will be used to determine the tick locations and 
format best suited for the plotted data. Further customization is possible.


    import matplotlib.pyplot as plt
    import timple
    
    tmpl = timple.Timple()
    tmpl.enable()
    
    # you can now just use matplotlib as always
    
    plt.plot(...some timedelta related data...)
    plt.show()

