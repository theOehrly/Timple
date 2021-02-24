# Timple

A package which provides extended support for plotting timedelta values in 
Matplotlib.


##  Overview

Matplotlib generally supports plotting of timedelta values but only as 
numeric values. It does not natively have locators and formatters to 
create fancy plot ticks.

This package provides the necessary locators and formatters to create 
axis ticks in intervals of minutes, hours, ... 
Ticks can be formatted into a more readable time format. For example, 
something like ``185`` seconds can be turned into ``3:05`` minute:seconds
representation.

#### Example images:

TBD


#### Features

- Formatters and Locators for timedelta
- Matplotlib patching, so that everything can happen with minimal effort
  
Additionally
- Support for ``pandas.NaT`` when plotting timedelta and optionally when
    plotting date values too.
  

## Usage

Minimal example: import and enable Timple. This will patch Matplotlib and
register Timple's timedelta converter. By default, Timple's automatic 
locators and formatters will be used to determine the tick locations and 
format best suited for the plotted data. Further customization is possible.

```
import matplotlib as mpl
import matplotlib.pyplot as plt
import timple

tmpl = timple.Timple(mpl)
tmpl.enable()

# you can now just use matplotlib as always

plt.plot(...some timedelta related data...)
plt.show()
```
