# regression_school
## Klassenarbeitsersatzleistung Mathe 8A
## Berkay, Peer

# Easy Regresion 
## What it is
EasyRegression is a python library to easily perform regression on data. 
It uses Gradient descent to minimize the squared error
## Installation
To install EasyRegression simply run 
```
$ pip install EasyRegression
``` 
Requierments are numpy and matplotlib
## Example
```python
from EasyRegression.EasyRegression import EasyRegression
x = [2,3,1,4,5,2,3,5,2,4,6,4,3,3]
y = [4,5,2,5,2,3,5,2,4,6,2,4,6,2]
easy_reg = EasyRegression(np.zeros(5))
easy_reg.train(x,y,1e-33, 20000)
```
A video to describe it better: \
[![Demo of EasyRegression](https://img.youtube.com/vi/ewt2G2DBh30/0.jpg)](https://www.youtube.com/watch?v=ewt2G2DBh30)