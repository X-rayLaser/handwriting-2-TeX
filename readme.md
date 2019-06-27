# Introduction

handwriting-2-TeX is Machine Learning based app used to recognize 
simple mathematical expressions and convert them to TeX markup. The 
user draws a mathematical expression on the canvas using their mouse 
and gets back a valid TeX markup and its visualization.

As of the moment of this writing, the program can recognize expressions 
consisting of digits, numbers, powers, fractions. Expressions can also 
contain sums, differences, multiplications, division lines.

Program is shipped together with a pre-trained convolutional neural net.
It also comes with tiny GUI utility which allows a user to fine-tune the
network on examples of their own drawings.

# Installation

Clone the repository
```
git clone <clone_url>
```

, create a virtualenv environment using Python 3
```
virtualenv --python='/path/to/python3/executable' venv
```
, activate the environment
```
. venv/bin/activate
```

, go inside the directory containing a hidden git folder and install 
all python modules required for the app
```
pip install -r requirements.txt
```

# Quick Start

Run main.py script
```
python main.py
```
You should see a window containing a list of 
pre-trained models, a white canvas to draw on and a blank area for 
rendering recognized TeX expression.

Draw a simple expression of a kind "25 + 37". It might take a few 
seconds to recognize the expression after which you should see 
identical TeX expression and its rendering.
Click a button "Copy to clipboard" to copy the TeX markup to the clipboard.
Click a button "Erase" to erase everything on the canvas.

Experiment with more complex expressions like the following:
![alt text](drawing_example.png "Batch of generated sequences starting with 'A'")

## Fine-tuning

In order to increase the accuracy of recognition, you may want to 
calibrate a pre-trained model with a few examples of your own handwriting.
To do that, run 'fine_tune.py' script.
```
python fine_tune.py
```

You should see a window that 
contains a drawing area (white canvas surrounded by a red border) and 
a few buttons.

Draw a digit '0'  on the canvas. If you need to try again, push the 
button "Erase". When you are satisfied with a drawing, press a button 
"Add example". After that, draw remaining digits and signs (plus, 
minus, multiplication or X).

Now that you have provided an example of a drawing for each symbol, 
you can start the fine-tuning process. To do that, click a button 
"Start tuning". When it is done, it will output the expected 
classification accuracy of a new model. The weights of your new model 
are stored in "tuned_model.h5". You can now use this model in the app 
instead of the pre-trained one.

You should provide at least 3-5 drawings per each symbol to achieve 
high accuracy. When you perform fine-tuning for the first time, the 
program will make a copy of a pre-trained model and call it 
"tuned_model.h5". The following fine-tuning sessions will keep training 
this model.

# Limitations
Due to the way the system operates, it has difficulty recognizing 
certain expressions and handwriting.

Specifically, it has a limited object resolution. That is 2 symbols 
standing too close to each will be incorrectly recognized as one symbol. 
Another difficult case happens when a drawing of a symbol touches or 
intersects a division line. One can also get weird results when 
recognizing symbols with jerky discontinuous drawings.

Finally, each symbol is expected to have the width and height of 40-45 
pixels. The system is robust to slight variations in size, but it will 
fail on drawings corresponding to too large or too small fonts.

# License

GNU General Public License v3.0
