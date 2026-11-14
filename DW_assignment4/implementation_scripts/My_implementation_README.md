1. little adjustments:
- removed numba and llvmlite from reqs, too outdated
- updated their versions to 0.60.0 and 0.43.0
- removed  .decorators function from numba in utils.py
- imported csv
- import numpy as np
- import pandas as pd

2. The script does not calculate the parity difference (DP_Diff),
the disparate impact ratio (DIR), and the consistency
from the original labels of the dataset before mitigation. So I had
to implement that in the code. The python codes calc_German.py,
calc_adult.py and calc_COMPAS.py show how I made those functions, before
implementing them in the main.py. I did not add the Compass dataset into
the dataloader() function in the main.py, because I did not understand
how to correclty implement that dataset. Although I did make the
function that calculates the parity difference, the disparate impact
ratio and the consistency of the Compass dataset.

3. I implemented 3 if-functions in main.py that
call the correct function for the chosen dataset.

4. Because I am using the calculation functions in three different
processes, I put the value "process" as an argument, containing numbers
1-3, representing in which process the calculation is being used
(whole data before mitigation, y_test before mitigation,
y_test after mitigation).


5. Implementation scripts: 
- calc_German.py
- calc_adult.py
- calc_COMPAS.py

6. The whole pipeline can be run by simply running Pipeline.ipynb.
Before doing that, project syncing needs to be done.
also:
- install all packages from requirements.txt

It can happen that it will show an error, then in terminal run:
python -m pip install --upgrade "matplotlib>=3.7,<3.10" "matplotlib-inline>=0.1.7,<0.2"