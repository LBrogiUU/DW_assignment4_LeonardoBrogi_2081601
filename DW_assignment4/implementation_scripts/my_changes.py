"""
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

3. I implemented 3 if-functions in main.py,
call the correct function for the chosen dataset.

4. because I am using the calculation functions in three different
processes, I put the value "process" as an argument, containing numbers
1-3, representing in which process the calculation is being used
(whole data before mitigation, y_test before mitigation,
y_test after mitigation).


Implementation scripts: my_changes.py, calc_German.pym
calc_adult.py, calc_COMPAS.py
"""






"""
Y TEST:      classification  gender
405               0       1
926               1       0
593               0       0
819               0       1
714               0       1
..              ...     ...
998               0       1
873               1       0
335               0       1
751               0       0
555               0       1

[100 rows x 2 columns]

X TEST: [[-0.44272692  0.2618066  -1.77361642 ...  0.          0.
   1.        ]
 [-0.79960035 -0.23564436  0.91368119 ...  0.          1.
   0.        ]
 [-0.19796222  0.2618066   0.01791532 ...  0.          0.
   1.        ]
 ...
 [ 0.04211627 -1.23054629 -1.77361642 ...  0.          0.
   1.        ]
 [-0.82591525 -0.23564436 -1.77361642 ...  0.          1.
   0.        ]
 [-0.69794549 -0.73309532 -0.87785055 ...  0.          1.
   0.        ]]
   
Y PREDICTION: [1 1 0 1 0 0 1 1 0 0 1 1 1 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0
 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1]
"""