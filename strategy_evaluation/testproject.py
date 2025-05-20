import numpy as np
from ManualStrategy import ManualStrategy
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from marketsimcode import compute_portvals
from util import get_data, plot_data
from experiment1 import run_exp1
from experiment2 import run_exp2

def author():
    return 'dhematillake3'
if __name__ == "__main__":
    run_exp1()
    run_exp2()






