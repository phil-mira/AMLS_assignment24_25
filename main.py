


if __name__ == "__main__":

    # Import the relevant libraries and files
    import pymongo
    import os

    import numpy as np
    import  pandas as pd    

    import prophet
    import sklearn as sk

    import matplotlib.pyplot as plt
    import seaborn as sns

    import yfinance as yf
    import holidays


    from A.main_A import main_A
    from B.main_B import main_B


    main_A()
    main_B()