import pandas as pd
import numpy as np

def load_mnist():
    df = pd.read_csv("../data/mnist_train.csv")
    
    data = df.drop("label", axis=1).values
    images = data.reshape(-1, 28, 28)
    
    return images[:1000]

