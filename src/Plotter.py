import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
import pandas as pd

class Plotter():
    def __init__(self, location_data, stations, iter=0, score=0): 
        self.location_data = location_data
        self.stations = stations
        self.iter = iter
        self.score = score

    def main(self): 
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot zipcode map of virginia using the min_weight as the "score" 
        self.location_data.plot(ax=ax, column = "min_weight", edgecolor="black", cmap="Reds", alpha=0.7)
        # Plot all generators
        for gen in self.stations:
            plt.scatter(gen[0], gen[1], marker='^', s=200, color = 'blue')
        # Title and save
        plt.title(f"Iteration {self.iter}")
        plt.savefig(f"../outputs/iter_{self.iter}.png")