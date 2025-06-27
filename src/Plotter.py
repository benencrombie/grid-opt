import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
import pandas as pd

#TODO use plotly express for webpage stuff
class Plotter():
    def __init__(self, data, locations:None, iter=0, score=0): 
        self.data = data
        self.locations = locations
        self.iter = iter
        self.score = score

    def base(self): 
        """
        Method to plot baseline map of virignia
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot zipcode map of virginia using population as color
        self.data.plot(ax=ax, column = "pop", edgecolor="black", cmap="Reds", alpha=0.7)
        ax.xaxis.set_visible(False)  # Hides x-axis (ticks and label)
        ax.yaxis.set_visible(False)  # Hides y-axis (ticks and la
        # Title and save
        plt.title(f"Population Map of Virginia")
        plt.savefig(f"assets/base_zipcode.png")

    def plot_plt(self): 
        """
        Method to plot the current map of the grid
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot zipcode map of virginia using the min_weight as the "score" 
        self.data.plot(ax=ax, column = "min_cost", edgecolor="black", cmap="Reds", alpha=0.7)
        # Plot all generators
        for loc in self.locations.values():
            plt.scatter(loc['x_coord'], loc['y_coord'], marker='^', s=(200*loc['weight']), color = 'blue')
        # Title and save
        plt.title(f"Iteration {self.iter}")
        plt.savefig(f"outputs/iter_{self.iter}.png")
        

        
        