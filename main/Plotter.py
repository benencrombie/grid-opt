import matplotlib
matplotlib.use('Agg') # Avoid using GUI for backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig(f"assets/base_zipcode.png", bbox_inches='tight')
        plt.close()

    def plot_plt(self, file_name): 
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
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig(f"outputs/{file_name}.png", bbox_inches='tight')
        plt.close()

        
        