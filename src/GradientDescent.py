import numpy as np
import pandas as pd
import geopandas as gpd
from Plotter import Plotter


class GradientDescent:
    """
    This class acts as a gradient descent iterator.
    """
    def __init__(self, path, num_stations): 
        self.num_stations = num_stations        # Number of stations
        self.stations = None                    # Station locations   
        self.path = path                        # Path to geopandas data
        self.max_iter = 5000                    # Maximum number of iterations
        self.learning_rate = 50                 # Learning rate. Score function is not sensitive enough
        self.epsilon = 1                        # Step size used to calculate gradient. Based on coordinates (~10e6)
        self.iter = 0                           # Current iteration
        self.score = 0                          # Current score
        self.gdf = None                         # Geopandas dataframe
        self.loadData()
        self.cleanData()
        self.getInitialGuess()                  # Call method to generate guesses

    def loadData(self): 
        """
        Method to load in geopandas data and add centroid

        Args: 
            str: Path to geopandas data

        Returns: 
            None
        """
        self.gdf = gpd.read_file(self.path)

    def cleanData(self): 
        """
        Method to clean geopadnas dataframe

        Args: 
            None

        Returns: 
            None
        """
        self.gdf['centroid'] = self.gdf.geometry.centroid
        # Splitting into x and y coordinates to facilitate calculation.
        self.gdf['x'] = self.gdf['centroid'].x
        self.gdf['y'] = self.gdf['centroid'].y
        # Renaming to pop to keep model functional.
        self.gdf = self.gdf.rename(columns={'POPULATION': 'pop'})
        # Noticed some populations equal to -99, this is an error code used in some datasets.
        self.gdf['pop'] = self.gdf['pop'].clip(lower=1)

    def getInitialGuess(self): 
        """
        Method to get an initial guess given the number of stations

        Args: 
            None

        Returns: 
            None
        """
        # x_upper = self.gdf['x'].max()
        # x_lower = self.gdf['x'].min()
        # y_upper = self.gdf['y'].max()
        # y_lower = self.gdf['y'].min()
        # self.stations = np.random.uniform(x_lower, x_upper, (2, self.num_stations))     # Generate 2xn array of x coordinates
        # self.stations[:, 0] = np.random.uniform(y_lower, y_upper, (2,))                 # Generate the y column
        self.stations = np.array([[-9.0e6,4.4e6],     
          [-8.8e6,4.4e6],
          [-8.6e6,4.4e6]])    


    def calcScore(self, placements):
        """
        Method to calculate the score of power station placements

        Args:
            None

        Returns: 
            np.array: nx2 matrix of power station coordinates
        """
        for i, station in enumerate(placements): 
            # Weight function: natrual log of population times cartesian distance.
            self.gdf[f"weight_{i}"] = np.log(self.gdf['pop']) * np.sqrt(((self.gdf['x'] - station[0])**2 + (self.gdf['y'] - station[1])**2))
        # Get the minimum weight of each location 
        self.gdf["min_weight"] = self.gdf[[f"weight_{j}" for j in range(len(placements))]].min(axis=1)
        # Return average weight of all locations
        self.score = self.gdf["min_weight"].mean()
        return self.score
    
    def calcGradient(self):
        """
        Method to calculate the gradient of given generator placements

        Args:
            None

        Returns: 
            np.array: nx2 matrix gradient
        """
        # Initialize the gradient with the same dimensions as our outputs.
        grad = np.zeros_like(self.stations)
        # Get a base score used to calculate the gradient.
        base_score = self.calcScore(self.stations)
        # The data is two dimensional, so need two loops to cover all elements.
        for row_idx, row in enumerate(self.stations):
            # Loop through all paramerers in the input matrix
            for col_idx, value in enumerate(row):
                # Make a copy of the inputs to change one parameter at a time.
                updated_inputs = self.stations.copy()
                # Add epsilon to the parameter 
                updated_inputs[row_idx, col_idx] = value + self.epsilon
                # Calculate the updated score after adding epsilon to a single variable
                updated_score = self.calcScore(updated_inputs)
                # Calculate the partial with respect to the variable
                partial_derivative = (updated_score - base_score) / self.epsilon
                # Add the partial to the gradient matrix in the same location as the parameters in the input matrix.
                grad[row_idx, col_idx] = partial_derivative
        return grad
    
    def main(self):
        """
        Main Method

        - Load and clean data
        - Iterate through gradient descent
        - Plot map 10 evenly spaced times (including last iteration)
        """
        for i in range(self.max_iter + 1): 
            # Calculate gradient using the defined function
            grad = self.calcGradient()
            # Step in direction of the calculated gradient each iteration
            self.stations -= self.learning_rate * grad
            # Plot n evenly spaced iterations
            if i % (self.max_iter/5) == 0: 
                # Create an instance of the plotter class to create a visual
                plotter = Plotter(self.gdf, self.stations, self.iter)
                plotter.main()
            self.iter += 1
        return self.stations


if __name__ == '__main__': 
    # Running main
    path = "../data/VA_Zip_Codes/VA_Zip_Codes.shp"
    instance = GradientDescent(path, 5)
    gen_optimized = instance.main()
    print(f"Optimized coordinates: {gen_optimized}")