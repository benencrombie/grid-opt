import numpy as np
import pandas as pd
import geopandas as gpd
import json
import copy
from Plotter import Plotter


class SimulatedAnnealing:
    """
    This class is as a gradient descent iterator.
    """
    def __init__(self, ids:list=None): 
        self.locations = None                                      # Updated location data
        self.gpd_path = "resources/VA_Zip_Codes/VA_Zip_Codes.shp"  # Path to geopandas data
        self.max_iter = 5000                                       # Maximum number of iterations
        self.learning_rate = 50                                    # Learning rate. Score function is not sensitive enough
        self.dxy = 1                                               # Step size of xy coords
        self.dw = 0                                                # Step size of weight
        self.iter = 0                                              # Current iteration
        self.score = 0                                             # Current score
        self.gdf = None                                            # Geopandas dataframe
        self.loc_ids = ids                                         # List of location IDs to iterate through
        self.loadMap()
        self.cleanData()
        self.loadJson('Profile')
        self.getLocationIds()                             

    def loadMap(self): 
        """
        Method to load in geopandas data and add centroid

        Args: 
            None

        Returns: 
            None
        """
        self.gdf = gpd.read_file(self.gpd_path)

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
        # Noticed some populations equal to -99, this is an error code used in some datasets. Set the minimum to 1
        self.gdf['pop'] = self.gdf['pop'].clip(lower=1) 
        
    def loadJson(self, profile): 
        """
        Method to load existing locations from json -> eventually sqlite databsae

        Args: 
            profile (str): The saved profile

        Returns: 
            None
        """
        with open('data/locations.json', 'r') as file:
            self.locations = json.load(file)[profile]

    def getLocationIds(self): 
        """
        Method to get location IDs
        """
        if self.loc_ids: 
            pass
        else:
            self.loc_ids = self.locations.keys()

    #TODO refine cost equation jaunt
    # - Include facility weight
    # - Spacing between facilities (repulsive forces maybe)
    # - How much should distant populations be weighed?
    def calcCost(self, locations):
        """
        Method to calculate the score of power station placements

        Args:
            locations (list): list of tuples ~ (x_coord, y_coord, weight)

        Returns: 
            float: Score of input locations
        """
        # Always calculate score based on entire location set, even if only part of it is being optimized.            
        for i, loc in enumerate(locations): 
            # Cost function: natrual log of population times cartesian distance times the "weight" of a given location 
            self.gdf[f"cost_{i}"] = self.gdf['pop'] * np.sqrt(((self.gdf['x'] - locations[loc]['x_coord'])**2 + (self.gdf['y'] - locations[loc]['y_coord'])**2)) / locations[loc]['weight']
        # Get the minimum weight of each location 
        self.gdf["min_cost"] = self.gdf[[f"cost_{j}" for j in range(len(locations))]].min(axis=1)
        # Return average weight of all locations
        self.score = self.gdf["min_cost"].mean()
        return self.score
    
    def calcGradient(self):
        """
        Method to calculate the gradient of given generator placements

        Args:
            Non

        Returns: 
            np.array: nx2 matrix gradient
        """
        # Initialize the gradient in the same format as the location data
        grad = {key: {param: 0 for param in self.locations[key]} for key in self.locations}
        # Get a base score used to calculate the gradient.
        base_score = self.calcCost(self.locations)
        # Loop through existing locations
        for loc in self.loc_ids: 
            # Loop through parameters
            for param in self.locations[loc]:
                # Create a copy of the original dictionary to update one param at a time
                updated_locs = copy.deepcopy(self.locations)
                # Different step size for weight
                if param == "weight":
                    updated_locs[loc][param] += self.dw
                else: 
                    updated_locs[loc][param] += self.dxy
                # Calculated the score one step away (for each parameter)
                updated_score = self.calcCost(updated_locs)
                # Calculate partial derivative with respect to individual variables
                partial_derivative = (updated_score - base_score) / self.dxy
                # Build gradient
                grad[loc][param] = partial_derivative
        return grad
    
    def opt(self):
        """
        Method to optimizer the placement of all locations

        Args: 
            ids (list): List of IDs to optimize, default to None
            
        Returns: 
            (TBD): All optimized locations
        """
        #for i in range(self.max_iter + 1): 
        for i in range(self.max_iter+1):
            # Calculate gradient using the defined function
            grad = self.calcGradient()
            # Step in direction of the calculated gradient each iteration
            for loc in self.loc_ids: 
                for param in self.locations[loc]:
                    # Increment by the gradient component times the learning rate 
                    self.locations[loc][param] -= (grad[loc][param] * self.learning_rate)
            # Plot n evenly spaced iterations
            if i % (self.max_iter/5) == 0: 
                # Create an instance of the plotter class to create a visual
                plotter = Plotter(self.gdf, self.locations, self.iter)
                plotter.plot_plt()
            self.iter += 1
        return self.locations

    def rewriteJson(self): 
        pass


if __name__ == '__main__': 
    instance = SimulatedAnnealing(['100', '400'])
    gen_optimized = instance.opt()
    print(f"Optimized coordinates: {gen_optimized}")
    
    # grad = instance.calcGradient()
    # print(f"Gradient: {grad}")