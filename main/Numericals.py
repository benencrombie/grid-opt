import numpy as np
import pandas as pd
import geopandas as gpd
import json
import copy
import os
from pathlib import Path
# from Plotter import Plotter
from .Plotter import Plotter


class SMROptimizer:
    """
    This class is as a gradient descent iterator.
    """
    def __init__(self, method="GD", max_iter=None,ids:list=None): 
        self.method = method                                       # Optimization method - Gradient Descent or Simulated Annealing
        self.iter = 0                                              # Current iteration, applicable to all computational methods
        self.gdf = None                                            # Geopandas dataframe
        self.max_iter = int(max_iter)                              # Max iterations, applicable to all computational methods
        self.loc_ids = ids                                         # List of location IDs to iterate through. Ones not included in list are static, but still used for calculating score
        self.base_dir = Path(__file__).resolve().parents[1]        # Getting a base directory so files can be accessed in Flask
        self.loadMap()                                             # Load in map data, previously scraped (outside of this module)
        self.cleanMapData()                                        # Perform any preprocessing operations of the map data
        self.loadModelConfig(self.method)                          # Load model configurations
        self.loadGridConfig('Profile')                             # Load existing grid configurations (i.e., existing placements of SMRs)
        self.getLocationIds()                                      # Get location IDs from grid configuration

    def loadMap(self): 
        """
        Method to load in geopandas data and add centroid

        Args: 
            None

        Returns: 
            None
            
        Raises: 
            None
        """
        # Appending the exact location to the grid-opt folder
        path = self.base_dir / "resources" / "VA_Zip_Codes" / "VA_Zip_Codes.shp"
        # Loading shape file(s) through geopandas 
        self.gdf = gpd.read_file(path)

    def cleanMapData(self): 
        """
        Method to clean geopadnas dataframe

        Args: 
            None

        Returns: 
            None
            
        Raises: 
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
        
    def loadModelConfig(self, method:str): 
        """
        Method to load the the model configurations
        
        Args: 
            method (str): The optimization method. 
                - GD: Gradient Descent
                - SA: Simulated Annealing
                
        Returns: 
            None
            
        Raises: 
            ValueError for unknown method
        """
        path = self.base_dir / "config" / "model_config.json"
        with open(path, "r") as file:
            model_config = json.load(file)[method] 
        if method == "GD":
            # If no max iter is chosen, use model defaults
            if not self.max_iter: 
                self.max_iter = model_config["max_iter"]
            self.learning_rate = model_config["learning_rate"]
            self.dxy = model_config["dxy"]
            # Weight should be standardized and treated equitably with position. The exact dw value should be determined outside of this module - default is 0 (SMR placements are not dependent on SMR size)
            self.dw = model_config["dw"]                   
            print("Running Gradient Descent")
        elif method == "SA":
            if not self.max_iter:
                self.max_iter = model_config["max_iter"]
            print("Runnign Simulated Annealing")
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def loadGridConfig(self, profile:str): 
        """
        Method to load existing locations from json

        Args: 
            profile (str): The saved profile

        Returns: 
            None
            
        Raises: 
            None
        """
        path = self.base_dir / "config" / "grid_config.json"
        with open(path, "r") as file:
            self.locations = json.load(file)[profile]

    def getLocationIds(self): 
        """
        Method to get location IDs
        
        Args: 
            None
            
        Returns: 
            None
            
        Raises: 
            None
        """
        if self.loc_ids: 
            pass
        else:
            self.loc_ids = self.locations.keys()
            
    def clearOutputs(self): 
        """
        Method to clear all images from the outputs folder.  
        
        Args: 
            None
            
        Returns: 
            None
            
        Raises: 
            None
        """
        folder = 'outputs'
        # Loop through all files within the folder, grab filepaths to unlink 
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            os.unlink(file_path)  

    def evaluateState(self, locations):
        """
        Method to calculate the score of power station placements

        Args:
            locations (list): list of tuples ~ (x_coord, y_coord, weight)

        Returns: 
            float: Score of input locations
            
        Raises: 
            None
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
    
    # === Gradient Descent Specific Methods ===
    
    def calcGradient(self):
        """
        Method to calculate the gradient of given generator placements

        Args:
            Non

        Returns: 
            np.array: nx2 matrix gradient
            
        Raises: 
            None
        """
        # Initialize the gradient in the same format as the location data
        grad = {key: {param: 0 for param in self.locations[key]} for key in self.locations}
        # Get a base score used to calculate the gradient.
        base_score = self.evaluateState(self.locations)
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
                updated_score = self.evaluateState(updated_locs)
                # Calculate partial derivative with respect to individual variables
                partial_derivative = (updated_score - base_score) / self.dxy
                # Build gradient
                grad[loc][param] = partial_derivative
        return grad
    
    def gradientDescent(self):
        """
        Method to optimizer the placement of all locations using gradient descent

        Args: 
            ids (list): List of IDs to optimize, default to None
            
        Returns: 
            dict: Locations with optimized coordinates
            
        Raises: 
            None
        """
        # Clear all output images (from previous runs)
        self.clearOutputs()
        # Iterate Gradient Descent
        for i in range(self.max_iter+1):
            # Calculate gradient using the defined function
            grad = self.calcGradient()
            # Step in direction of the calculated gradient each iteration
            for loc in self.loc_ids: 
                for param in self.locations[loc]:
                    # Increment by the gradient component times the learning rate 
                    self.locations[loc][param] -= (grad[loc][param] * self.learning_rate)
            # Plot n evenly spaced iterations
            if i % (self.max_iter/10) == 0: 
                # Create an instance of the plotter class to create a visual
                plotter = Plotter(self.gdf, self.locations, self.iter)
                # Declare filepath to pass to plotter and also access in the application
                img_name = f"iter_{self.iter}"
                plotter.plot_plt(img_name)
                # Yield intermediate results for matplotlib figs to update in the frontend
                path_name = f"{img_name}.png"
                yield (i, path_name)
            self.iter += 1
        return self.locations

    # === Simulated Annealing Specific Methods === 

    def identifyCandidates(self): 
        """ 
        Method to identify candidate model states using pertrubation vectors
        """
        pass
    
    def simulatedAnnealing(self): 
        """
        Method to optimizer the placement of all locations using simulated annealing

        Args: 
            ids (list): List of IDs to optimize, default to None
            
        Returns: 
            dict: Locations with optimized coordinates
            
        Raises: 
            None
        """
        # Clear all output images (from previous runs)
        self.clearOutputs()
        # Iterate Simulated Annealing
        for i in range(self.max_iter+1):
            if i % (self.max_iter/10) == 0: 
                # Create an instance of the plotter class to create a visual
                plotter = Plotter(self.gdf, self.locations, self.iter)
                # Declare filepath to pass to plotter and also access in the application
                img_name = f"iter_{self.iter}"
                plotter.plot_plt(img_name)
                # Yield intermediate results for matplotlib figs to update in the frontend
                path_name = f"{img_name}.png"
                yield (i, path_name)
            self.iter += 1
        return self.locations
    
    # === Other Methods as Need ===
    

if __name__ == '__main__': 
    instance = SMROptimizer(method = 'GD', max_iter = 100)
    print('Cleared Previous Outputs')
    try: 
        gen_optimized = instance.gradientDescent()
        for i in gen_optimized: 
            print('Intermediate Step Complete')
    except StopIteration as e:
        final_result = e.value
        print(f"Optimized coordinates: {final_result}")