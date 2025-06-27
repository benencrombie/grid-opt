import json
with open('data/locations.json', 'r') as file:
    data = json.load(file)['Profile']
    

for i in data.keys():
    print(i)
# base_locations = [(data_dict[loc["x_coord"]], data_dict[loc["y_coord"]], data_dict[loc["weight"]]) for loc in data_dict]
# print(base_locations)
