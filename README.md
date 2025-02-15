# Temp
When a town/location/population "joins the grid", the placement of certain generators is more or less guess and checked.

The goal of this project is to optimize this placement. The distance between location epicenters and generators can cost energy companies millions of dollars in power line technologies.
Ex: 1 mile of heavy duty line costs 1mil or something like that. 

I try to model a function that sums distances between population centers and account for populations themselves.

Ideal placements do not ignore islanded smaller populations. 

The project is organized into an ETL pipeline (or not, TBD)

Population epicenters are calculated using a center of mass equation (calculus)

Note to Benen - build this page out more.


TODO - 
    Populations API
    Build out plotter for a map (geoapndas?)

