# Toplogical Error Detection and Correction at Scale

## Instructions
### Use Case 1
Load Delhi building dataset(```Building_Delhi.gpkg```) and run ```error_gen.py``` with the the required parameters. This will generate a new .gpkg (corrupted.gpkg) which is used for Use Case 1(uploaded as delhi_corrupted.gpkg). This file contains intermediate layers and the final corrupted layer.
Open ```u1_delhi.py``` in QGIS Python console and ensure that the layer name and parameters are correctly set. Execute the script to see corrected layers with evaluation metrics.

### Use Case 2
Load the reprojected Jaipur layer(```jaipur_reprojected.gpkg```) and open ```u2_jaipur.py``` in QGIS python console. Ensure that the layer name and parameters are correctly set and execute the script to see corrected layers with evaluation metrics.

## Datasets
The datasets used can be found here: https://drive.google.com/drive/folders/1-yDtB3-0QdViY04RkZMM1Qt987EUIYh_?usp=sharing
