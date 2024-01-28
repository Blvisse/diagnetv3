## This script is used to unzip the data files downloaded into the model
try:
    import pandas as pd
    import zipfile
    from tqdm import tqdm as tq
    from dotenv import load_dotenv
    from tqdm import tqdm as tq
    print("Successfully Imported Libararies")
except ModuleNotFoundError as mnf:
    print("Error: Module Not Found: ")
    print("Details: {} ".format(mnf))


#describe file path
tb_chest_data="../data/tuberculosis-chest-xrays-shenzhen.zip"
destination_path="../data/unzipped_images/shenzhen_images/"

#Unzip data files and create a status bar to notify progress
print("Extracting images from: {}".format(tb_chest_data))
try:
    with zipfile.ZipFile(tb_chest_data, 'r') as zipped_images:
        zipped_images.extractall(destination_path) 
except Exception as ie:
    print("Error while extracting data: ")
    print("Deatils: {}".format(ie))
    exit(1)
print("Done Extracting to {}".format(destination_path))



