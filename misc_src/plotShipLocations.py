"""
Code to plot circles around ships in xView3 datasets
Must ensire are looking at correct label file - eiher public, validation or training.
Codes for files are suffixed with 'p', 'v' or 't' depending on which one.
So if scene_id is xxxxxxxp ensure that the csv file loaded is the 'public.csv' file

If working on Tiny dataset from xView3 and not files taken from Copernicus hub not the different naming conventions:
    The Tiny dataset does not contain the slew of folders associated with the Copernicus data

Also lots of NaN data mapped to -32768 and appears (for some files at least) that the data is compressed in amplitde between 
in a narrow range so may need to equalize ( or at least, stretch, the histogram


"""





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.colors import LogNorm
from matplotlib import colors
import rasterio
import skimage.measure as skm

# Step 1: Read CSV label file into a pandas dataframe
#csv_file     = 'public.csv' 
csv_file      = 'validation.csv'
#csv_file      = 'train.csv'
csv_path      = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/xView3Labels/'
csv_file_path = csv_path + csv_file
if not os.path.isfile(csv_file_path):
    print(f"Error: CSV file '{csv_file_path}' not found.")
    exit(1)


df = pd.read_csv(csv_file_path)
print("First 10 lines of the dataframe:")
print(df.head(10))

df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
print("Columns in the dataframe:", df.columns.tolist())

# Ensure 'SceneID' is present and correctly spelled
if 'scene_id' not in df.columns:
    print("Error: Column 'scene_id' not found in the CSV file.")
    exit(1)



# Step 2: Find all entries with label “SceneID” = xxxyyy
scene_id = '590dd08f71056cacv'                            ##
#scene_id = '05bc615a9b0e1159t'                             ##S1A_IW_GRDH_1SDV_20200722T164121_20200722T164146_033570_03E3EB_91B4 
#scene_id = 'b1844cde847a3942v'                            ##S1A_IW_GRDH_1SDV_20200126T052956_20200126T053025_030967_038E50_B43E  
#scene_id = '48c6404968760672p'                            ##S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53   
#scene_id = '028df92a2e68b7bap'                            ##S1B_IW_GRDH_1SDV_20200401T051836_20200401T051901_020946_027BB4_B4B9
filtered_df = df[df['scene_id'] == scene_id]

# Step 3: Extract required columns and store them in a dataframe
ship_locations = filtered_df[['detect_scene_row', 'detect_scene_column', 'detect_lat', 'detect_lon', 'confidence', 'is_vessel']]

# Step 4: Load the image file
base_dir   = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/'
root_name  = 'S1B_IW_GRDH_1SDV_20200401T051836_20200401T051901_020946_027BB4_B4B9/'
mode_type  = 'GRD/'
safe_dir   = 'S1B_IW_GRDH_1SDV_20200401T051836_20200401T051901_020946_027BB4_B4B9.SAFE/'
#image_path = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53/GRD/S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53.SAFE/measurement/'

#image_path = base_dir + root_name + mode_type + safe_dir +'measurement/'

#image_file ='s1a-iw-grd-vh-20200421t051412-20200421t051437-032221-03ba16-002.tiff'
#image_file ='s1a-iw-grd-vv-20200421t051412-20200421t051437-032221-03ba16-001.tiff'
#image_file = 's1b-iw-grd-vh-20200401t051836-20200401t051901-020946-027bb4-002.tiff'

#### Testing Tiny set
image_path = '/home/joe/Downloads/xView3TinyDataset/590dd08f71056cacv/'
#image_path = '/home/joe/Downloads/xView3TinyDataset/05bc615a9b0e1159t/'
#image_path = '/home/joe/Downloads/xView3TinyDataset/b1844cde847a3942v/'
image_file = 'VV_dB.tif'

image_file_path = image_path + image_file
if not os.path.isfile(image_file_path):
    print(f"Error: Image file '{image_file_path}' not found.")
    exit(1)


"""    
Image.MAX_IMAGE_PIXELS = None  # This will disable the decompression bomb error
image = Image.open(image_file_path)
if image.mode not in ['RGB', 'RGBA', 'L']:
    image = image.convert('RGB')

# Step 5: Downsample the image
down_sampling_rate = 10
downsampled_image = image.resize((image.width // down_sampling_rate, image.height // down_sampling_rate), Image.LANCZOS)
downsampled_image_array = np.array(downsampled_image)

downsampled_image_array = downsampled_image_array.astype(float)
downsampled_image_array -= np.min(downsampled_image_array)
downsampled_image_array /= np.max(downsampled_image_array)
"""

down_sampling_rate = 10
######################
i=0
src_file=image_file_path
src = rasterio.open(src_file)
image = src.read(1)
#image[image <=0]=1
mean_image = skm.block_reduce(image, (down_sampling_rate,down_sampling_rate), np.mean)



min_image = np.min(mean_image)
max_image = np.max(mean_image)
avg_image =np.mean(mean_image)
stddev_image = np.std(mean_image)

print(f"Image Min  = {min_image}")
print(f"Image Max  = {max_image}")
print(f"Image Mean = {avg_image}")
print(f"Image StdDev = {stddev_image}")
n_bins = 20
#hist = np.histogram(mean_image,n_bins, (min_image, max_image))
if 0:
   fig = plt.figure(figsize=(10,7))
   plt.hist(mean_image, n_bins)
   plt.title(" Image Historgram")
   plt.show()


input("Press Enter to continue...")


mean_image += 1500
mean_image[mean_image <0] = 0
min_image = np.min(mean_image)
max_image = np.max(mean_image)
avg_image =np.mean(mean_image)
stddev_image = np.std(mean_image)

print(f"Image Min  = {min_image}")
print(f"Image Max  = {max_image}")
print(f"Image Mean = {avg_image}")
print(f"Image StdDev = {stddev_image}")

if 0:
   fig = plt.figure(figsize=(10,7))
   plt.hist(mean_image, n_bins)
   plt.title(" Image Historgram")
   plt.show()


input("Press Enter to continue...")

plt.figure(figsize=(10, 10))
image_list=[]
image_list.append(mean_image)
#plt.imshow(np.fliplr(image_list[i]),norm=colors.LogNorm(vmin=10, vmax=512))
#plt.imshow(np.fliplr(image_list[i]))
#plt.imshow(image_list[i],norm=colors.LogNorm(vmin=10, vmax=512))
v_min = 1450
v_max =1500

plt.imshow(image_list[i],norm=colors.LogNorm(vmin = v_min, vmax = v_max))
#plt.imshow(np.flip((image_list[i]),1),norm=colors.LogNorm(vmin = v_min, vmax= v_max))
#plt.imshow(image_list[i], vmin = v_min, vmax = v_max)
#axis[i].imshow(np.flip(np.flip((image_list[i]),1),0),norm=colors.LogNorm(vmin=10, vmax= v_max))
#axis[i].imshow(np.transpose(image_list[i]),norm=colors.LogNorm(vmin = v_min, vmax=v_max))
#axis[i].set(xticks=[], yticks=[], facecolor="black")
#axis[i].set_title(f"{polarization[i]}", color ='green')
#plt.show()
#plt.show(block=False)


"""
####################
# Display the downsampled image
plt.figure(figsize=(10, 10))
plt.imshow(downsampled_image_array, norm=LogNorm(), cmap='viridis')
#plt.imshow(downsampled_image_array, norm=colors.LogNorm(vmin=10, vmax=512))
plt.colorbar()
"""


# Step 6: Plot circles at co-ordinates [detect_scene_row, detect_scene_column]
N = 300
j=0
num_true_vessels = 0
num_True_High_Confidence = 0
for _, row in ship_locations.head(N).iterrows():
    y = row['detect_scene_row'] // down_sampling_rate
    x = row['detect_scene_column'] // down_sampling_rate
    confidence = row['confidence']
    is_vessel  = str(row['is_vessel'])
    j+=1
    if is_vessel == 'True':
        num_true_vessels +=1
        if confidence == 'HIGH':
            edge_color = 'green'
            num_True_High_Confidenece+=1
        elif confidence == 'MEDIUM':
            edge_color = 'orange'
        else:
            edge_color = 'red'
    else:
        edge_color = 'black'

    
    circle = patches.Circle((x, y), radius=5, edgecolor=edge_color, facecolor='none', linewidth=2)
    plt.gca().add_patch(circle)

print(f"Number of true vessels = {num_true_vessels}")
print(f"Number of true HIGH-Confidence vessels = {num_True_High_Confidence}")
plt.show()
