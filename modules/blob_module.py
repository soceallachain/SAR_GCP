###############################################################################
##  Module containing code for blob detection 
##      Accepts binary numpy image
##      Locates all blobs
##      Connects blobs separated by 2 pixels or less
##      Removes blobs of size 2x2 and smaller
##      Returns number of blobs and centroids


import numpy as np
#import matplotlib as plt
from skimage.measure import find_contours, label
from scipy.ndimage import label, find_objects, center_of_mass, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import itertools
from SAR_code_module import *
from SAR_focusing_module import *


from collections import Counter

def find_duplicates_with_counts(lst):
    counter = Counter(lst)
    # Filter elements with count > 1 and return as a list of (element, count) tuples
    return [(item, count) for item, count in counter.items() if count > 1]


# Merge blobs based on centroid separation
def merge_blobs(blobs, centroids, distance_threshold=2):
    merged = []
    used = set()
    for i, centroid1 in enumerate(centroids):
        if i in used:
            continue
        current_blob = blobs[i]
        for j, centroid2 in enumerate(centroids):
            if j in used or i == j:
                continue
            dist = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
            if dist <= distance_threshold:
                current_blob = current_blob | blobs[j]
                used.add(j)
        merged.append(current_blob)
    return merged 




# Merge blobs based on closest approach distance
def merge_nearby_blobs(blobs, threshold):
    """
    Merge blobs (as masks) whose contours are closer than a given threshold.

    Parameters:
        blobs (List[np.ndarray]): List of boolean arrays, each a mask for one blob.
        threshold (float): Distance below which blobs should be merged.

    Returns:
        List[np.ndarray]: List of merged boolean blob masks.
    """
    num_blobs = len(blobs)

    # Extract contours
    contours = [
        find_contours(blob.astype(float), level=0.5)[0]
        for blob in blobs
    ]

    # Build a merge graph based on contour proximity
    merge_graph = {i: set() for i in range(num_blobs)}
    for i, j in itertools.combinations(range(num_blobs), 2):
        tree = cKDTree(contours[j])
        distances, _ = tree.query(contours[i])
        if distances.min() < threshold:
            merge_graph[i].add(j)
            merge_graph[j].add(i)

    # Find connected components in the graph
    def dfs(node, visited, component):
        visited.add(node)
        component.add(node)
        for neighbor in merge_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, component)

    visited = set()
    components = []
    for i in range(num_blobs):
        if i not in visited:
            comp = set()
            dfs(i, visited, comp)
            components.append(comp)

    # Merge blobs in each connected component
    merged_blobs = []
    merged_centroids = []
    for component in components:
        merged_mask = np.zeros_like(blobs[0], dtype=bool)
        for blob_idx in component:
            merged_mask |= blobs[blob_idx]
        merged_blobs.append(merged_mask)
        #merged_centroids.append(center_of_mass(merged_mask))

    return merged_blobs



def remove_small_blobs(blobs, min_size=4):
    filtered = []
    for blob in blobs:
        if np.sum(blob) > min_size:
            filtered.append(blob)
    return filtered

def get_largest_blobs_centroid(non_binary_image, feature_location, 
                               intensity_threshold, blob_crop_size = [512,512]
                               ):
    
    r,c, loc_in_crop = place_roi(non_binary_image, blob_crop_size, 
                                 feature_location
                                 )
    
    binary_image = non_binary_image > intensity_threshold
    mask = binary_image[r:r+blob_crop_size[0],
                        c:c+blob_crop_size[1]
                        ]
    mask = dilation(erosion(mask))
    #merge blobs within nearness_threshold of each other
    nearness_threshold = 10
    blob_data = detect_blobs(mask, nearness_threshold)
    num_blobs = blob_data[0]
    centroids = blob_data[1]
    blobs     = blob_data[2]
    radii     = blob_data[3]
    if num_blobs == 1:
        return centroids, blobs, radii
    else:
        #Find largest blob by area or radius
        largest_blob_index = np.argmax(radii)
        largest_blob       = blobs[largest_blob_index]
        com_largest_blob   = centroids[largest_blob_index]
        return com_largest_blob, largest_blob, radii[largest_blob_index], blobs
        
        
    



# Find blobs. Any smaller than 2x2 are removed.
# Any within 2 pixels of each other are merged
def detect_blobs(binary_image, threshold):
    #Ensure separate all non-contiguous regions into blobs using 8-connectivity
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity for 2D
    labeled, num_features = label(binary_image, structure=structure)
    #labeled, num_features = label(image)
    blobs = []
    centroids = []
    for i in range(1, num_features + 1):
        blob = labeled == i
        centroid = center_of_mass(blob)
        blobs.append(blob)
        centroids.append(centroid)
    #merged_blobs = merge_blobs(blobs, centroids)
    proximity_threshold = threshold
    filtered_blobs = remove_small_blobs(blobs)
    merged_blobs  = merge_nearby_blobs(filtered_blobs, proximity_threshold)
    merged_centroids = [center_of_mass(blob) for blob in merged_blobs]
    
    # Calculate radii for each merged blob
    radii = []
    for blob in merged_blobs:
        dt = distance_transform_edt(blob)
        radius = np.amax(dt)
        radii.append(radius)
   
    return len(merged_blobs), merged_centroids, merged_blobs, radii
    
    


def plot___blobs(image, blobs):
    fig1, ax1 = plt.subplots()
    plt.show()
    ax1.imshow(image, cmap='gray', aspect ='auto')
    for blob in blobs:
        contours = np.argwhere(blob)
        y, x = zip(*contours)
        ax1.scatter(x, y, color='red', s=1)
    #plt.title(f'Number of Blobs: {len(blobs)}')
    fig1.suptitle(f'Number of Blobs: {len(blobs)}')
    #plt.show()
    
    
    
def CFAR_statistics(chip_file, label_file, save_path, plot_results=1, 
                    pol='vh', swath=2, 
                    train_cells_size= 20, guard_ring_size = 3, pfa = 5e-2
                    ):
    
    
    csv_filename = label_file
    swath_of_interest = swath
    polarization      = pol
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index
    # What mode are the data be used for : SLC or RAW
    sar_mode = 'RAW'

    #Set paramters to enable finding chip number for each ship
    chip_size =[512, 3072]
    # SLC range sampled at 64.345238MHz, but RAW is swath dependent
    # Have to adjust slc burst size to match raw burst size depending on 
    # range sampling frequency so set freq_mutipliers to RAW_f/SLC_f
    freq_multipliers=[1, 54.595959/64.345238, 46.915402/64.345238]
    
    if sar_mode == 'RAW':
       sampling_freq_mul=freq_multipliers[IW_mode]
    else:
       sampling_freq_mul = 1
    
    kernel1    = np.array([[1,1,1],
                           [1,1,1],
                           [1,1,1]])
    print(f'Chip_file_name={chip_file}')
    chip_info=get_slc_id_burst_and_chip_numbers(chip_file)
    SLC_ID = chip_info[0]
    burst_number = chip_info[1]
    chip_number = chip_info[2]
    
    ship_locations = get_ship_locations(chip_info, csv_filename)

    chip_and_burst_data = get_slc_chip_and_bursts(SLC_ID, burst_number, 
                                chip_number, sampling_freq_mul, chip_size
                                )
    chip_data = get_chip_data(chip_and_burst_data[1], chip_number, chip_size)
    #plot_chip(chip_data, SLC_ID, burst_number, chip_number)

    #Do CFAR detection
    cfar_gamma_map2=optimized_cfar_detector(chip_data, train_cells_size, 
                                            guard_ring_size, pfa
                                            )
    # Perform morphological operations for removing noise / opening -closing
    cfar_gamma_map2_eroded=erosion(cfar_gamma_map2,kernel1)
    cfar_gamma_map2_eroded_dilated=dilation(cfar_gamma_map2_eroded,kernel1)              
    cfar_gamma_map2_eroded_dilatedx2=dilation(cfar_gamma_map2_eroded_dilated,
                                              kernel1
                                              )


    # Find blobs >2x2 in size and 
    # Find blobs > 5 pixels apart
    nearness_threshold = 5
    blob_data=detect_blobs(cfar_gamma_map2_eroded_dilatedx2, nearness_threshold)
    
    number_of_blobs = blob_data[0]
    blob_centroids  = blob_data[1]
    blobs           = blob_data[2]


    #plot___blobs(cfar_gamma_map2, blobs)
    # Set distance threshold below which is considered a match beteen ship & blob
    distance_threshold  = 10
    ship_chip_locations = ship_locations.iloc[:,4:6].to_numpy().astype(int)
    blob_chip_locations = np.array(blob_centroids).astype(int)  
    number_of_ships     = len(ship_chip_locations)
    distances           = distance.cdist(blob_chip_locations, ship_chip_locations)
    ship_min_distances  = np.min(distances, axis=0)
    blob_min_distances  = np.min(distances, axis=1)
    ship_min_indices    = np.argmin(distances, axis = 0)
    blob_min_indices    = np.argmin(distances, axis = 1)
    
    #Get ship detection statistics
    if number_of_ships <= number_of_blobs:
        # Determine how many ships are within threshold distance of blobs
        #First check that unique mapping between blobs and ships exists
        if len(set(ship_min_indices)) == len(ship_min_indices):
            number_true_positives  = sum(ship_min_distances 
                                   < distance_threshold)
        else:
            #Have at least one instance of more than 1 ship per blob 
            #For each instance find which ship is closest to blob
            #Fnd how many repeated indices exist
            
            valid_blob_min_distances = blob_min_distances[list(set(ship_min_indices))]
            number_true_positives  = sum(valid_blob_min_distances 
                                   < distance_threshold)
            
        number_false_positives = number_of_blobs - number_true_positives
        number_false_negatives = number_of_ships - number_true_positives 
    else:
        if len(set(blob_min_indices)) == len(blob_min_indices):
            number_true_positives = sum(blob_min_distances < distance_threshold)
        else:
            valid_ship_min_distances = ship_min_distances[list(set(blob_min_indices))]
            number_true_positives  = sum(valid_ship_min_distances 
                                   < distance_threshold)
            
        number_false_positives = number_of_blobs - number_true_positives
        number_false_negatives = number_of_ships - number_true_positives
        



    if plot_results:
        save_fig_path= save_path
        save_fig_file=(f'{SLC_ID[-4:]}_Burst#{burst_number}_Chip#{chip_number}'
                      f'_#S{number_of_ships}_#B{number_of_blobs}_#TP{number_true_positives}'
                      f'_#FP{number_false_positives}_#FN{number_false_negatives}.jpg')
        if not os.path.exists(save_fig_path):
            print(f'ERROR: Directory to save figure in: {save_fig_path} does not exist. Exiting')
            sys.exit(0)
            
        #plot slc of chip 
        rect_width=100
        rect_height=100      
        fig, ax=plt.subplots(2,1)
        ax[0].imshow(chip_data, norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
        ax[0].set_title(f'{SLC_ID}  Burst#{burst_number} Chip#{chip_number}')
        s=np.reshape(ship_chip_locations,[len(ship_chip_locations),2])
        x=s[:,1]
        y=s[:,0]
        for xi, yi in zip(x, y):
            rect = patches.Rectangle((xi-rect_width/2, yi-rect_height/2), rect_width, rect_height,
                     edgecolor='y', facecolor='none', alpha=0.6)
            ax[0].add_patch(rect)
            
        ax[1].imshow(chip_data, norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
        plot_blobs_in_figure(ax[1],blobs)
        param_legend = [
        Line2D([0], [0], color='none', label=f'#Ships = {number_of_ships}'),
        Line2D([0], [0], color='none', label=f'#Blobs = {number_of_blobs}'),
        Line2D([0], [0], color='none', label=f'#TPs    = {number_true_positives}'),
        Line2D([0], [0], color='none', label=f'#FPs    = {number_false_positives}'),
        Line2D([0], [0], color='none', label=f'#FNs    = {number_false_negatives}')
        ]
        ax[1].legend(handles=param_legend, loc='upper left', framealpha=0.0)
        plt.savefig(f'{save_fig_path}/{save_fig_file}', dpi=300)
        plt.close(fig)
        
    
        
    result = {
       'SLC_ID': SLC_ID,
       'burst_number': burst_number,
       'chip_number': chip_number,
       'number_of_ships': number_of_ships,
       'number_of_blobs': number_of_blobs,
       'number_true_positives': number_true_positives,
       'number_false_positives': number_false_positives,
       'number_false_negatives': number_false_negatives
    }

    # Return as a one-row DataFrame
    return pd.DataFrame([result])
        







