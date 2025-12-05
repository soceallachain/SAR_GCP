"""
Script to read SM SLC ammotation file  and plot ROI 
User can chose point in ROI based on plot co-ordinates and then
obtain SCL co-ordinates for pixel in SLC space.
Script also returns GPS location of point oin ROI based on 
reference Azimuth, Range grid and GPS grid

"""

from  SAR_code_module import *
from matplotlib.patches import Rectangle









filepath  = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048953_05D835_4B93/SLC/product/S1A_S3_SLC__1SDH_20230518T213602_20230518T213627_048593_05D835_8BAA.SAFE/"

SM_SLC_fname ="s1a-s3-slc-hh-20230518t213602-20230518t213627-048593-05d835-001.tiff"
src_file        = filepath + "measurement/" + SM_SLC_fname
annotation_file = filepath + "annotation/"  + change_extension(SM_SLC_fname,"xml")


if os.path.exists(src_file):
    print(f"{SM_SLC_fname} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")


max_line=10500
min_line=8000
max_range=9500
min_range=6500
sub_array=array[min_line:max_line, min_range:max_range]
map_grid=np.array([(0,0), (0, max_line-min_line), 
                   (max_range-min_range, max_line-min_line), (max_range-min_range,0)])
SLC_grid =np.array([(min_line, min_range), (max_line, min_range), 
                    (max_line, max_range), (min_line, max_range)])


map_grid = np.array([(0,0), (0, max_line-min_line), (max_range-min_range, 
                     max_line-min_line), (max_range-min_range,0)])

SLC_grid = np.array([(min_line, min_range), (max_line, min_range), 
                    (max_line, max_range), (min_line, max_range)])

### map Point of interest
map_point = [1433,1362]
slc_pt    = np.round(extrapolate(map_point,map_grid, SLC_grid))
rec_xsize = 50
rec_ysize = 50
rec_xpos  = map_point[0] - rec_xsize//2
rec_ypos  = map_point[1] - rec_ysize//2
rect      = Rectangle((rec_xpos, rec_ypos), rec_xsize, rec_ysize, linewidth=2, edgecolor='yellow', facecolor='none')

plt.gca().add_patch(rect)
im=plt.imshow(abs(np.flip(np.flip(sub_array,1),1)), origin='lower', norm=colors.LogNorm(vmin=30, vmax=700))
# Show the plot
#plt.colorbar(im)
plt.show()

target_line  = slc_pt[0]
target_pixel = slc_pt[1]

grid_df = get_SM_GridPoints(annotation_file)
grids   =get_SM_ref_grids(grid_df, target_line, target_pixel)
ref_grid  = grids[0]
gps_grid  = grids[1]

point=[target_line, target_pixel]
ans=isPointInROI(point,ref_grid)
gps_location = extrapolate(point,ref_grid, gps_grid)
print(f' GPS location of SLC point {slc_pt} is : {gps_location}')
