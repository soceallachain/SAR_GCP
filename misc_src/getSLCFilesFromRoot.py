############################################################
# Function to find and return the SLC tiff and annotation
# files for a given product
# Must provide the root directory of the data and the swath
# of interest
#
# Note: Remember to import glob

def getSLCFilesFromRoot(path, swath_number):
    fpath   = f'{path}/SLC/'
    SafeDir = os.listdir(fpath)
    fpath   = f'{path}/SLC/{SafeDir[0]}/measurement/'
    src     = glob.glob(f'{fpath}*iw{swath_number}*vh*')[0]
    fpath   = f'{path}/SLC/{SafeDir[0]}/annotation/'
    annotation_file = glob.glob(f'{annotation_file_path}*iw{swath_number}*vh*')[0]
    return src, annotation_file
