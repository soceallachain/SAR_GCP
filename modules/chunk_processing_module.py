##############################################################################
##
##   Module for code to process chunks of Strip Map S1 data
##
##   Mainly to determine offset between Raw and SLC data in azimuth and range
###############################################################################
## Function to process number of files belonging to different swaths
## Continues processing until requisite number of each are processed
##

from .SAR_code_module import *
from .image_processing_module import *


def process_files_for_chunk_offsets(slc_ids, root_path, swaths,
                                    n_files_per_swath, pol, slice_size,
                                    plot_plots
                                    ):

    #Initialize swath file counters dictionary
    swath_file = {swath: 0 for swath in swaths}
    total_files_required = n_files_per_swath * len(swaths)
    swath_offsets = []

    for slc_id in slc_ids:
        for swath in swaths:
            # Exclude 2016 as seems to be corruption issue with <line>0 in 
            # some annotation files
            if swath in slc_id and '_2016' not in slc_id:
                if swath_file[swath] >= n_files_per_swath:
                    continue


                for pol in pols:
                    
                   files       = get_SM_files_from_slc_id(slc_id_path, pol)
                   if files != -1:
                       break  # Found files
                                
                if files == -1:
                    print(f'Warning: {slc_id}:  Files not found ')
                    continue
                else:
                    raw_slc_annot_files  = files
                    
                chunk_info = process_file_for_chunks(raw_slc_annot_files)

                if chunk_info == -1:
                    continue
                else:
                    offsets = process_chunks_for_offsets(chunk_info,
                                                         slice_size,
                                                         plot_plots
                                                         )
                    swath_file[swath]+=1
                    swath_offsets.append([swath, slc_id, offsets])
                    break # Move to the next slc_id once matched and processed

        if sum(swath_file.values()) == total_files_required:
            break


    return swath_offsets


##############################################################################
## Function to process a SM acquisition and return the raw and metadata for
## each chunk comprising teh acquisiton and the timing based offsets and the
## of the individual files of for the acquisition
##
def process_file_for_chunks(raw_slc_annot_files):
   from modules.SAR_code_module import (
        get_SM_raw_burst_info_v2,
        get_SM_raw_slc_offsets_v2,
        get_SM_slc_matched_chunk_dimensions,
        get_SM_slc_block_offsets

    )
    
   raw_file        = raw_slc_annot_files[0]
   annotation_file = raw_slc_annot_files[2]   
   parts =  raw_file.split('/')
   for part in parts:
       if '_SLC_' in part:
           break
   slc_id = part
   parts  = slc_id.split('_')
   slc_id_name =f'{parts[0]}_{parts[1]}_{parts[-1]}'

   raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
   number_of_chunks       = len(raw_burst_info[0])

   print(f'{slc_id} has {number_of_chunks} chunks : {raw_burst_info[0].index[:].to_list()}')

   raw_slc_offsets = get_SM_raw_slc_offsets_v2(raw_file, annotation_file)

   # Get chunk data and place in slice array
   chunk_list     = raw_burst_info[0].index[:].to_list()
   chunk_data     = []
   chunk_metadata = []
   raw_data       = []
   number_failed_chunks = 0
   for chunk in chunk_list:
       try:
           chunk_metadata.append(l0file.get_acquisition_chunk_metadata(chunk))
           chunk_data.append(l0file.get_acquisition_chunk_data(chunk))

           print(f'{slc_id} Chunk#{chunk} successfully read')
       except:
           print(f'{slc_id} Chunk#{chunk} failed!')
           chunk_metadata.append([])
           chunk_data.append([])
           number_failed_chunks+=1
           chunk_failed = True

   if number_failed_chunks == 0:
       for i in range(len(chunk_data)):
           az_start    = int(raw_slc_offsets[i][0])
           az_end      = int(raw_slc_offsets[i][1])
           range_start = int(raw_slc_offsets[i][2])
           range_end   = int(raw_slc_offsets[i][3])
           #raw_data.append(chunk_data[i][az_start:az_end, range_start:range_end])
   else:
       print(f'Warning : {slc_id_name} has {number_failed_chunks} failed chunk(s):'
             f' Ignoring and moving to next file'
             )
       return -1


   return chunk_data, chunk_metadata, raw_slc_offsets, raw_slc_annot_files, l0file


###############################################################################
##
## Input Args:
##            chunk_info:
##               chunk_data      - array of raw chunk data
##               slc_image       - array of full slc data
##               raw_slc_offsets - ofsets measured from start of raw chunk
##               files           - raw, slc and annotation file names + paths
##               l0file          - class for accessing raw and metadata
##            slice_size      - size of NxM slice to use for cross correlation
##
## Returns:
##            offsets          - array with top and bottom range and azimuth
##                               offsets for each chunk
##
def process_chunks_for_offsets(chunk_info, slice_size, plot_plots):
   from modules.SAR_code_module import (
        get_SM_raw_burst_info_v2,
        get_SM_raw_slc_offsets_v2,
        get_SM_slc_matched_chunk_dimensions,
        get_SM_slc_block_offsets

    )


   chunk_data, meta_data, raw_slc_offsets, files, l0file = chunk_info
   slc_file = files[1]
   # Use context manager to properly manage rasterio handles
   # Note slc_image is larger than individual raw chunks
   with rasterio.open(slc_file) as slc_data:
       slc_image = slc_data.read(1)

   n_slc_az_lines      = slc_image.shape[0]
   n_slc_range_samples = slc_image.shape[1]
   n_raw_az_lines      = sum(chunk_data[i].shape[0] for i in range(len(chunk_data)))
   offsets             = []
   slc_chunk_offsets   = get_SM_slc_matched_chunk_dimensions(raw_slc_offsets)

   for chunk in range(len(chunk_data)):
       slc_data = slc_image[slc_chunk_offsets[chunk][0]:
                            slc_chunk_offsets[chunk][1],:
                            ]
       # chunk_offsets[0] & [1] have bottom & top offsets
       # chunk_offsets[0][0] is bottom az offset, [1][0] - top
       chunk_offsets = get_chunk_top_bottom_offsetsII(chunk_data[chunk], 
                                                          slc_data, 
                                                          slice_size, 
                                                          meta_data[chunk], 
                                                          raw_slc_offsets[chunk],
                                                          l0file,
                                                          plot_plots
                                                          )
       # Only first chunk incurs azimuth offset error
       # Other chunks should have zero azimuth offset error due to continuous
       # raw and SLC azimuth lines
       # Convert tuples to lists so elements can be modified
       bot_off = list(chunk_offsets[0])
       top_off = list(chunk_offsets[1])

       if chunk == 0:
           offsets.append([bot_off, top_off, chunk_offsets[2]])
       else:
           # Set bottom & top az offsets to zero
           bot_off[0] = 0
           top_off[0] = 0
           offsets.append([bot_off, top_off, chunk_offsets[2]])

   return offsets


"""
def process_files_for_chunk_offsets(slc_ids, root_path, swaths, n_files_per_swath, pol='VV'):
    # Initialize file counter dictionary for each swath name
    swath_file = {swath: 0 for swath in swaths}
    total_required_files = n_files_per_swath * len(swaths)

    for slc_id in slc_ids:
        # Check if all swaths have met their file quotas
        if sum(swath_file.values()) == total_required_files:
            break

        for swath in swaths:
            if swath in slc_id:
                # Skip if this swath quota is already satisfied
                if swath_file[swath] >= n_files_per_swath:
                    continue

                # Process chunk offsets
                chunk_info = process_file_for_chunks(slc_id, root_path, pol)

                # Check for processing failure
                if chunk_info == -1:
                    continue
                else:
                    swath_file[swath] += 1
                    break  # Move to the next slc_id once matched and processed

    return swath_file


"""

###############################################################################
## Function to determine offset at start and end of a single raw chunk
## Used to verify that offset is constant within a chunk
##
## Input Args:
##             chunk      - 2D array of complex whole chunk data
##             slc_data   - 2D array of slc data matched to chunk size
##             slice_size - size of NxM slice to use for cross correlation
## Returns:
##             bottom_offset - array with azimuth and range offset near start
##             top_offset    - array with azimuth and range offset near end
def get_chunk_top_bottom_offsets(chunk, slc_data, slice_size, meta_data,
                                 chunk_raw_slc_offsets, l0file, plot_plots
                                 ):

   # Offset near start of chunk
   #chunk_slice   =  chunk[0:slice_size[0], 0:slice_size[1]-1]
   #chunk_offsets = [0, slice_size[1], 0, slice_size[0]] 
   #focused_chunk = focus_SM_burst(meta_data, chunk_slice, chunk_offsets,
   #                               l0file, 'rma'
   #                                )
   #focused_chunk_slice = intensity(focused_chunk)                                          

   #slc_slice           = intensity(slc_data[0:slice_size[0], 0:slice_size[1]-1])
   #bottom_offsets      = compute_image_offset(focused_chunk_slice,
   #                                           slc_slice, plot_surface=True
   #                                           )


   # Ensure position compatible wih slice_size and slc chunk size
   position_shift = [0,0]   # Start of chunk
   position_shift = np.minimum(position_shift,
                            np.array(slc_data.shape) - slice_size
                            )
   slice_az_start = chunk_raw_slc_offsets[0] + position_shift[0]
   slice_az_end   = chunk_raw_slc_offsets[0] + position_shift[0] + slice_size[0]
   slice_rng_start= chunk_raw_slc_offsets[2] + position_shift[1]
   slice_rng_end  = chunk_raw_slc_offsets[2] + position_shift[1] + slice_size[1]

   chunk_slice    =  chunk[slice_az_start:slice_az_end,
                                    slice_rng_start:slice_rng_end
                                   ]
   # Note: chunk_offsets should be [rng_start, rng_end, az_start, az_end] unlike
   # other offset variables which are [az_start, az_end, rng_start, rng_end]
   chunk_offsets = [slice_rng_start, slice_size[0],
                    slice_az_start, slice_size[1]
                    ]
   focused_chunk = focus_SM_burst(meta_data, chunk_slice, chunk_offsets,
                               l0file, 'rma'
                                )
   focused_chunk_slice = intensity(focused_chunk)
   # plot raw and slc slices using for NCC
   slc_slice      = intensity(slc_data[position_shift[0]:
                                       slice_size[0] + position_shift[0] - 1,
                                       position_shift[1]:
                                       slice_size[1] + position_shift[1] - 1
                                       ]
                                        )
   bottom_offsets = compute_image_offset(focused_chunk_slice,
                                         slc_slice, plot_plots
                                         )
   if plot_plots:
      plot_2d_array(focused_chunk_slice,title='Bottom RDA Focused Slice')
      plot_2d_array(slc_slice,title='Bottom SLC Slice')

   # Offset near end of chunk
   # raw_slc_offsets[chunk][1] is the final azimuth line of the chunk
   # position_shift =[raw_slc_offsets[chunk][1] - 2*slice_size[0], 0]
   # Check end of range
   position_shift = [chunk_raw_slc_offsets[1] - 2*slice_size[0],
                     chunk_raw_slc_offsets[3] -   slice_size[1] - 1
                     ]
   # Ensure position compatible wih slice_size and slc chunk size
   position_shift = np.minimum(position_shift,
                               np.array(slc_data.shape) - slice_size
                               )
   slice_az_start = chunk_raw_slc_offsets[0] + position_shift[0]
   slice_az_end   = chunk_raw_slc_offsets[0] + position_shift[0] + slice_size[0]
   slice_rng_start= chunk_raw_slc_offsets[2] + position_shift[1]
   slice_rng_end  = chunk_raw_slc_offsets[2] + position_shift[1] + slice_size[1]

   chunk_slice    =  chunk[slice_az_start:slice_az_end,
                           slice_rng_start:slice_rng_end
                           ]
   #chunk_offsets = [0, slice_size[1], 0, slice_size[0]]
   chunk_offsets = [slice_rng_start, slice_size[0],
                    slice_az_start,  slice_size[1]
                    ]

   focused_chunk = focus_SM_burst(meta_data, chunk_slice, chunk_offsets,
                               l0file, 'rma'
                                )

   # Detect slc and focused raw data
   focused_chunk_slice = intensity(focused_chunk)

   slc_az_start_idx  = slice_az_start  - chunk_raw_slc_offsets[0]
   slc_az_end_idx    = slice_az_end    - chunk_raw_slc_offsets[0]
   slc_rng_start_idx = slice_rng_start - chunk_raw_slc_offsets[2]
   slc_rng_end_idx   = slice_rng_end   - chunk_raw_slc_offsets[2]
   slc_slice         = intensity(slc_data[slc_az_start_idx : slc_az_end_idx,
                                      slc_rng_start_idx: slc_rng_end_idx
                                      ])
   if plot_plots:
      plot_2d_array(focused_chunk_slice,title='Top RDA Focused Slice')
      plot_2d_array(slc_slice,title='Top SLC Slice')

   top_offsets   = compute_image_offset(focused_chunk_slice,
                                     slc_slice, plot_plots
                                     )
   return bottom_offsets, top_offsets

###############################################################################
#
# Function to find offset between SLC and RAW using NCC at specific point
# Can then be used twice in get_chunk_top_bottom_offsets
def get_chunk_offset_at_position(position, chunk, slc_data, slice_size, meta_data,
                                 chunk_raw_slc_offsets, l0file, plot_plots
                                 ):

   # Ensure position compatible wih slice_size and slc chunk size
   position       = np.minimum(position, np.array(slc_data.shape) - slice_size)
   slice_az_start = chunk_raw_slc_offsets[0] + position[0]
   slice_az_end   = chunk_raw_slc_offsets[0] + position[0] + slice_size[0]
   slice_rng_start= chunk_raw_slc_offsets[2] + position[1]
   slice_rng_end  = chunk_raw_slc_offsets[2] + position[1] + slice_size[1]

   chunk_slice    =  chunk[slice_az_start:slice_az_end,
                           slice_rng_start:slice_rng_end
                           ]
   # Note: chunk_offsets should be [rng_start, rng_end, az_start, az_end] unlike
   # other offset variables which are [az_start, az_end, rng_start, rng_end]
   chunk_offsets = [slice_rng_start, slice_size[0],
                    slice_az_start,  slice_size[1]
                    ]
   focused_chunk = focus_SM_burst(meta_data, chunk_slice, chunk_offsets,
                                  l0file, 'rma'
                                  )
   # Detect slc and focused raw data
   focused_chunk_slice = intensity(focused_chunk)
   
   #slc_az_start_idx  = slice_az_start  - chunk_raw_slc_offsets[0]
   #slc_az_end_idx    = slice_az_end    - chunk_raw_slc_offsets[0]
   #slc_rng_start_idx = slice_rng_start - chunk_raw_slc_offsets[2]
   #slc_rng_end_idx   = slice_rng_end   - chunk_raw_slc_offsets[2]
   #slc_slice         = intensity(slc_data[slc_az_start_idx : slc_az_end_idx,
   #                                       slc_rng_start_idx: slc_rng_end_idx
   #                                       ])
   # Can replace 5 lines above with single line:
   slc_slice          = intensity(slc_data[position[0]:position[0] + slice_size[0],
                                           position[1]:position[1] + slice_size[1]
                                           ])
   if plot_plots :
      plot_2d_array(focused_chunk_slice,title='Top RDA Focused Slice')
      plot_2d_array(slc_slice,title='Top SLC Slice')

   offsets   = compute_image_offset(focused_chunk_slice,
                                     slc_slice, plot_plots
                                     )
   return offsets



###############################################################################
def get_chunk_top_bottom_offsetsII(chunk, slc_data, slice_size, meta_data,
                                   chunk_raw_slc_offsets, l0file, plot_plots
                                   ):
   bottom_position = [0,0]
   bottom_offsets  = get_chunk_offset_at_position(bottom_position, chunk,
                                                slc_data, slice_size, meta_data,
                                                chunk_raw_slc_offsets, l0file,
                                                plot_plots
                                                )
   top_position = [chunk_raw_slc_offsets[1] - 2*slice_size[0],
                   chunk_raw_slc_offsets[3] -   slice_size[1] - 1
                   ]
   top_offsets  = get_chunk_offset_at_position(top_position, chunk, slc_data, 
                                                slice_size, meta_data,
                                                chunk_raw_slc_offsets, l0file,
                                                plot_plots
                                                )
   

   # Get SWST, f & TXPL for each chunk
   SWST                 = meta_data['SWST'].unique()[0]
   SWL                  = meta_data['SWL'].unique()[0]
   pulse_length         = meta_data['Tx Pulse Length'].unique()[0]
   RGDEC                = meta_data["Range Decimation"].unique()[0]
   range_sample_freq    = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
   #chirp_filter_length = int(np.round(pulse_length * range_sample_freq))
   
   parameters =[SWST, SWL, pulse_length, range_sample_freq, 
                chunk_raw_slc_offsets[0], chunk_raw_slc_offsets[2]
                ]


   return bottom_offsets, top_offsets, parameters





##############################################################################
##
def print_chunk_slc_offsets(offsets):
   # Header string formatted with fixed column widths
   print(f"{'Chunk':<12} | {'Bottom (Az, Range)':<22} | {'Top (Az, Range)':<22}")
   print("-" * 62)

   for i in range(len(offsets)):
       # Format each inner list/tuple into a uniform width string
       bot_str = f"[{offsets[i][0][0]}, {offsets[i][0][1]}]"
       top_str = f"[{offsets[i][1][0]}, {offsets[i][1][1]}]"

       print(f"Chunk#{i:<5} | {bot_str:<22} | {top_str:<22}")






def get_slc_matched_chunk_data(chunk_data, raw_slc_offsets):

    for i in range(len(chunk_data)):
        az_start    = int(raw_slc_offsets[i][0])
        az_end      = int(raw_slc_offsets[i][1])
        range_start = int(raw_slc_offsets[i][2])
        range_end   = int(raw_slc_offsets[i][3])
        raw_data.append(chunk_data[i][az_start:az_end, range_start:range_end])

    return raw_data



