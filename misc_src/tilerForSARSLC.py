################################################################################
# Script to read in a file produced by processing SLC data and output tiles of
# size 512x3072
# Tiler accepts filename of file to tile from, tile size in row (y) and column(x)
# directions.
# Typically input files will be burst data either of the real, imaginary or binary
# variety

# Method: Read in list of filenames, do some sanity check on files to ensure 
# correct size. Input filename should be :
# <SLC_safe_id_Real/Imag/Bin> and to which should be appended a suffix of the form
# suffix = print(f'IW{swath_of_interest}_burst#{burst_number)_n{i}.txt or
# suffix = print(f'IW{swath_of_interest}_burst#{burst_number)_r{i}.txt 
# where the 'n' denotes non-ovelapping data(more or less)  and 'r' denotes 
# randomly cut from data.


def SAR_data_tiler(filename, swath,  burst, number_random_tiles)

   load(filename)
   data=getData(filename)a
   burst = <>
   swath = <>
   data_rows=data.shape[0]
   data_cols=date.shape[1]
   n_row_tiles = div(data_rows, tile_row_size_) + 1
   n_col_tiles = div(data_rows, tile_cole_size)

   for i,j in range(0,n_row_tiles;0,n_col_tiles):
       tile = data[i*tile_row_size:tile_row_size*(i+1) -1, j*tile_col_size:tile_col_size*(j+1) -1]
       suffix = print(f'IW{swath_of_interest}_burst#{burst}_n{i}.dat')
       output_filename = filename + suffix
       np.savetxt(output_filename, tile, delimiter = ',')

   for j in range(0,n_col_tiles):
       tile= data[max_row - tile_row_size :max_row, j*tile_col_size:tile_col_size*(j+1)] 
       suffix = print(f'IW{swath_of_interest}_burst#{burst}_n{i}.dat')
       output_filename = filename + suffix
       np.savetxt(output_filename, tile, delimiter = ',')

   for i in range(0,n_row_tiles):
       tile= data[max_row - tile_row_size :max_row, (i-1)*tile_col_size:tile_col_size*j] 
       suffix = print(f'IW{swath_of_interest}_burst#{burst}_n{i}.dat')
       output_filename = filename + suffix
       np.savetxt(output_filename, tile, delimiter = ',')



   for j in range(0,number_random_tiles):

       np.random(row,col,0 to data_rows-tile_row_size,0 to data_cols - tile_col_size))
       tile = data[row:row+tile_row_size,col:col+tile_col_size]
       suffix = print(f'IW{swath_of_interest}_burst#{burst}_r{i}.dat')
       output_filename = filename + suffix
       np.savetxt(output_filename, tile, delimiter = ',')


