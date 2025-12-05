
# Function to take 1 burst image and create binary mask for burst
# img                 - image of burst already extracted from full SLC image
# burst_boundaries    - [row,col] co-ordinates of burst vertices in full SLC image co-ordinates
# ship_bounding_boxes - [row,col] vertices of box around ship locations relative to full SLC image vertices
#
def generate_burst_binary_mask(img, burst_boundary, ship_bounding_boxes, ship_indices, chip_size):
    
    img_mask               = np.zeros(img.shape,dtype=np.uint8)
    img_rectangle_mask     = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask        = np.zeros(img.shape,dtype=np.uint8)
    ship_chip              = np.ones(chip_size, dtype= np.uint8)
    binary_ship_chip       = np.ones(chip_size, dtype= np.uint8)
    rectangle_mask         = np.zeros(chip_size, dtype= np.uint8)
    dilated_rectangle_mask = np.ones(chip_size, dtype= np.uint8)

    burst_start_row = burst_boundary[0][0]
    row_offset = round(bounding_box_border/2)
    col_offset = round(bounding_box_border/2)

    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])

    for j in range (0, ship.bounding_boxes.shape[0]):
        row     = ship_bounding_boxes[j][0][0] - burst_start_row -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = ship_bounding_boxes[j][0][1]                   -25
        h       = ship_bounding_boxes[j][1][0] - ship_bounding_boxes[j][0][0]
        w       = ship_bounding_boxes[j][3][1] - ship_bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)

        img_mask [row:row+h, column:column+w]                                  = roi
        ship_chip[row_offset:row_offset + h, col_offset:col_offset + w]        = roi
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = create_ship_binary_mask(roi)
        binary_img_mask[row:row+h, column:column+w]                            = create_ship_binary_mask(roi)
        if j==0:
            ship_chips_mask         = ship_chip
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction

        # Need to highlight positions of ships in a mask : use dilation and subtraction to generate
        # outline of boundary and place in mask image
        rectangle_mask  [row_offset:row_offset + h, col_offset:col_offset + w] = 1
        dilated_rectangle_mask                                                 = dilation(rectangle_mask, kernel)
        number_of_dilations = 4
        for j in range(0,number_of_dilations):
            dilated_rectangle_mask = dilation(dilated_rectangle_mask, kernel)
        dilated_rectangle_mask = dilated_rectangle_mask - rectangle_mask          # Create outline
        dilated_rectangle_mask = dilated_rectangle_mask * 1e4                     # Give outline high value
        dilated_rectangle_mask = dilated_rectangle_mask + 1                       # Make inner pixels =1 
        img_rectangle_mask[row - row_offset:row + row_offset + h ,column-col_offset:column + col_offset + w] = dilated_rectangle_mask

    # Multiply image mask with roi around ships by mask with bounding boxes. Should end up with original image with bounding box
    # around each ship
    img_rectangle_mask = img_rectangle_mask * img_mask

    return img_rectangle_mask, binary_img_mask, ship_chips_mask, binary_ship_chips_mask







##############################################








burst_list=[4]
img                    = []
nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)
azFraction =1
rangeFraction=1

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')

#with plt.style.context("dark_background"):
#    plt.figure(figsize=(20, 20))
#    plt.title(f"SLC IW  {SAFE_dir}")
mpl.rcParams['figure.figsize'] = [14.0, 8.0]
fig, ax = plt.subplots(3,1)

azimuth_offset = 0 
range_offset   = 0 
rec            = []
bounding_box_border = 25   # Should be clearance/2
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) # +1?
ship_chips_mask     = np.ones([bounding_box_dim + 2*bounding_box_border, bounding_box_dim + 2*bounding_box_border], dtype= np.uint8)
binary_ship_chips_mask =np.ones(ship_chips_mask.shape, dtype = np.uint8)
for burst_number in burst_list:
    # Get bounding boxes for ships only in this burst
    bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
    # Get burst boundary
    burst_boundary = burst_boundaries[burst_number-1]
    b_start        = burst_boundary[0][0]
    b_end          = burst_boundary[1][0]
    b_az_start     = round(b_start + (1-azFraction)*(b_end - b_start))
    b_az_end       = round(b_end   - (1-azFraction)*(b_end - b_start))
    b_range_start  = round(nRangeSamples/2 * (1-rangeFraction))
    b_range_end    = round(nRangeSamples/2 * (1+rangeFraction))
    # Extract burst image and copy to img 
    img.append(img_array[b_az_start:b_az_end, b_range_start:b_range_end])


    img = np.concatenate(img, axis=0)
    img = np.real(np.sqrt(img*np.conjugate(img)))
    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')
    #create a mask same size as burst image
    img_mask            = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask     = np.zeros(img.shape,dtype=np.uint8)
    ship_chip           = np.ones(ship_chips_mask.shape, dtype= np.uint8)
    binary_ship_chip    = np.ones(ship_chips_mask.shape, dtype= np.uint8)

    row_offset = round(bounding_box_border/2)
    col_offset = round(bounding_box_border/2)
    # Loop over each ship in burst and generate coordinates for bounding rectangle 
    # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
    # If plotting random bursts must subtract burst start position as offset for rectangles
    # Add end row of previous burst to position
    for j in range (0, bounding_boxes.shape[0]):
    #for j in range (0, 6):
        row     = bounding_boxes[j][0][0] - b_az_start -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = bounding_boxes[j][0][1]              -25
        h       = bounding_boxes[j][1][0] - bounding_boxes[j][0][0]
        w       = bounding_boxes[j][3][1] - bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)
        #roi     =img[row:row+h, column:column+w]
        
        img_mask[row:row+h, column:column+w]                                   = roi
        ship_chip       [row_offset:row_offset + h, col_offset:col_offset + w] = roi
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = create_ship_binary_mask(roi)
        binary_img_mask[row:row+h, column:column+w]                            = create_ship_binary_mask(roi)
        #mask4[row_offset:row_offset + h, col_offset:col_offset +w]      = binary_ship_mask
        if j==0:
            ship_chips_mask         = ship_chip
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction
        # If not do not plot them
        #if row in range(b_az_start,b_az_end) and column in range(b_range_start, b_range_end):
        # If plotting multiple bursts need to ensure ship bounding rectangles move with offset introduced by each successive burst 
    #img=img*img_mask
    

    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 700), aspect = '5')
    
    #plt.imshow(amp_img,norm=colors.LogNorm(vmin=10,vmax = 700),  aspect='15')
    #Use these with multiple burst processing 
    azimuth_offset = b_az_end - b_az_start
    range_offset   = 0 
    
ax[0].imshow(ship_chips_mask.T,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
ax[1].imshow(binary_ship_chips_mask.T,cmap='Greys_r', aspect = '1')
ax[2].imshow(binary_img_mask,cmap='Greys_r', aspect = '10')
#ax[3].imshow(np.real(np.sqrt(img_array*np.conjugate(img_array))),norm=colors.LogNorm(vmin=15,vmax = 800), aspect = '1')


plt.style.context("dark_background")
plt.tight_layout()
plt.xlabel("Range Samples")
plt.ylabel("Azimuth Lines)")
plt.show()

