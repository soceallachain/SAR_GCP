def get_chips_and_masks(img_array, burstList, burst_boundaries, ship_bounding_boxes, ship_burst_indices, chip_border_size):

    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')
    

    img                    = []
    burst_bounding_boxes   = np.array(ship_bounding_boxes)
    bounding_box_row_size  = burst_bounding_boxes[0,1,0]- burst_bounding_boxes[0,0,0]
    chip_row_size          = bounding_box_row_size + 2*chip_border_size
    bounding_box_col_size  = burst_bounding_boxes[0,3,1]- burst_bounding_boxes[0,0,1]
    chip_col_size          = bounding_box_size + 2*chip_border_size
    # Need to know the maximum number of ships per burst to set size of chips_from_all_bursts
    # to ensure successive bursts can be appended. Determine form incidence of bursts in 
    # ship_burst_indices     Note: need to import collections
    max_ships_per_burst = collections.Counter(ship_burst_indices).most_common(1)[0][1]
    chips_from_all_bursts = np.zeros(chip_size, chip_size*max_ships_pr_burst)
    chips_from_one_burst  = np.zeros(chip_size, chip_size*max_ships_pr_burst)

    for burst_number in burstList:
        # Get bounding boxes for ships only in this burst
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        # Get burst boundary
        burst_boundary = burst_boundaries[burst_number-1]
        print(f' In plotBurstsIII   ...... burst boundary is : {burst_boundary}')
        print(f' Number of ships in this burst #{burst_number} is {bounding_boxes.shape}')
        az_start       = burst_boundary[0][0]
        az_end         = burst_boundary[1][0]
        range_start    = burst_boundary[0][1]
        range_end      = burst_boundary[3][1]
        # Extract burst image and copy to img
        img.append(img_array[az_start:az_end, range_start:range_end])
        # Now save real and imaginary components of SLC file for this burst
        numpy.savetxt(f"{fname}_real_burst#{burst_number}.csv", np.real(img), delimiter = ",")
        numpy.savetxt(f"{fname}_imag_burst#{burst_number}.csv", np.imag(img), delimiter = ",")
        img = np.concatenate(img, axis=0)
        img = np.real(np.sqrt(img*np.conjugate(img)))
        mask = np.zeros(img.shape, dtype=np.uint8)
        # Loop over each ship in burst and generate coordinates for bounding rectangle 
        # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
        # If plotting random bursts must subtract burst start position as offset for rectangles
        # Add end row of previous burst to position

        ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(img_array[az_start:az_end, range_start:range_end], burst_boundary, bounding_boxes, chip_border_size)
        numpy.savetxt(numpy.savetxt(f"{fname}_real_burst#{burst_number}.csv", binary_image_ground_truth, delimiter = ",")
        
        # Determine size of ship_chips array and then assign it to a space of the same size in 
        # chips_from_one_burst
        ship_chips_transposed                                             = ship_chips.T 
        ship_chips_row_size                                               = ship_chips_transposed.shape[0]
        ship_chips_col_size                                               = ship_chips_transposed.shape[1]
        chips_from_one_burst                                              = chips_from_one_burst * 0
        chips_from_one_burst[0:ship_chips_row_size,0:ship_chips_col_size] = ship_chips_transposed
        chips_from_all_bursts.append(chips_from_one_burst)
        ship_chips_transposed                                             = binary_ship_chips.T
        chips_from_one_burst[0:ship_chips_row_size,0:ship_chips_col_size] = ship_chips_transposed
        chips_from_one_burst                                              = binary_ship_chips.T
        chips_from_all_bursts.append(chips_from_one_burst)


    fig, ax = plt.subplots()
    ax.imshow(chips_from_all_bursts,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
    plt.show()

    fig1, ax1 = plt.subplots(2,1,2)
    ax1[1].imshow(binary_image_ground_truth,cmap='Greys_r', aspect = '10')
    ax1[0].imshow(ship_loc_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
    plt.show()
    return chips_from_all_bursts, ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips

