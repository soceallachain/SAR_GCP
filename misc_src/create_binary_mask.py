def create_ship_binary_mask(ROI):
    
    min_ROI    = np.min(ROI)
    max_ROI    = np.max(ROI)
    avg_ROI    = np.mean(ROI)
    stddev_ROI = np.std(ROI)
    kernel1    = np.array([[1,1,1],
                           [1,1,1],
                           [1,1,1]])
    #rrrd = dilation(rrrd, kernel)
    
    threshold = avg_ROI + (1.0*stddev_ROI)
    #threshold = threshold_otsu(ROI)
    ROI2 = np.copy(ROI)
    if max_ROI - avg_ROI < 8 : 
        ROI_eq = exposure.equalize_hist(ROI)
        eq1av = np.mean(ROI_eq)
        ZQ = ROI_eq.copy()
        
        #ROI_eq[ROI_eq <= 0.99]=0
        #ROI_eq[ROI_eq >  0.99]=1
        
        for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
            for c in range(ROI.shape[1]):  # Iterate through columns in the row
                if ROI_eq[r, c] > 0.99:
                    ROI_eq[r, c] = 1
                else:
                    ROI_eq[r, c] = 0
        ROI2 = ROI_eq 
        # apply erosion operation to remove noise, then dilate
        ROI2 = erosion(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        
    else:
        #AGREGAR THRESHOLD PARA CLAHE
        # Apply thresholding to create a binary mask
        #ROI[ROI <= threshold]=0
        #ROI[ROI >  threshold]=1
        for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
            for c in range(ROI.shape[1]):  # Iterate through columns in the row        
                if ROI[r, c] > threshold:
                    ROI2[r, c] = 1
                else:
                    ROI2[r, c] = 0
        
        ROI2 = erosion(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        #ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = erosion(ROI2, kernel1)
    return ROI2        


