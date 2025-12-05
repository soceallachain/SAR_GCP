min_ROI = np.min(ROI)
        max_ROI = np.max(ROI)
        avg_ROI =np.mean(ROI)
        stddev_ROI = np.std(ROI)
        
        aas += 1
        #print(f' count:{aas}, row top left: {row}, column top left:{column}')
        #print(f"Image Min  = {min_image}")
        #print(f"Image Max  = {max_image}")
        #print(f"Image Mean = {avg_image}")
        #print(f"Image StdDev = {stddev_image}")
        
       
        
        #threshold = avg_ROI + (2* stddev_ROI)
        threshold = threshold_otsu(ROI)
        
        ROI2 = np.copy(ROI)
        
        if max_ROI - avg_ROI < 8 :
            ROI_eq = exposure.equalize_hist(ROI)
            eq1av = np.mean(ROI_eq)
            ZQ = ROI_eq.copy()
                
            for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
                for c in range(ROI.shape[1]):  # Iterate through columns in the row
                        
                    if ROI_eq[r, c] > 0.99:
                        ROI_eq[r, c] = 1
                    else:
                        ROI_eq[r, c] = 0
            mask[row-4:row+3, column-4:column+3] = ROI_eq
            
        else:
        
            #AGREGAR THRESHOLD PARA CLAHE
            
            # Apply thresholding to create a binary mask
            for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
                for c in range(ROI.shape[1]):  # Iterate through columns in the row
                        
                    if ROI[r, c] > threshold:
                        ROI[r, c] = 1
                    else:
                        ROI[r, c] = 0
            
            # Update the mask with the binary ROI
            mask[row-4:row+3, column-4:column+3] = ROI
