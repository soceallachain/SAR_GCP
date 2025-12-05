#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:26:53 2025
Script to scan though /raw_public_chip_data and find all chips that are
wholly sea based and have no labelled ships.
Returns file with list of real chips
@author: joe
"""

import os
import numpy as np
import random


src_dir = '/home/joe/Misc/tempTest/Data/'
chip_list_file   = 'public_burst_chip_list.txt'
output_chip_list ='public_sea_only_chips.txt'

#Read chip_list_file
with open(f'{src_dir}{chip_list_file}', 'r') as file:
    chip_list = [line.strip() for line in file]

#Find all real and imaginary chips that are sea based
burst_list = [f for f in chip_list if 'True' in f and 'binary' not in f]

#Find bursts these chips belong to (chips are only drawn from sea based bursts)
search_string = '_chip'
burst_list = [f.split(search_string)[0] for f in burst_list if search_string in f]

#Find all chips in sea based bursts and remove chips with labelled ships
sea_chips  = [f2 for f2 in chip_list if any(f1 in f2 for f1 in burst_list) 
              and 'Ship' not in f2]



#Save list to file
with open(f'{src_dir}{output_chip_list}', 'w') as f:
    f.write('\n'.join(sea_chips))

real_sea_chips = [f for f in sea_chips if '_real_' in f]
random_real_sea_chips = random.sample(real_sea_chips, 300)
random_imag_sea_chips = [f.replace('_real_','_imag_') for f in random_real_sea_chips]

random_sea_chips =[]
random_sea_chips = random_real_sea_chips + random_imag_sea_chips

#Save list to file
with open(f'{src_dir}public_300_sea_chipnames.txt', 'w') as f:
    f.write('\n'.join(random_sea_chips))
   
