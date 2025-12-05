from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from math import ceil
import os

# Settings
#image_dir = "/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/images/public_hc_sea_ships_no_nonships_gt100"
#image_dir = "/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/Train_gt100m"
image_dir = "/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_velocity_hypothesis/NonShip_gt100m"
#image_dir = "/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/images/public_m_and_hc_sea_ships_no_nonships_gt100"
output_document = "Train_NonShips_gt100m_images"

# Get list of image files either from directory listing 
# Sort by creation date - typically have generated images sequentially
files = os.listdir(image_dir)
files_sorted = sorted(files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
image_files  = [f for f in files_sorted if f.lower().endswith(".jpg")]

# ... or sort alphabetically with jpg extension
#image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])

#output_pdf = f'{image_dir}/sea_chip_images.pdf'
#output_pdf = f'{image_dir}/gt100m_HShips_chip_images.pdf'
# Or get list of image files from file_list
#file_list      = 'sea_only_no_ships_files_to_keep_list.txt'
#file_list_path = f'{image_dir}/{file_list}'
#with open(file_list_path, 'r') as file:
#    image_files = [line.strip() for line in file]



#Break into groups of num_images to prevent pdf generator from failing
num_images = 200
#Determine how many groups there are
num_full_groups = ceil(len(image_files) / num_images)
for j in range(0,num_full_groups):
    
    start = j*num_images +1
    if j==num_full_groups-1:
        end = num_images*j + len(image_files) % num_images
    else:
        end = num_images*(j+1)
    output_pdf = f'{image_dir}/{output_document}_{start}_{end}.pdf'
    # Create PDF canvas
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    width, height = landscape(letter)

    # --- Contents Page ---
    c.setFont("Helvetica-Bold", 10)
    c.drawString(1 * inch, height - 1 * inch, "Contents")

    c.setFont("Helvetica", 8)
    toc_y = height - 1.5 * inch
    print(f'Processing file names from file_list........')
    for i, img_name in enumerate(image_files[start-1:end]):
        display_name = os.path.splitext(img_name)[0]
        page_label = f"page_{i+1}"
    
        # Write TOC entry
        c.drawString(1.2 * inch, toc_y, f"{i+1}. {display_name}")
        c.linkAbsolute(display_name, page_label, Rect=(1.2*inch, toc_y-2, 6*inch, toc_y+10))
    
        toc_y -= 0.3 * inch
        if toc_y < 1 * inch:
            c.showPage()
            c.setFont("Helvetica", 8)  # Reset font after new page
            toc_y = height - 1 * inch

    c.showPage()

    print(f'Create TOC pages ....')
    # --- Image Pages ---
    for i, img_name in enumerate(image_files[start-1:end]):
        image_path = os.path.join(image_dir, img_name)
        display_name = os.path.splitext(img_name)[0]
        page_label = f"page_{i+1}"

        c.bookmarkPage(page_label)
        c.addOutlineEntry(display_name, page_label)

    # Fit image to page
        c.drawImage(image_path, 0, 0, width=width, height=height)
        c.showPage()

    # Save the PDF
    c.save()
print(f'***************** Completed document creation *********************')
