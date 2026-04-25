import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pa
import pandas as pd
import io
from PIL import Image
import csv
import os

# --- DEFINING CSV FILE PATH ---
csvPath = 'extracted_photo_features.csv'

# DEFINE COLUMNS OF CSV FILE
field_names = [
    'labelA',
    'labelB',
    'average_brightness',
    'average_contrast',
    'average_noise',
    'noise_deviation'
]

# CREATE THE CSV FILE AND WRITE THE HEADER ONLY IF IT DOESN'T EXIST
if not os.path.isfile(csvPath):
    with open(csvPath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
    print(f"Created new file: {csvPath}")
else:
    print(f"File already exists, appending to: {csvPath}")

def brightness(image):
  # Use numpy to calculate mean pixel value in image
  avgBrightness = np.mean(image)
  return avgBrightness

def contrast(image):
  # Use std() function to calculate standard deviation of image
  contrast = image.std()
  return contrast

def noise(image):
  # Apply Gaussian blur to the grayscale image
  blurredImage = cv2.GaussianBlur(image, (5, 5), 0)

  # Calculate the noise by subtracting the blurred image from the original grayscale image
  noise = image - blurredImage

  # Calculate the mean and standard deviation of the noise
  # Higher mean noise value indicates more noise in the image
  # Higher standard deviation indicates more variable and less predictable noise
  mean_noise = np.mean(noise)
  std_noise = np.std(noise)

  return mean_noise, std_noise

table = pa.read_table('Defactify_Image_Dataset/data/train-00000-of-00007.parquet')
#print(table) # UNCOMMENT TO SEE FULL TABLE
df = table.to_pandas()
#print(df.head().T) # UNCOMMENT TO SEE SIMPLIFIED TABLE

# LOOP OVER DATASET GETTING EACH ROW
for index, row in df.iterrows():
    # UNCOMMENT TO SEE HOW THE IMAGES ARE STORED
    #print(row['Image'])

    photoLabelA = row['Label_A']
    photoLabelB = row['Label_B']

    # CONVERT BYTES TO IMAGE LIKE OBJECT
    image_stream = io.BytesIO(row['Image']['bytes'])

    # PROCESSING IMAGES
    try:
      # OPEN IMAGE
      image = Image.open(image_stream)
    
      """
      TEST IMAGE PROCESSING CODE
      print(f"Image format: {image.format}")
      print(f"Image size: {image.size}")
      print(f"Image mode: {image.mode}")

      # DO NOT USE =========================
      # image.show()
      # DO NOT USE =========================
      """
      # CONVERT PIL IMAGE TO NUMPY ARRAY
      image = np.array(image) 

      # CONVERT FROM RGB TO BGR
      cvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

      # CONVERT FROM COLOR TO GRAYSCALE
      grayImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)

      # GET PHOTO ATTRIBUTES
      avgBright = brightness(grayImage)
      avgContrast = contrast(grayImage)
      avgNoise, avgNoiseDev = noise(grayImage)

      # DICT TO INSERT INTO CSV FILE
      rowData = {
        'labelA': photoLabelA,
        'labelB': photoLabelB,
        'average_brightness': avgBright,
        'average_contrast': avgContrast,
        'average_noise': avgNoise,
        'noise_deviation': avgNoiseDev
      }

      # OPEN THE FILE IN APPEND MODE AND WRITE THE ROW
      with open(csvPath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writerow(rowData)
    
    except Exception as e:
      print(f"Error opening image: {e}")