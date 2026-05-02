import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pa
import pandas as pd
import io
from PIL import Image
import csv
import os
import glob

# ===== FEATURE EXTRACTION FUNCTIONS ===== 
    
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

def sharpness(image):
    # Calculates the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def edge_density(image):
    # Find edges using the Canny algorithm
    edges = cv2.Canny(image, 100, 200)
    # Calculate what percentage of the image pixels are edges
    return np.sum(edges > 0) / edges.size

def high_frequency_content(image):
    # Perform a Fast Fourier Transform (FFT)
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Calculate the mean of the frequency magnitudes
    return np.mean(magnitude_spectrum)


# ================= START OF MAIN CODE =================

# DEFINING CSV FILE PATH
csvPath = 'extracted_training_features.csv'

# DEFINE COLUMNS OF CSV FILE
field_names = [
    'labelA',
    'labelB',
    'average_brightness',
    'average_contrast',
    'average_noise',
    'noise_deviation',
    'sharpness',
    'edge_density',
    'high_frequency'
]

# CREATE THE CSV FILE AND WRITE THE HEADER ONLY IF IT DOESN'T EXIST
if not os.path.isfile(csvPath):
    with open(csvPath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
    print(f"Created new file: {csvPath}")
else:
    print(f"File already exists, appending to: {csvPath}")

# GET ALL TRAINING FILES
trainFiles = glob.glob('Defactify_Image_Dataset/data/train-*.parquet')

# SORT THE TRAINING FILES BY NAME 
trainFiles.sort()

print('===== STARTING EXTRACTION OF TRAINING DATA =====')

for path in trainFiles:
    print(f"Starting extraction from file: {path}")
    
    # READ THE CURRENT TRAINING DATASET INTO PANDAS DATAFRAME
    table = pa.read_table(path)
    df = table.to_pandas()

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
            sharpnessVar = sharpness(grayImage)
            edgeDensity = edge_density(grayImage)
            highFrequency = high_frequency_content(grayImage)

            # DICT TO INSERT INTO CSV FILE
            rowData = {
                'labelA': photoLabelA,
                'labelB': photoLabelB,
                'average_brightness': avgBright,
                'average_contrast': avgContrast,
                'average_noise': avgNoise,
                'noise_deviation': avgNoiseDev,
                'sharpness': sharpnessVar,
                'edge_density': edgeDensity,
                'high_frequency': highFrequency
            }

            # OPEN THE FILE IN APPEND MODE AND WRITE THE ROW
            with open(csvPath, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                writer.writerow(rowData)
        
        except Exception as e:
            print(f"Error opening image: {e}")
    print(f'Finished extracting data from file: {path}')

print('===== FINISHED EXTRACTING TRAINING DATA STARTING EXTRACTION OF VALIDATION DATA =====')

# ================= START OF VALIDATION EXTRACTION CODE =================
# DEFINING CSV FILE PATH
csvPath = 'extracted_validation_features.csv'

# CREATE THE CSV FILE AND WRITE THE HEADER ONLY IF IT DOESN'T EXIST
if not os.path.isfile(csvPath):
    with open(csvPath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
    print(f"Created new file: {csvPath}")
else:
    print(f"File already exists, appending to: {csvPath}")

# GET ALL VALIDATION FILES
validationFiles = glob.glob('Defactify_Image_Dataset/data/validation-*.parquet')

# SORT THE VALIDATION FILES BY NAME 
validationFiles.sort()

for path in validationFiles:
    print(f"Starting extraction from file: {path}")
    
    # READ THE CURRENT VALIDATION DATASET INTO PANDAS DATAFRAME
    table = pa.read_table(path)
    df = table.to_pandas()

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
            sharpnessVar = sharpness(grayImage)
            edgeDensity = edge_density(grayImage)
            highFrequency = high_frequency_content(grayImage)

            # DICT TO INSERT INTO CSV FILE
            rowData = {
                'labelA': photoLabelA,
                'labelB': photoLabelB,
                'average_brightness': avgBright,
                'average_contrast': avgContrast,
                'average_noise': avgNoise,
                'noise_deviation': avgNoiseDev,
                'sharpness': sharpnessVar,
                'edge_density': edgeDensity,
                'high_frequency': highFrequency
            }

            # OPEN THE FILE IN APPEND MODE AND WRITE THE ROW
            with open(csvPath, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                writer.writerow(rowData)
        
        except Exception as e:
            print(f"Error opening image: {e}")
    print(f'Finished extracting data from file: {path}')
print('===== FINISHED EXTRACTING VALIDATION DATA STARTING EXTRACTION OF TESTING DATA =====')

# ================= START OF TEST EXTRACTION CODE =================
# DEFINING CSV FILE PATH
csvPath = 'extracted_test_features.csv'

# CREATE THE CSV FILE AND WRITE THE HEADER ONLY IF IT DOESN'T EXIST
if not os.path.isfile(csvPath):
    with open(csvPath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
    print(f"Created new file: {csvPath}")
else:
    print(f"File already exists, appending to: {csvPath}")

# GET ALL TEST FILES
testFiles = glob.glob('Defactify_Image_Dataset/data/test-*.parquet')

# SORT THE TEST FILES BY NAME 
testFiles.sort()

for path in testFiles:
    print(f"Starting extraction from file: {path}")
    
    # READ THE CURRENT TEST DATASET INTO PANDAS DATAFRAME
    table = pa.read_table(path)
    df = table.to_pandas()

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
            sharpnessVar = sharpness(grayImage)
            edgeDensity = edge_density(grayImage)
            highFrequency = high_frequency_content(grayImage)

            # DICT TO INSERT INTO CSV FILE
            rowData = {
                'labelA': photoLabelA,
                'labelB': photoLabelB,
                'average_brightness': avgBright,
                'average_contrast': avgContrast,
                'average_noise': avgNoise,
                'noise_deviation': avgNoiseDev,
                'sharpness': sharpnessVar,
                'edge_density': edgeDensity,
                'high_frequency': highFrequency
            }

            # OPEN THE FILE IN APPEND MODE AND WRITE THE ROW
            with open(csvPath, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                writer.writerow(rowData)
        
        except Exception as e:
            print(f"Error opening image: {e}")
    print(f'Finished extracting data from file: {path}')
print('===== FINISHED EXTRACTING ALL FILES :) =====')