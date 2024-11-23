import cv2 as cv
import numpy as np
from Image_Algorithm import Algorithm

numberOfFiles = input('How many files do you want to process in the Input Directory?')

#Commented out but might be used later

# uploadFiles = input('Would you like to upload into the output directory? (y/n)')
# if uploadFiles == 'y':
#     uploadFiles = True
# else:
#     uploadFiles = False


#Directory names
inputDirectory = 'Input_Images/BW_'
outputDirectory = 'Output_Images/'
testDirectory = 'Test_Images/GT_'

def processor(num):
    #Outputs a list of scores after using an image algorithm on num number of files

    scoreList = []
    seq = generate_number_sequence(int(num))
    for i in seq:
        print(i)
        image = cv.imread(inputDirectory + i + '.png')
        output = Algorithm(image)
        test = cv.imread(testDirectory + i + '.png')
        scoreList.append(scorer(image, output, test)['Accuracy'])
    return scoreList
def scorer(input, output, test):
    # Obtains accuracy by dividing total number of pixels not matching test by total number of staff line pixels
    totalStaffLinePixels = np.sum(input != test)
    totalErrorPixels = np.sum(output != test)
    if totalErrorPixels != 0:
        score = {'StaffPixels': totalStaffLinePixels, 'ErrorPixels': totalErrorPixels, 'Accuracy': (1-totalStaffLinePixels/totalErrorPixels)*100}
    else:
        score = {'StaffPixels': totalStaffLinePixels, 'ErrorPixels': totalErrorPixels, 'Accuracy': np.float64(100)}
    # Returns dictionary of StaffPixels, ErrorPixels, and Accuracy score
    return score


def generate_number_sequence(n):
    # Formats int into 0001, 0002, etc for image access in directory
    sequence = [f"{i:04}" for i in range(1, n + 1)]
    return sequence


print(processor(numberOfFiles))






