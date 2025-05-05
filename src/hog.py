import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import cv2

class HOG:
    def __init__(self, image_file, cell_size=8, bins=9):
        with Image.open(image_file) as img:
            img = img.convert("L")
            self.image_array = np.array(img)
        
        self.cell_size = cell_size
        self.bins = bins

        self._gradient = None
        self._orientation = None
        self._histogram = None
        self._features = None

        self.I_x = np.array([
                             [1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]
                            ])
        
        self.I_y = np.array([
                             [1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]
                            ])

    @property
    def gradient(self):
        if self._gradient is None:
            self.compute_gradient()
        return self._gradient
    
    @property 
    def orientation(self):
        if self._orientation is None:
            self.compute_gradient()
        return self._orientation

    @property
    def histogram(self):
        if self._histogram is None:
            self.compute_histogram()
        return self._histogram
    
    @property
    def features(self):
        if self._features is None:
            self.extract_hog_features()
        return self._features

    def compute_gradient(self):
        img_array = self.image_array
        
        # Convolution to multiply sub matrices to I_x and I_y
        gx = convolve(img_array, self.I_x, mode='constant', cval=0.0)
        gy = convolve(img_array, self.I_y, mode='constant', cval=0.0)
        
        g_magnitude = np.sqrt(gx**2 + gy**2)

        # turn to angles and limit from 0-180
        g_orientation = np.arctan2(gy, gx) * 180 / np.pi
        g_orientation = np.mod(g_orientation, 180)

        self._gradient = g_magnitude
        self._orientation = g_orientation
        return g_magnitude, g_orientation

    def compute_histogram(self):
        height, width = self.gradient.shape
        
        cell_rows = height // self.cell_size
        cell_cols = width // self.cell_size
        
        # Initialize a 3D array to store histograms for each cell
        histograms = np.zeros((cell_rows, cell_cols, self.bins))

        # Flatten gradient and orientation arrays into cells
        for i in range(cell_rows):
            for j in range(cell_cols):
                # Extract cell magnitude and orientation

                cell_magnitude = self.gradient[
                    i * self.cell_size:(i + 1) * self.cell_size,
                    j * self.cell_size:(j + 1) * self.cell_size
                ]
                cell_orientation = self.orientation[
                    i * self.cell_size:(i + 1) * self.cell_size,
                    j * self.cell_size:(j + 1) * self.cell_size
                ]

                bin_indices = (cell_orientation // (180 // self.bins)).astype(int).flatten()
                magnitudes = cell_magnitude.flatten()

                np.add.at(histograms[i, j], bin_indices, magnitudes)

        self._histogram = histograms.sum(axis=(0, 1))
        return self._histogram
    
    def display_histogram(self):
        plt.bar(np.array(range(self.bins))+0.5, self.histogram, width=1.0, color='blue', alpha=0.7)
        plt.xlim(0, self.bins)
        plt.xlabel('Orientation Bins')
        plt.ylabel('Magnitude')
        plt.title('HOG Histogram')
        plt.show()
        plt.close()

    def extract_hog_features(self):
        hist = self.histogram

        max_bin_ratio = np.max(hist) / (np.sum(hist) + 1e-6)
        
        # + laplacian variance
        laplacian_variance = cv2.Laplacian(self.image_array, cv2.CV_64F).var()

        hog_features = [max_bin_ratio, laplacian_variance]

        self._features = {}

        self._features = {i:j for i,j in zip(["max_bin_ratio", "laplacian_variance"], hog_features)}

        return self._features