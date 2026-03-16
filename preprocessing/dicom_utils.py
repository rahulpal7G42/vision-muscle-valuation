import pydicom
import numpy as np
from skimage import exposure

class DicomProcessor:
    """Utilities for loading and normalizing medical DICOM images for model inference."""
    def load_dicom(self, path):
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array.astype(float)
        
        # Normalize to [0, 1]
        rescaled = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        
        # Contrast Enhancement
        enhanced = exposure.equalize_adapthist(rescaled)
        return enhanced

    def prepare_batch(self, paths):
        batch = [self.load_dicom(p) for p in paths]
        return np.expand_dims(np.stack(batch), axis=-1)
