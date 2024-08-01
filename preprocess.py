import argparse
import datetime
import os

from lungmask import LMInferer
import nrrd
import numpy as np
import pandas as pd
from pydicom import dcmread
from scipy.ndimage import zoom, binary_fill_holes
import SimpleITK as sitk
from skimage import measure


class Processor:
    """Processes original dicom files in several steps including interpolation, alignment and lung cropping.
    The processed images will be saved in .nrrd format to the location /<outpath>/<Subject ID>_<Modality>.nrrd
    Args:
        df (pandas.DataFrame): Dataframe containing the metadata with at least following columns:
            'Subject ID', 'Modality', and 'File Location'. 'File Location' should be relative to datapath.
        datapath (str): Path to the directory containing the original dicom files
        outpath (str): Path to the directory where the processed images will be saved
        target_spacing (tuple): Desired spacing between two voxels
        """

    def __init__(self, df, datapath, outpath, target_spacing=(1.0, 1.0, 1.0)):
        self.df = df
        self.datapath = datapath
        self.outpath = outpath
        self.target_spacing = tuple(float(t) for t in target_spacing)

    def __call__(self):

        for pid in self.df['Subject ID'].unique():

            if os.path.exists(os.path.join(self.outpath, pid+'_CT.nrrd')) and os.path.exists(os.path.join(self.outpath, pid+'_PT.nrrd')):
                continue

            print(f"{pid} is being processed...")

            ct_path = os.path.join(self.datapath, self.df[(self.df['Subject ID'] == pid)
                                                          & (self.df['Modality'] == 'CT')]['File Location'].item())
            pt_path = os.path.join(self.datapath, self.df[(self.df['Subject ID'] == pid)
                                                          & (self.df['Modality'] == 'PT')]['File Location'].item())

            ct_array, ct_origin, ct_spacing = self._read_volume(ct_path)
            pt_array, pt_origin, pt_spacing = self._read_volume(pt_path)

            # Interpolate images to make them have same spacing
            ct_resampled = zoom(ct_array, (ct_spacing[2] / self.target_spacing[2],
                                           ct_spacing[0] / self.target_spacing[0],
                                           ct_spacing[1] / self.target_spacing[1]), order=1)  # Linear
            pt_resampled = zoom(pt_array, (pt_spacing[2] / self.target_spacing[2],
                                           pt_spacing[0] / self.target_spacing[0],
                                           pt_spacing[1] / self.target_spacing[1]), order=1)  # Linear

            # Align PET and CTs since the boundary coordinates are different
            ct_aligned, pt_aligned = self._align(ct_resampled, pt_resampled, ct_origin, pt_origin)

            # Segment lungs
            inferer = LMInferer()
            mask = inferer.apply(ct_aligned)  # Returns a mask with different labels for different parts of the lung
            mask = np.where(mask == 0, 0, 1)  # Unify the labels

            # Clear artifacts
            mask_resampled = self._remove_artifacts(mask)

            # Apply the masks
            ct_mask_applied = np.where(mask_resampled == 1, ct_aligned, np.min(ct_aligned))
            pt_mask_applied = np.where(mask_resampled == 1, pt_aligned, np.min(pt_aligned))

            # Crop the lungs
            nonzeros = np.nonzero(mask_resampled)
            ct_cropped = ct_mask_applied[nonzeros[0].min(): nonzeros[0].max(),
                                         nonzeros[1].min(): nonzeros[1].max(),
                                         nonzeros[2].min(): nonzeros[2].max()]
            pt_cropped = pt_mask_applied[nonzeros[0].min(): nonzeros[0].max(),
                                         nonzeros[1].min(): nonzeros[1].max(),
                                         nonzeros[2].min(): nonzeros[2].max()]

            # Make sure that scans have the same shape, so they are aligned correctly
            assert ct_cropped.shape == pt_cropped.shape, 'CT and PT shapes should be equal!'

            # Save the files
            nrrd.write(os.path.join(self.outpath, f'{pid}_CT.nrrd'), ct_cropped)
            nrrd.write(os.path.join(self.outpath, f'{pid}_PT.nrrd'), pt_cropped)

    def _read_volume(self, path):
        """Helper function to read volumes"""

        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        img_array = sitk.GetArrayFromImage(image)

        photometric_interpretation = reader.GetMetaData(slice=1, key='0028|0004').strip()
        if photometric_interpretation == 'MONOCHROME1':
            img_array = img_array * -1 + np.max(img_array)

        # SUV conversion for PET images. Hounsfield conversion is applied as default to CT images.
        if reader.GetMetaData(slice=1, key='0008|0060').strip() == 'PT':
            img_array = self._suv_conversion(img_array, dicom_names[0])

        return img_array, origin, spacing

    def _suv_conversion(self, img, ds_path):
        """Converts original PET pixel intensities into Standard Uptake Values (SUV)
        Reference: https://gist.github.com/pangyuteng/c6a075ba9aa00bb750468c30f13fc603"""

        ds = dcmread(ds_path)
        try:
            weight = float(ds.PatientWeight) * 1000
        except:
            print("Patient weight is not available. Estimated value (75 kg) is used.")
            weight = 75000
        injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        scan_time = datetime.datetime.strptime(ds.SeriesTime.split('.')[0], '%H%M%S')
        injection_time = datetime.datetime.strptime(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime.split('.')[0], '%H%M%S')
        half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        decay_ratio = np.exp(-np.log(2)*(scan_time - injection_time).seconds/half_life)
        current_dose = injected_dose * decay_ratio
        suv_img = img / (current_dose / weight)

        return suv_img

    def _align(self, ct_img, pt_img, ct_origin, pt_origin):
        """Crops the CT and PT images to have the same border coordinates."""

        # Calculate the bottom right corner coordinates
        ct_end = (ct_origin[0] + ct_img.shape[1] * self.target_spacing[0],
                  ct_origin[1] + ct_img.shape[2] * self.target_spacing[1],
                  ct_origin[2] + ct_img.shape[0] * self.target_spacing[2])

        pt_end = (pt_origin[0] + pt_img.shape[1] * self.target_spacing[0],
                  pt_origin[1] + pt_img.shape[2] * self.target_spacing[1],
                  pt_origin[2] + pt_img.shape[0] * self.target_spacing[2])

        # Find the intersection points of upper left and bottom right coordinates
        common_origin = tuple(max(ct, pt) for ct, pt in zip(ct_origin, pt_origin))
        common_end = tuple(min(ct, pt) for ct, pt in zip(ct_end, pt_end))

        # Calculate the output shape
        common_shape = tuple(round((end - origin) / spacing) for end, origin, spacing in zip(common_end, common_origin, self.target_spacing))

        # Crop images
        ct_aligned = ct_img[round((common_origin[2] - ct_origin[2]) / self.target_spacing[2]): round((common_origin[2] - ct_origin[2]) / self.target_spacing[2]) + common_shape[2],
                     round((common_origin[0] - ct_origin[0]) / self.target_spacing[0]): round((common_origin[0] - ct_origin[0]) / self.target_spacing[0]) + common_shape[0],
                     round((common_origin[1] - ct_origin[1]) / self.target_spacing[1]): round((common_origin[1] - ct_origin[1]) / self.target_spacing[1]) + common_shape[1]]

        pt_aligned = pt_img[round((common_origin[2] - pt_origin[2]) / self.target_spacing[2]): round((common_origin[2] - pt_origin[2]) / self.target_spacing[2]) + common_shape[2],
                     round((common_origin[0] - pt_origin[0]) / self.target_spacing[0]): round((common_origin[0] - pt_origin[0]) / self.target_spacing[0]) + common_shape[0],
                     round((common_origin[1] - pt_origin[1]) / self.target_spacing[1]): round((common_origin[1] - pt_origin[1]) / self.target_spacing[1]) + common_shape[1]]

        return ct_aligned, pt_aligned

    def _remove_artifacts(self, mask):
        """Removes artifacts in the segmentation masks."""

        # Connected component analysis
        labeled_mask, num_labels = measure.label(mask, connectivity=1, return_num=True)

        # Calculate region sizes
        region_sizes = [np.sum(labeled_mask == label) for label in range(1, num_labels + 1)]

        # Sort regions by size
        sorted_regions = sorted(zip(range(1, num_labels + 1), region_sizes), key=lambda x: x[1], reverse=True)

        # Keep the largest portions
        if len(sorted_regions) >= 2:
            selected_labels = [label for label, _ in sorted_regions[:2]]  # If the lungs are separated
        else:
            selected_labels = [label for label, _ in sorted_regions]  # If the lungs are connected

        # Create a new mask containing only the selected regions
        new_mask = np.zeros_like(mask)
        for label in selected_labels:
            new_mask[labeled_mask == label] = 1

        # Fill holes
        new_mask = binary_fill_holes(new_mask)

        return new_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--database_path', help='Path to database')
    parser.add_argument('-dp', '--data_path', help='Path to folder containing images')
    parser.add_argument('-op', '--output_path', help='Path to output folder')
    parser.add_argument('-ts', '--target_spacing', nargs='+', help='Target spacing')
    args = parser.parse_args()

    df = pd.read_csv(args.database_path)
    if args.target_spacing is not None:
        processor = Processor(df, args.data_path, args.output_path, args.target_spacing)
    else:
        processor = Processor(df, args.data_path, args.output_path)
    processor()
