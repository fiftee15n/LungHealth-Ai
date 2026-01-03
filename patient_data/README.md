# Patient Data Folder Structure

This folder contains patient imaging data uploaded by the radiology department.

## Folder Structure

Each patient folder should be named: `Patient_XX` where XX is the patient number.

Each patient folder must contain:
1. `patient_info.txt` - Patient demographics and information
2. `xray.jpg` or `xray.png` - X-Ray image
3. `ct_lung.jpg` or `ct_lung.png` - CT Lung Window image
4. `ct_mediastinal.jpg` or `ct_mediastinal.png` - CT Mediastinal Window image

## patient_info.txt Format

```
Name: [Patient Name]
Age: [Age in years]
Gender: [Male/Female/Other]
Smoking: [Non-smoker/Former smoker/Current smoker]
Diagnosis: [Central with patent bronchus/Central without patent bronchus/Peripheral malignancy/Infection/Normal]
```

## Example

Patient_01/
├── patient_info.txt
├── xray.jpg
├── ct_lung.jpg
└── ct_mediastinal.jpg

## Notes

- Images should be in JPG or PNG format
- All three images are required for complete analysis
- The diagnosis in patient_info.txt is used for validation/ground truth
