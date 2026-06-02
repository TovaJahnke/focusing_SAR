# Ship Detection

This folder contains the code used for ship detection analysis in my master's thesis. The model in the thesis was trained using a seperate HRSID data set.

The ship detection code is used to evaluate how different focused SAR images affect downstream detection performance. The purpose is to compare detection results for images produced using different data reduction and focusing configurations.

The detection results are used as an application-level metric to complement image quality metrics such as PSLR, ISLR, FWHM, and visual analysis.

## Main purpose

- Detect ships in focused SAR images
- Compare detection performance between different focusing configurations
- Evaluate the effect of data reduction on practical image usability
- Support the analysis of quality–cost trade-offs in onboard SAR processing

## Notes

This code was developed for research purposes as part of a master's thesis project. Some paths, thresholds, and dataset-specific settings may need to be adapted before running the scripts on another system.
