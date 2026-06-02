# SAR Focusing

This folder contains the code used for SAR image focusing in my master's thesis.

The focusing pipeline is based on the Polar Format Algorithm (PFA) in Python and is used to form SAR images from phase-history data. The implementation includes processing steps such as k-space construction, interpolation/gridding, inverse Fourier transformation, multilooking, and GeoTIFF output generation.

The code was developed to evaluate how different data reduction strategies affect image quality, computational cost, and suitability for onboard SAR processing.

## Main purpose

- Focus SAR phase-history data into image products
- Compare different gridding/interpolation settings
- Evaluate pulse decimation and aperture reduction strategies
- Generate outputs used for visual and quantitative analysis in the thesis

## Notes

This code was developed for research purposes as part of a master's thesis project. Some paths, parameters, and dataset-specific settings may need to be adapted before running the scripts on another system.
