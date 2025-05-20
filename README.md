# Cell Detection Prototype

This project is a deep learning-based prototype for the **automatic detection and localization of cells** in histological image data (e.g., `.czi` files). It aims to support researchers by reducing the time-consuming task of manual cell counting and annotation.

## Motivation

The motivation behind this project was twofold:  
First, to gain hands-on experience with **deep learning for image processing** beyond theoretical understanding. Second, to help automate a real-world task that was previously done manually and repetitively.

This project became an opportunity to apply deep learning to a meaningful use case, while simultaneously building a tool that can assist someone in their research workflow.

## What the Project Can Do

- Load and normalize `.czi` microscopy images  
- Detect cell centers using a trained convolutional neural network and `peak_local_max`  
- Generate prediction heatmaps and visualize cell contours using `find_contours`  
- Map predictions back onto the original image  
- Visualize cell locations with bounding boxes or shape-based contours  
- Export detected cell data to CSV

## Prototype Status

This is a **functional prototype**. The goal was to prove feasibility and explore deep learning concepts in a practical setting. However, it is important to note:

- The model architecture, dataset, and parameters are kept intentionally simple  
- The project prioritizes core functionality over optimization or scalability  
- There is significant potential for further development and refinement

## Possible Extensions

- Train on larger and more diverse datasets  
- Add support for multi-class or multi-channel detection  
- Develop an interactive GUI (e.g., PyQt)  
- Improve model evaluation, checkpointing, and error handling  
- Integrate into lab workflows or batch-processing pipelines
