# feto2024
UCL MSc project for Computer Graphics, Vision and Imaging 

Abstract:

The following project focuses on enhancing the analysis of fetoscopic surgery for twin-to-twin transfusion syndrome through advanced computer vision techniques. To simulate the patient undergoing fetoscopic laser ablation surgery, a twin-to-twin transfusion syndrome doll (phantom) was employed. The project aims to address the challenges posed by the narrow field of view in surgical cameras, which are crucial for tasks like segmentation and image mosaicking. The research involves streamlining data acquisition and validating outcomes, beginning with hand-eye calibration, followed by manual tracking and video data synchronisation. A significant aspect involves verifying computer vision algorithms and employing segmentation and mosaicking techniques on newly recorded datasets. Accuracy metrics are employed to assess and potentially refine algorithmic performance. Success in these areas will lead to the development of a system for synchronous data acquisition between surgical cameras and NDI Aurora tracking sensors, further advancing the application of computer vision in medical procedures.

Scripts:

The following scripts require scikit-surgery (Calibrate, NDI tracker) for purposes of capturing and calibrating data

Calibration Directory -> Calibtrationg scrips, reprojection and hand eye 

ImageEdit Directory -> Any image manipulation (Cropping, resizing, greyscale conversion)

Remaining scripts are to achieve registration, mosaicking and trajectory analysis

Segmentation scripts:
To be uploaded 

