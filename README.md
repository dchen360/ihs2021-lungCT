# IHS 2021 Code Repository: COVID-19 Severity Prediction from Lung Radiography Images Using Deep Learning

The official code repository corresponding to the results obtained in [COVID-19 Severity Prediction from Lung Radiography Images Using Deep Learning](./IHS_2021_Submission.pdf).

Even though the results listed are subpar, we believe our work holds value in suggesting possible avenues by which researchers may avoid the challenges we faced.

*- Zitian, Danni, Peter, Sofia, and Lawrence*

## Background
- CT scans are more rapid, sensitive than RT-PCR testing for COVID-19.
- High case volumes lead to medical errors as doctors examine CT scans with limited time.
- Need for automated algorithm to accurately triage COVID-19 patients based on CT scans.

## Dataset
1,110 CT scans from hospitals in Moscow, using 25th slice of CT scan

## Method
- VGG16 :
  - A modified VGG-16 model pretrained on ImageNet was used for classification in a transfer learning approach.
  - The last fully connected layer was substituted to achieve necessary output dimensions.
  - All weights were frozen except those of the last layer.

![image](https://user-images.githubusercontent.com/91340560/232606207-664eb37a-c16a-45ea-bc0d-40327d5c8b81.png)
 
- Model Training

![image](https://user-images.githubusercontent.com/91340560/232606328-71eb7f0b-c965-4ce4-b626-2fce5a52ddf3.png)

## Results
![image](https://user-images.githubusercontent.com/91340560/232606582-c0bdbbdb-3ed5-44ac-bd81-759434bebf7f.png)
![image](https://user-images.githubusercontent.com/91340560/232606607-b91cd3cc-8e49-4636-b67e-b3f6b968197e.png)
![image](https://user-images.githubusercontent.com/91340560/232606660-80d70056-ef40-42da-acfb-41bbb6b43a0a.png)

## Insight
- Model performed well on training data but guessed randomly on testing data.  
- Overfitting 
  - Insufficient data, especially on non-dominant classes
  - Extract >1 slice from each image may help
  - Slice choice may not be ideal
- Future Work
We’d try to troubleshoot the model by...
  - Tuning the learning rate to see if global minima are being skipped over.
  - Pooling lung images of different slices to mimic basic ‘data augmentation.’
  - Using explainable AI frameworks (e.g. GradCAM) to investigate features driving prediction in the absence of other leads.
