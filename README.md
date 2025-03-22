# Blood Cell Classification with Deep Learning

## Abstract
The project focuses on the classification of 96×96 pixel RGB images of blood cells into eight predefined categories. The project addresses class imbalance, limited data, and visual similarity between classes. 

![0](https://github.com/user-attachments/assets/8300fcda-fc35-48bc-aa7c-c4751cd6d4e8)
![1](https://github.com/user-attachments/assets/52c6194b-5eff-4689-9681-3fb17bea4a62)
![2](https://github.com/user-attachments/assets/eef26a4c-ea13-4f82-9d36-9233ef0b8c0a)
![3](https://github.com/user-attachments/assets/b1d56b9a-ece1-4e96-b767-e5d768c7c307)
![4](https://github.com/user-attachments/assets/d70bc426-38a7-4368-8445-15675dd69b3a)
![5](https://github.com/user-attachments/assets/dc03c174-d19b-4ef6-bdd8-a68de72e13f4)
![6](https://github.com/user-attachments/assets/8c415592-5e2a-442b-a4fd-fe02bb9464ec)
![7](https://github.com/user-attachments/assets/b463f0d4-2ef1-425d-8b83-1a476bd4f20c)

## Dataset
13,759 images, reduced to 11,959 after removing duplicates

8 classes: Basophil, Eosinophil, Erythroblast, Immature Granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet

Data split: 80% training / 20% validation

Preprocessing: Normalization, one-hot encoding, augmentation

## Techniques & Models
Data Augmentation: ImageDataGenerator, AugMix, CutMix, Mixup, Random Translation

Models: Custom CNN, EfficientNetB0 → B2 (with fine-tuning & class weights), ConvNeXt (Tiny → Base), Ensemble (EfficientNet + ConvNeXt) – best performance

## Results
EfficientNetB2: 72% test accuracy
ConvNeXt Base: 73% test accuracy
Ensemble Model: 78% test accuracy
