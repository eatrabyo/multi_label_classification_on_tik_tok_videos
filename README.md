# DADS 7202 Multi-label Classification of TikTok Videos

## Introduction
Perform multi-label classification on a total of 12 classes of TikTok videos dataset and compare performance between 2 transformer architectures (MViT, Swin 3D) and a non-transformer architecture (Video ResNet).

## Dataset
We obtained 796 videos from TikTok using the web scraping library “douyin_tiktok_scraper,” which we manually labeled into 12 classes.
* Weight Lifting
* Running
* Yoga
* Healthy Lifestyle & Weight Loss
* Haircare
* Makeup
* Skincare
* Outfit
* Accommodation
* Adventure
* Culture
* Food and drink
 
## EDA
The number of videos used for each class is imbalanced (varies from 7 to 172 videos). The imbalance of the dataset can impact the performance of the training models.
The stratification data split and BCE loss are applied to help address this issue.

![tiktok_classbalance](https://github.com/user-attachments/assets/af00cbbd-72eb-4fc8-9a60-b65f7c353d44)

Most of the Videos used in this study are single-label videos (containing only one class). There are about 14% of the dataset are classified as multi-label videos (containing more than one class)

![num_label](https://github.com/user-attachments/assets/50dee11c-f79b-4789-aca7-42d822c2a4bd)


The duration of videos varies from 5.6 to 600 secs which can cause errors in the training of models.  The data preprocessing to resize all videos to the same length is applied.

![vid_duration](https://github.com/user-attachments/assets/5bb94f5f-f620-46b8-813a-ba54300a7f4d)  |  ![vid_duration_class](https://github.com/user-attachments/assets/ce4e0ce6-1e6a-44ce-adbf-25fefd304aaa)
:-------------------------:|:-------------------------:

## Splitting Data
We split Train:Test:Validate by 80:10:10, using Iterative Stratification Split resulting in 636 videos for training, 80 videos for testing, and 80 videos for validating.
From each dataset, we read each video only the first 10 seconds and slice by step of total frames floor division by 16 and then use only 16 frames.
After slicing, we put it into the data loader by using these parameters:
* Batch Size = 32
* Shuffle = True
* Num_workers = 4

![train](https://github.com/user-attachments/assets/7c5e3335-f55c-4d1f-baba-461366ada082) | ![test](https://github.com/user-attachments/assets/5ecd8991-df80-4dcc-9362-f6d7e49a0d48) | ![validate](https://github.com/user-attachments/assets/390d2dc3-81fb-46f2-b12f-bc81ebca1abf)
:-------------------------:|:-------------------------:|:-------------------------:
|Train set | Test set | Validation set|

## Model Training

### Classifier
Each model has the same classifier and uses pre-trained weight from KINETICS400_V1, as shown below:
1. MViTV2_S
```
mvit = mvit_v2_s(weights="KINETICS400_V1")
for param in mvit.parameters():
    param.requires_grad = False

for param in mvit.blocks[15].parameters():
    param.requires_grad = True

mvit.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768,12),
            nn.Sigmoid()
        )
```

3. SWIN3D_T
```
model_swin = swin3d_t(weights="KINETICS400_V1")
for param in model_swin.parameters():
    param.requires_grad = False

for param in model_swin.features[6].parameters():
    param.requires_grad = True

model_swin.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768,12),
            nn.Sigmoid()
        )
```

4. R2PLUS1D_18
```
model_resnet = r2plus1d_18(weights="KINETICS400_V1")
for param in model_resnet.parameters():
    param.requires_grad = False

for param in model_resnet.layer4.parameters():
    param.requires_grad = True

model_resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512,12),
            nn.Sigmoid()
        )
```

### Training Method
|   Training Method  |                         |
|:------------------:|:-----------------------:|
|      Optimizer     |           Adam          |
|        Loss        |   Binary Cross Entropy  |
|        Epoch       |            10           |
|    Learning Rate   |          0.001          |
|     Training on    | Last Layer + Classifier |
| Pre-trained Weight |      KINETICS400_V1     |

## Evaluation

|          |    Accuracy   |       F1      |     Recall    | Avg Precision |
|:--------:|:-------------:|:-------------:|:-------------:|:-------------:|
| MViTV2_S | 0.93 ± 0.0514 | 0.98 ± 0.0171 | 0.97 ± 0.0257 | 0.97 ± 0.0214 |
| SWIN3D_T | 0.89 ± 0.0566 | 0.96 ± 0.0188 | 0.94 ± 0.0283 | 0.95 ± 0.0236 |
| R2Plus1D | 0.86 ± 0.0311 | 0.95 ± 0.0104 | 0.93 ± 0.0155 | 0.94 ± 0.0129 |
