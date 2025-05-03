Submit a link to your group project code on github. I am expecting functions with docstrings and at least 5 examples of unit tests of those functions. The tested functions do not have to be overly complicated, but you should be able to tell me, in the code, what acceptance criteria (from your requirements document) is being met by this successful test.



Your "README" file should include your project summary, fixed rough draft, and a description of the notebooks or python scripts that are included on your github page. You can use this page as an example: SVRIMG/README.md at master · ahaberlie/SVRIMG · GitHub. If you click on "raw", it will show you the markdown code and how to embed images, make headers, etc. You can copy and paste aspects of this readme and modify the text and use your own figures / demonstration pictures (I would suggest making an 'img' folder in your github project: SVRIMG/img at master · ahaberlie/SVRIMG · GitHub).

# Machine Learning Cloud Classification using Satellite Imagery 

Authors: Paul Eldridge and Lindsay Briggs 

## 1. Introduction and Background 


This cloud classification machine learning model was developed as a part of the EAE 495: Data 
Science for the Geosciences course at Northern Illinois University under Dr. Alex Haberlie. In 
atmospheric science and meteorology alike, clouds have a significant influence on temperature 
and are the source of precipitation (i.e. rain, snow, hail). Clouds give vital insight to forecasters 
and weather scientists of current conditions in the atmosphere. Identifying cloud types via 
satellite can provide another method of estimating rainfall rates.

Clouds and their classifications

Insert table here 

Clouds are classified by height and shape. There are ten basic cloud types that are categorized 
into three subclassifications based on their heights. 


The aim of this machine learner is not to predict the severity or convective mode produced, but 
rather to identify the type of cloud depicted in satellite imagery. Machine learning can be trained to recognize cloud types from different sources such as ground 
based image observations from the surface, satellite imagery and climate models. An issue with 
ground-based imagery is that this method would not be as effective at night due to photos losing 
contrast during nighttime. This is where our second and third methods come into play. By 
training the CCM on satellite imagery, a product such as Long Wave Infrared, clouds would be 
able to be depicted after nightfall to allow our model to function during dark hours.

## 2. Literature 

“A Machine-Learning-Based Study on All-Day Cloud Classification Using Himawari-8 Infrared Data”
“Cloud type classification using deep learning with cloud images” The dataset in this study comprised of 11 folders of pictures, representing a cloud type, the 
eleventh type being contrails. Contrails were included due to similarities in appearance from 
ground observations. There was a total of 2,543 images. The dataset split consisted of 70% 
training, 20% validation, and 10% for the test dataset. Their split was random but maintained 
balanced. 

This study used data from the Himawari-8 satellite and the cloud type product 2B-CLDCLASS-
LIDAR. 130 days of data were selected from the months November 2018, January 2019, March 
2019, June 2019, and July 2019. The Himawari-8 CLTYPE products encompassed ten cloud 
types (clear-sky (Clear), cirrus (Ci), deep convective (Dc), altostratus (As), altocumulus (Ac), 
nimbostratus (Ns), stratocumulus (Sc), stratus (St), and cumulus (Cu)). CPR/CALIOP products 

consisted of 9 types. For the purpose of consistency, clouds were classified into 9 types, merging 
Ci and Cs. Cu has the poorest classification performance during daytime and nighttime. 

1. 
https://github.com/tpmao/cloud-classification-data

The cloud classification model was based off data from Geostationary Operational Enviromental 
Satellite (GOES)16, formerly known as GOES-R, using its Long Wave Infared radiation (LWIR) 
Product. The NOAA s3 bucket was used on AWS to download data. This bucket is part of the 
Registry of Open Data on AWS and can be accessed anonymously at no cost to the user. See 
https://registry.opendata.aws/ for more information. CONUS images were used as opposed to 
full disk images due to the file size. GOES images are stored as netCDF files with each advanced 
baseline imager (ABI) band stored in a different file for a given time. Band 13 (longwave 
infrared radiation) is the specific ABI band of interest to us. These files also contain data about 
the position of the satellite at the time the image was taken. This information was then used to 
convert the “x” and “y” coordinates of the image into degrees of latitude and longitude to flatten 
the image. Each image also contains “nan” data points – these occur at places where the sensor is 
peering into space and not at the earth. These values were masked over to allow for image 
flattening. Each image was then cropped to a set size to ensure uniformity.

## 3. Methodology
1. Data Collection

GOES Satellite imagery was used, specifically band (). The data was accessed through the NOAA s3 bucket on AWS. This bucket is part of the Registry of Open Data on AWS and can be accessed anonymously at no cost to the user. See https://registry.opendata.aws/ for more information. CONUS images were used as opposed to full disk images due to the file size. 
GOES images are stored as netCDF files with each advanced baseline imager (ABI) band stored in a different file for a given time. Band 13 (longwave infrared radiation) is the specific ABI band of interest to us. These files also contain data about the position of the satellite at the time the image was taken. This information was then used to convert the “x” and “y” coordinates of the image into degrees of latitude and longitude to flatten the image. Each image also contains “nan” data points – these occur at places where the sensor is peering into space and not at the earth. These values were masked over to allow for image flattening. Each image was then cropped to a set size to ensure uniformity. 
3. Preprocessing
4. Feature Engineering (?)
5.  Model Selection

K-Nearest Neighbors 

Random Forest 

## 4. results

## 5. References 

## Appendix

Fu, Y., X. Mi, Z. Han, W. Zhang, Q. Liu, X. Gu, and T. Yu, 2023: A machine-learning-based study on all-day cloud classification using Himawari-8 infrared data. *Remote Sens.*, 15, 5630, https://doi.org/10.3390/rs15245630.

# Requirements Document

| PELBCCM-01  | `   
|---------|------------| 
| Priority | Level |
| Sprint | 1 |
| Assigned To | Name |
| User Story   |  |                                                                                                                                       | 
| Requirements | |
| | 1. |
| | 2.|
| | 3. |
| | 4. |
| Acceptance Criteria | |
| | 1. |
| | 2. |
| | 3. . |
| | 4. .|
| Unit Test | | 
```
```
