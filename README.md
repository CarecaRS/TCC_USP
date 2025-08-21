# Credit default prediction through Machine Learning models: a study using US interest rates

## Abstract
Financial institutions implement their credit policies based on credit and default analysis systems, while being constrained by regulations established by supervisory bodies regarding best practices, calculation methodologies, financial reserves and other factors.

This study uses data from credit contracts provided by Lending Club Corporation, along with current and future U.S. interest rates in effect at the time of each contract, aiming to develop five different models for predicting default rates on such proposed loans, in order to analyze their performance in terms of correct classification and, thus, improve business profitability by avoiding potential losses and reducing amounts retained as default risk.

None of the models performed better than the others in every metric analyzed (accuracy, sensitivity, specificity, and precision). The neural network model (TensorFlow) obtained the best result in the Accuracy and Sensitivity metrics (0.994406 and 0.965922, respectively), while the Gradient Boost model (CatBoost) obtained the best result in the Specificity and Precision metrics (0.999971 and 0.999767, respectively). 

## Data Sources
The original dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club/data?select=accepted_2007_to_2018Q4.csv.gz). The effective federal funds rate (EFFR), the proxy for the interest rates used here, is available at the [Federal Reserve Bank of St. Louis](https://www.stlouisfed.org/).

## Acknowledgments

I'm sincerely thankful to my advisor, Professor Elton Gean Araújo, who throughout our short journey together always encouraged me to produce more knowledge and always make it more accessible.

## The coding and running
All the code for the whole project is at the `tcc.ipynb` file. For reproducibility please be aware of the absolute paths and file formats. Beside that, everything should run pretty good, given that one has the necessary Python packages installed. This work was done using Python 3.12.3 on Arch Linux (btw).

The main machine was an old Ryzen 5-1600 CPU, with 16Gb RAM and a NVIDIA 1060 GPU, using regular SSD for storage. It was mandatory for me to allocate some more Swap memory using the free space in the SSD, or else there was always an overload of RAM, specifically in the code part where It was needed to merge some data. Maybe with better specs this isn't necessary.
