Data download scripts contains notebooks that generate the list of urls that need to be hit to download some of the datasets used in this project. After creating a list of urls, you can download them with a url downloading tool. I recommend aria2c, which is what I used.

For EEA data, you can bulk download directly from https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d. Click on the air quality data download service, and follow the instructions.

Sinaica data uses a custom scraper I built, using the R package Rsinaica as a reference. As a result sinaica_preprocess.py does both the download and preprocessing together.

We do not use openaq data in our paper, but I have added scripts to allow downloads and preprocessing from there as well. This requires a OpenAq API key which you can create for free on their website. It then reads from the OpenAq AWS S3 bucket to bulk download data.

The Data Process scripts convert the raw data downloaded into one unique format. They also convert all timezones to local time and handle unit conversion. They generate csv files that contain one station and one pollutant per file.

visualise.py allows you to create a heatmap of all the data for a pollutant and dataset, letting you see trends and missing data.

imputation.py filters and creates the imputed dataset with no missing data.

data.yaml contains paths for each script to use to save raw files, imputed and visualise files etc. It also contains the limits for pollutants for visualise.