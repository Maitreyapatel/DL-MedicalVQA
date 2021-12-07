# Medical Visual Question-Answering

### Installation:
Please use requirements.txt to install the necessary python packages:
`pip3 install -r requirements.txt`

### Folder management
Please create following folders:
`models`: To store trained models in various experiments.
`dataset`: To store the various data.
`runs`: To store the tensorboard training summary.

### Dataset
1. To get the dataset, please visit [https://www.aicrowd.com/challenges/imageclef-2021-vqa-med-vqa](https://www.aicrowd.com/challenges/imageclef-2021-vqa-med-vqa) and request for the dataset. Because of the agreement issues, we are not sure whether we are allowed to share them or not. 
2. After downloading all the zip files, including 2020-challenge train/val and 2021 challenge new validation set, extract them inside dataset directory.
3. At last, follow the path requirements in `./data_preprocessing.ipynb` to combine 2020 challenge's training and validation data single training data. This will create final training data.
4. Now, visit [https://github.com/abachaa/VQA-Med-2021](https://github.com/abachaa/VQA-Med-2021) to get the testing dataset and follow the same instruction as step 2.

### Pre-trained weights
Pre-trained model weights can be found at: [https://drive.google.com/drive/folders/1K9f-huVsGUSgSVGfaYKL-XeygAm7IDMU?usp=sharing](https://drive.google.com/drive/folders/1K9f-huVsGUSgSVGfaYKL-XeygAm7IDMU?usp=sharing)

### Results
Please refer the image id wise predicted answers by various models in `results` folder. One needs to run `./eval.ipynb` to get the accuracy and bleu scores. Keep in mind that this part also requires the access to the testing dataset.

### View existing training logs
Please run folllowing command in exitsting project directory:
`tensorboard --logdir=runs`

Note: Consider the experiment results with highest run count. For example, xx_run_3 is the final version instead of xx_run_2.

### Additional contraints
Now, we have dataset prepared then run any `.ipynb` notebooks with following path specific constraints:
1. All the models are stored in `models` folder for simplicity. However, keep in mind that model does not get overwirtten. This is ensured while performing the training.