# Kaggle-Dogs_vs_Cats_PyTorch
kaggle competition: Dogs_vs_Cats_PyTorch Presentation 

step1:
      git clone https://github.com/JackwithWilshere/Kaggle-Dogs_vs_Cats_PyTorch
      
step2:
      from the website https://www.kaggle.com/c/dogs-vs-cats/data download the file train and test1 to the file "./data"
      In the command line:
      cd Kaggle-Dogs_vs_Cats_PyTorch
      python dog_rename.py (to rename the dog.jpg)

step3:
      In the command line:
      python train.py #train
      python test.py  #test and generate the submission csv file
      
In addition,the processing of the train data can be separated to two files to hold the cat and dog picture respectively,
thus we can use the ImageFolder in the PyTorch.
