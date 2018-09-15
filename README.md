# DCDE-MSVMs
DCDE: An Efficient Deep Convolutional Divergence Encoding Method for Human Promoter Recognition
==============================================================================================
DEPENDENCIES:

OS: Windows 10

1. Python 2.7
2. numpy 1.12.1
3. Biopython 1.68
4. scikit-learn 0.18.2
5. theano 0.9.0
6. Keras 2.0.6
7. kPAL 2.1.1

============================================================================================
USAGE:

Data:

example:
  traning set(positive): training_2_positive.fasta
  traning set(negative): training_2_negative.fasta
  test set(positive):    test_2_positive.fasta
  test set(negative):    test_2_positive.fasta
  
STEP 1:  Deep Convolutional Divergence Encoding: Informative kmers settlement

  Example run:
    python IFkmerS.py
   
STEP 2:  Deep Convolutional Divergence Encoding: CNN secondary encoding

  Example run:
    python CNN_2nd.py
    
STEP 3:  MSVMs recognition
  
  Example run:
    python mySVM.py
    
STEP 4:  Bilayer Dicision Model

  Example run:
    python BD.py
