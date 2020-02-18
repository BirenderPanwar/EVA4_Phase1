# Assignment-5: MNIST Model (Design Model with Discipline!)

![](images/steps.png)

### Step-1: EVA4S5F1_BaseModel.ipynb

Colab link: https://colab.research.google.com/drive/1BCy1rR2tFRDWYqJFt2-k5KVtXiiLL4u7#scrollTo=YtssFUKb-jqx

Target:
1. Base model and working code

Result: 

1. Parameters: 122,256
2. Best Train Accuracy: 99.03 (15th Epoch) & 99.02 (19th Epoch)
3. Best Test Accuracy : 98.83 (12th Epoch) & 98.63 (13th Epoch)

Analysis:
1. Model is working. Model work flow is ready for incremental changes in subsequent steps.
2. No overfitting as gap between train and test accuracy is less.
3. We can push this model to increase accuracy. However, our target is to design model with less than 20K parameters.
   so we will be updating this model to meet this criteria


### Step-2: EVA4S5F2_LightWeightModel.ipynb

Colab link: https://colab.research.google.com/drive/1BCy1rR2tFRDWYqJFt2-k5KVtXiiLL4u7#scrollTo=YtssFUKb-jqx

Target:
1. Design the lighter model. 
2. Reducing number of kernals in convolution block-1 and block2 layers

Result: 
1. Parameters: 9,576
2. Best Train Accuracy: 98.61 (13th Epoch)
3. Best Test Accuracy : 98.57 (15th Epoch)

Analysis:
1. Good model! No overfitting
3. We can continue working on this model to increase accuracy.


### Step-3: EVA4S5F3_BatchNorm.ipynb

Colab link: https://colab.research.google.com/drive/1P_DSih1JjdPgFtSgjmBo0nPVQ-ABF2IN#scrollTo=8kH16rnZ7wt_

Target:
1. Add Batch Normalization to increase model efficiency. BN is added at each layer except at the last conv layer.

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 98.56 (15th Epoch) and 98.53 (14th Epoch)
3. Best Test Accuracy : 99.39 (11th Epoch) and 99.36 (15th Epoch)

Analysis:
1. Batch Normalization has improved the model efficiency
2. Its Good model! with slight overfitting compared to earlier model.
3. This model can be pushed further to improve efficiency, ideally upto 99.83 (99.39 + 0.44)


### Step-4: EVA4S5F4_Dropout.ipynb

Colab link: https://colab.research.google.com/drive/1bi4S9r0AhjN0FgEOC6z-apGNkjiIqRNC#scrollTo=8kH16rnZ7wt_

Target:
1. Let make model more generalized by adding Regularization using Dropout. Dropout is added at each layer except at the last conv layer.

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 99.17 (15th Epoch) and 99.14 (14th Epoch)
3. Best Test Accuracy : 99.30 (14th Epoch) and 99.29 (15th Epoch)

Analysis:
1. Dropout had helped to regularized the model and reduced overfitting
2. However, with current capacity this model is difficult to push to improve efficient.


### Step-5: EVA4S5F5_ImageAugmentation.ipynb

Colab link: https://colab.research.google.com/drive/16nRXZ2svnZaCr6oYmd3X8A5vUC4BQdzd#scrollTo=8kH16rnZ7wt_

Target:
1. As we analyse the training samples we can see that few images are written in slight rotations. 
   Lets add image augmentation to transofrm the images with slight rotation of +-7 degree.

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 99.01 (15th Epoch)
3. Best Test Accuracy : 99.36 (12th Epoch) and 99.35 (13th Epoch)

Analysis:
1. Train accuracy is reduced. this is expected as few transformed images in trained data might not be existing in test data.
2. However, with current capacity this model is difficult to push further to improve efficient.
3. Model is overfitting. Might be Dropout and Image Augmentation for exsting model does not working well.

### Step-6: EVA4S5F6_StepLR.ipynb [Solution-1]

Colab link: https://colab.research.google.com/drive/1kaq2lVeIbC8Sgkqd0pvvtK9z3FCPVwK6#scrollTo=8kH16rnZ7wt_

Target:
1. Apply StepLR to adjust Learning Rate to help overcomming local minima issue. this is continuation of Step-5 model

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 99.17 (15th Epoch)
3. Best Test Accuracy : 99.50 (13th Epoch) and above 99.40 from 8th Epoch onward

Analysis:
1. No overfitting!, lowest gap seem bettween train and Test accuracy so far.
2. Model is overfitting. Train accuracy is less compared to test accuracy due to transformation images in train dataset.
3. Model still have capacity to push further to improve efficiency. 

### As Dropout and Image augmentaion doest not working well together for the existing model as per Step-5. Let also make attempt to build model without adding Dropout layer. 

### Step-5A: EVA4S5F5_ImageAugmentation_NoDropout.ipynb

Colab link: https://colab.research.google.com/drive/1o_LRkVy_i-EI5rC1aRjZlrvdHatvlAfN#scrollTo=8kH16rnZ7wt_

Target:
1. Apply image augmentation for rotation(+-7 degree) without any Dropout layer

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 99.37 (15th Epoch) and 99.33 (15th Epoch)
3. Best Test Accuracy : 99.44 (13th Epoch) and 99.43 (14th Epoch) [in 3 epochs more that 99.40 is seen] 

Analysis:
1. No overfitting!, not much gap between training and test accuracy
2. Image aumentation without Dropout is working well
3. However, with current capacity this model is difficult to push further to improve efficieny.
4. As per epoc history, model might we stucking in local minima and adding adaptive learning rate might further help in 
improving the efficient and to come out of local minima issue.

### Step-6A: EVA4S5F6_StepLR_NoDropout.ipynb[Solution-2]

Colab link: https://colab.research.google.com/drive/1qbeJncKZSPXGQPenESY3GeSrPKDo-gZ8#scrollTo=8kH16rnZ7wt_

Target:
1. Apply StepLR to adjust Learning Rate to help overcomming local minima issue. this is continuation of Step-5A model

Result: 
1. Parameters: 9,752
2. Best Train Accuracy: 99.43 (13th, 14th Epoch)
3. Best Test Accuracy : 99.49 (11th Epoch) and above 99.40 from 7th Epoch onward

Analysis:
1. No overfitting!, lowest gap seem bettween train and Test accuracy so far.
2. Good model1 as both Train and Test accuracy are high and with less Gap.
3. Model still have capacity to push further to improve efficiency. 

