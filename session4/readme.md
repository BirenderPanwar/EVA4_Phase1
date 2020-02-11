Assignment-4: MNIST Model
-------------------------

Attempt-1:
----------

Total params: 9,720

Validation Accuracy: 99.46%  at Epoch 17

Colab link: https://colab.research.google.com/drive/1vX1ouo0XCsNyUIP5Z01DlfF93_QHRbgw#scrollTo=0m2JWFliFfKT



Attempt-2:
----------

Total params: 7,976

Validation Accuracy: 99.41%  at Epoch 19

Colab link: https://colab.research.google.com/drive/1TYurp6C0Ok7wxKhTCY5d7nCt8Xw35XST#scrollTo=0m2JWFliFfKT


Overall Model Architecture Summary:
-----------------------------------

1. Two Convolution Block is used. No Padding and No Bias in the model.

2. First conv block is followed by 1X1 conv to combine the channels and then maxpooling.

3. Batch normalization is used at each layer to normalize the feature and for regularization.

4. Dropout layer is added after Convolution block-1 to prevent overfitting and model regularization.

5. Batch normaliztaion and Dropout are avoided at last convolution layer which is close to output

6. GAP layer is used at 7X7

7. Network is designed for Global Receptive field of 20X20 

8. StepLR fxn of optim library is used to implemented learning rate scheduler. step_size and gamma parameter is finalized after 
experimenting with different values. 



