# Nural_Network

The Neural Network is Based on the stochastic gradient descent  (SGD) algorithm, and was tested on 2 dimensional vectors. 
The original intent was to include both standard as well as Residual Network (ResNet) steps, though in the end only the standard steps were fully implemented

The network can be run from [testNN](https://github.com/edanyaar/Nural_Network/blob/master/test_NN.m), where the parameters such as the batch size and the size of each layer can be adjusted prior to each run. 

The Network was used to sort several data-sets, one such exmaple is attached below:

![peaks](/examples/Peaks.png)

a batch size of 100 and a layer size of [2-5-5-5] identified correctly ~ 94% of the data. results closer to 100% can likely be achived with some further finetuning.

![peaks_res](/examples/Peaks_results.png)

