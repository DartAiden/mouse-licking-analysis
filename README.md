This is a CNN that helps analyze mouse behavior using both OpenCV and PyTorch by determining if a mouse is licking or not in a frame.

#Methods
To perform the analysis of licking, I created a convolutional neural network (CNN) with PyTorch (Paszke et al., 2019). This CNN was trained on 5,319 frames and consists of two convolution layers, two pooling layers, two activation layers, two dropout layers, and four fully-connected layers. ColorJitter and RandomCrop transforms were applied to reduce overfitting. The training data was cropped to better standardize the data. To analyze whether or not stimulation was present, I extracted the intensity of the blue channel of the initial frame and compared it to the intensity of the blue channel of every succeeding frame; those with around ~2.6% higher intensity of the blue channel were flagged as stimulation. Then, every frame from the testing data was fed into both the CNN and the function to analyze blue channel intensity, and the time stamps were recorded. Then, a cross-correlation was performed between the extracted time series for the licking signals and the stimulation signals. The data was graphed with MatPlotLib (Hunter, 2007). 

A graph of the time series, with the cross-correlations labeled. 

![alt text](https://github.com/DartAiden/mouse-licking-analysis/blob/main/final_data/figures_stim.png "Figure stim")

#Citations
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. _In Advances in Neural Information Processing Systems_ 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf 
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering_, 9(3), 90-95.
