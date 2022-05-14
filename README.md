
# Image Processing based ML Framework for Leaf Classification and Disease Detection


## Abstract:
Image processing techniques are employed in this
paper to supplement machine learning models for categorizing
leaf species and detecting unhealthy spots in them. Multiple public
datasets were mixed with a self-curated dataset of leaves from a
homemade garden for the same. A set of 17 distinct features are
extracted from background-subtracted leaf photos to capture
information about texture, colour, and shape for leaf
classification. These characteristics are used to train machine
learning models such as Support Vector Machine and K-Nearest
Neighbors, which do not require hardware intensive training.
These models are trained on several dataset combinations,
including the merged one, and then compared using precision,
recall, and F1-score measures.

## Methodology
A feature vector was generated from the leaf image and subjected to various machine learning models to identify the type of leaf from its image. For this, the data collection comprised of Flavia, Mendeley, Diseased Mendeley, and Garden Images

<img src="https://user-images.githubusercontent.com/52126773/168424027-da53b0db-4454-4aad-99c6-8bd1eb065b20.png" data-canonical-src="https://user-images.githubusercontent.com/52126773/168424027-da53b0db-4454-4aad-99c6-8bd1eb065b20.png" width="200" height="400" />

A. Pre-processing

All photos were downsized to a standard size of 1600*1200*3 before being converted from RGB to grayscale for additional thresholding. A gaussian blur with kernel size
(55,55) was used to minimise the noise in this image. This noise-reduced image is then thresholded using Otsu's approach  to produce a binary image that separates the leaf body from the background. For more uniform masking, holes were also closed. This mask is applied to the original leaf image, as illustrated in Fig. 3(e), to ensure that the background does not influence leaf classification.

B. Leaf Classification

Features extracted:

![Capture](https://user-images.githubusercontent.com/52126773/168425248-8a578a85-a555-4795-9285-be862eafd9d3.PNG)

Colour based Features:

The extracted colour features were the mean value of the R component, the mean value of the G component, and the mean value of the B component. In addition, the standard deviation of these components, namely R, G, and B, was taken into account. 

Texture based Features:

The texture-based characteristics are estimated from the grayscale background-subtracted leaf acquired in the preceding phases. The Mahotas library  was used to calculate texture-based characteristics. Contrast, correlation, inverse difference moments, and entropy were chosen from among the 13 Haralick traits to be included in the set. 

Shape based Features:

The following characteristics are extracted: Area, Perimeter, Physiological Length, Physiological Width, aspect ratio, rectangularity, and circularity. The shape-based feature extraction yields a total of seven features. 

![App Screenshot](https://user-images.githubusercontent.com/52126773/168425434-68129ba3-be39-4512-bf70-08ed3ba97dfd.png)

C. Disease Detection 

Different colour spaces were investigated for the disease detection part. The masked image with the background removed was utilised to detect illness. This RGB image has been translated into the HSV and Lab colour spaces. Each component of both colour spaces was identified, yielding six distinct images. Otsu thresholding was used on these six separate components, and the output was analysed further. 

<img src="https://user-images.githubusercontent.com/52126773/168425586-e6d0276d-de54-48ef-888e-cc6d35b4b4ba.png" data-canonical-src="https://user-images.githubusercontent.com/52126773/168425586-e6d0276d-de54-48ef-888e-cc6d35b4b4ba.png" width="400" height="400" />

## Results

https://www.youtube.com/watch?v=BJHqc2oKOUs

## Conclusion

This study suggests a method for classifying leaves and finding diseased spots within them. Instead of utilising over parameterized deep learning models, the emphasis in this study is on establishing a workflow that successfully uses image processing techniques to complement machine learning models. This has the advantage of increasing approach simplicity, interpretability, and lowering the computing resource and training time burden. The obtained findings support the idea that this hybrid strategy can generate good performance in terms of various accuracy measures even when the combined dataset comprises of many classes originating from multiple datasets taken under different conditions.
