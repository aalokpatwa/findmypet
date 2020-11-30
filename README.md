# findmypet
- 1st in Category Award, Synopsys Santa Clara Science and Technology Championship, Spring 2018
- Abstract

```
Find My Pet: UnLEASHING the Power of Image Neural Networks to Unite Lost Pets with Their Owners
Aalok Patwa
Archbishop Mitty High School

Estimates indicate that over 10 million pets go missing each year, and one in three pets will become lost at one point in their life. 
Intelligence about the location of the lost pet would save time, money, and stress.  

My project relies on a photo of a pet, taken by a pedestrian, to determine the pet’s identity. 
Image recognition technology would match the photo to a lost pet and notify the owner of the pet’s location.

In order to evaluate the efficacy of my algorithm, I decided on three design criteria: first, the algorithm accurately classify a s
ubmitted smartphone photo, no matter the angle or the distance that the photo was taken; second, the owner of the lost pet should 
only be required to submit 4-10 images in making the profile for their pet; third, the algorithm should match the found pet with three or less 
lost pets with 80% accuracy.

Using a process known as transfer learning, I re-trained a Google Inception v3 Convolutional Neural Network without using a 
large dataset or extreme computing power. In transfer learning, the Inception Network, which has already been trained on the 
immense lmageNet database, adds another layer of neurons for the classes of pets needed for identification. That layer, called a 
Softmax classifier, can classify the image as a certain dog. The classifier takes in an image embedding – a 2048-dimensional vector 
representation of the photo and learns weights and biases to connect those numbers with the right classes. In this way, Inception’s 
pre-learned knowledge can be specifically used for a certain niche.

I used various novel image augmentation algorithms to increase the diversity and size of my training dataset. After experimentation, 
I found that a random combination of flipping, blurring, and rotating, done by the Python OpenCV2 module, leads to the most accurate model. 

I also coded a script to extract GPS metadata from an image using the Python Imaging Library PILLOW directory. This data allows for usage of 
neighborhood-specific models and also crucially provides the owner with the location of their pet.

After training and testing multiple different models, I found that I met all three of my design criteria. My project only requires a smartphone 
picture that contains the pet – it need not be a professional headshot. In creating a profile for their pet, the owner must simply submit 4-10 images, 
as image augmentation scripts transform the images into a sizeable training dataset. Lastly, the algorithm matches the found pet to three profiles 
with 87.93% accuracy.

Using cutting-edge techniques, I created a system that costs no money to use and very little to operate, and is simple enough for the average 
person to understand. In the future, I can turn this idea into a real application, either using SMS APIs or writing an app. Overall, the project 
helps reunite people with what they love most: their pet!
```



