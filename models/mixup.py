import numpy as np
import matplotlib.pyplot as plt

x,y = np.load("../datasets/mnist/mnist_orig.npy"), np.load("../datasets/mnist/mnist_orig_labels.npy")
print(x.shape)
print(y.shape)

#generate new training data
generated_images = []
generated_labels = []
for i in range(0,60000):
    #select 2 images at random
    idx1 = np.random.randint(0,x.shape[0])
    image1 = x[idx1]
    label1 = y[idx1]
    idx2 = np.random.randint(0,x.shape[0])
    image2 = x[idx2]
    label2 = y[idx2]

    alpha = 0.5
    #interpolate between the two images
    image = alpha*image1 + (1-alpha)*image2

    #interpolate between the two labels
    label = alpha*label1 + (1-alpha)*label2

    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(image)
    # ax[1].imshow(image1)
    # ax[2].imshow(image2)
    # print(label)

    #add to the list
    generated_images.append(image)
    generated_labels.append(label)
    if i % 6000 == 0:
        print(i)

generated_images = np.array(generated_images)
generated_labels = np.array(generated_labels)
print(generated_images.shape)
print(generated_labels.shape)
#append generated data to original data
x = np.concatenate((x, generated_images), axis=0)
y = np.concatenate((y, generated_labels), axis=0)
#convert to bytes
x = x.astype(np.uint8)
print(x.shape)
print(y.shape)
# print(x[70000])
# print(y[70000])
#save the generated images and labels
np.save("../datasets/mnist/mixup.npy", x)
np.save("../datasets/mnist/mixup_labels.npy", y)