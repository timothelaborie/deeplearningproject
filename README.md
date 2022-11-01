# files

## main.py (uses deepfool.py)
obtains an accuracy and robustness score for each of the various augmented datasets.

## datasets folder
each dataset has its own folder, containing a bunch of .npy files:
- train and train_labels: the original train set
- test and test_labels: the original test set
- mixup and mixup_labels: the train set, along with random pairs of interpolated images
- test_blurred: blurred test images
- vae_images and vae_latent: random images from the vae with their corresponding latent vector, used to train a model to retrieve the latent vector
- vae and vae_labels: the train set, augmented by interpolated images from the VAE

## data_augmentation folder
contains files used to augment the train/test sets using the various methods

### GAN.py
generates interpolated images using a GAN. (todo)

### VAE.py
generates interpolated images using a VAE.

### feature_extractor.py
trains and saves a generic classifier that is used to find the latent vectors for the 2 files above

### dump_orig_datasets.py
saves mnist, cifar-10 etc to a numpy file so other data can be appended

### mixup.py
interpolates images using mixup

### blurring.py
blurs test images

## trained_models folder
contains the parameters for all the trained models

