# reproduce our results on CIFAR10
It might be helpful to modify scripts/grid_search.py to generate start_search.sh to generate a run script to generate results on your machine.
## environment
stylegan3-editing-cifar10/environment/sg3_env.yaml contains the library dependencies.

you can use miniconda
```
conda env create -f stylegan3-editing-cifar10/environment/sg3_env.yaml
conda activate sg3e
```

## standard, mixup, manifold mixup for cifar10

To reproduce the results for standard, mixup and manifold mixup use a variation of the following command:
--variant can be changed to a value in [standard, mixup, manifold_mixup]

```
python main.py --variant [standard | mixup | manifold_mixup] --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment [none | cifar10] --random_seed [42 | 4711 | 314159]
```

for example (standard, cifar10, 42):
```
python main.py --variant standard --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment cifar10 --random_seed 42
```

## stylegan 2 cifar10 gan mixup
Setup:
Make sure to place sg2c10-32.pkl and grad_latents_00000_50000.npy in the root folder of this project or change the following linex in main.py to point to these files.
```
latent = np.load("/cluster/home/bgunders/dl_inversion_data/grad_latents_00000_50000.npy", allow_pickle=True).item()
gan_model = SG3Generator(checkpoint_path='/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl').decoder.eval().cuda()
```

To reproduce the results for gan_mixup for cifar10 use a variation of the folliwing command:
```
python3.9 main.py --variant mixup_gan --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment [none | cifar10] --random_seed [42 | 4711 | 314159]
```
for example (mixup_gan, cifar10, 42):
```
python3.9 main.py --variant mixup_gan --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment cifar10 --random_seed 42
```

## Retrieve stylegan2 cifar10 latents
Warning: This will take a lot of computational time (encoder will probably take more than 5 days on a 3090, gradient step takes 30 seconds per image on a 1080 ti, you can parallelize this by manually changing the start and end index in the gradient_invert.py file and running it multiple times). You can use our precomputed latent vectors (grad_latents_00000_50000.npy which are the optimized latent vectors using gradient descent with the latent vectors as target, grad_latents_00000_50000_0.025.npy which additionally improved all latents which resultsed in an mse >= 0.025 again but with 4000 steps, and grad_latents_00000_50000_0.02.npy which improved on grad_latents_00000_50000_0.025.npy again with mse >= 0.02 and 4000 steps)

Download models:
download the model from https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl
place it in stylegan3-editing-cifar10/pretrained_models/sg2c10-32.pkl

if it doesnt work directly convert it using https://github.com/NVlabs/stylegan3/edit/main/legacy.py

you might have to set pythonpath to the stylegan3-editing-cifar10 folder or set pwd to .../stylegan3-editing-cifar10 for the following inversion commands
```
PYTHONPATH=.../stylegan3-editing-cifar10
```

### Train encoder
```
python stylegan3-editing-cifar10/inversion/scripts/train_restyle_psp.py \
--dataset_type cifar10_endocde \
--encoder_type ResNetBackboneEncoder \
--exp_dir experiments/cifar10_psp \
--batch_size 8 \
--test_batch_size 8 \
--workers 8 \
--test_workers 8 \
--val_interval 5000 \
--save_interval 10000 \
--start_from_latent_avg True \
--lpips_lambda 0.8 \
--l2_lambda 1 \
--id_lambda 0.1 \
--input_nc 6 \
--n_iters_per_batch 3 \
--output_size 32 \
--max_steps 1000000 \
--stylegan_weights stylegan3-editing-cifar10/pretrained_models/sg2c10-32.pkl
```

### get latents from encoder:
```
python stylegan3-editing-cifar10/inversion/scripts/inference_iterative.py \
--output_path experiments/cifar10_psp/inference \
--checkpoint_path experiments/cifar10_psp/checkpoints/best_model.pt \
--data_path /path/to/cifar10/train \
--test_batch_size 8 \
--test_workers 8 \
--n_iters_per_batch 3 \
```

### improve images using gradient descent
(edit the file directly if you want to only apply it on a subset) note that this will take a very long time:
```
python stylegan3-editing-cifar10/inversion/scripts/gradient_invert.py
```
improve the latents even further (additional 4k steps on all mse(original, inversion) >= mse you decide by updating the script manually):
```
python stylegan3-editing-cifar10/inversion/scripts/gradient_invert_improve.py
```
you will have to manually adjust the paths in the file:
```
decoder_path = '/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl'
latents_path = '/cluster/home/bgunders/dl_inversion_data/grad_latents_00000_50000_0.025.npy'
np.save(out_path + f'grad_latents_00000_50000_0.02.npy', l)
```

## statistics in latex table
run multiple runs and then scripts/latex_stats.py will aggregat them for you
adjust scripts/latex_stats.py paths to point to the results folder (root="..." in the script) and then run (you might have to  ```pip install pandas```)
```
python scripts/latex_stats.py
```
## plot mse between original and inversion
adjust scripts/plots.py to point to the correct latent.npy, running the following command will generate a plot of mse between original and inverted images (you might have to  ```pip install pandas```)
```
python scripts/plots.py
```

# reproduce our results on MNIST
Since we trained the GAN ourselves, you can simply run:
```
main.py --dataset mnist --variant mixup_gan --epochs 50 --mixup_ratio 1.0 --optim adam --learning_rate 0.001 --gamma 0.9 --gan_epochs 80
```
This will automatically train a GAN, the visual feature extractor, the latent code initializer, the latent codes, and the classifier. If the GAN and latent codes are already present, then only the classifier is trained.

To train a normal classifier:
```
main.py --dataset mnist --epochs 50 --optim adam --learning_rate 0.001 --gamma 0.9
```


# files

## main.py
Contains the parser. Also used to load existing files to prepair the training process. Use this file to run experiments.

## deepfool.py
Checks the adversarial robustness of a model.

## model.py
Contains model architectures : classifiers, VAEs and GANs.

## train_test.py
Trains and evaluates the models and generates the reports.

## utils.py
Various functions.

## augmented_datasets folder (automatically generated)
each dataset has its own folder, containing a bunch of .npy files:
- blurred test images, which are used to test robustness to blurring
- the VAE latent codes of the train set (obtained using the VAE's encoder)
- the GAN latent codes of the train set

## models folder (automatically generated)
contains the .pth files for all the trained models:
- The GAN for the specified dataset
- The visual feature extractor used to obtain GAN latent codes (not needed for CIFAR10)
- The latent code initializer, used to obtain GAN latent codes faster (not needed for CIFAR10)
- The VAE

## stylegan3-editing-cifar10, dnnlib, torch_utils

modified code of https://github.com/yuval-alaluf/stylegan3-editing to invert stylegan2 model trained on cifar10 (https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl)

builds on https://github.com/yuval-alaluf/stylegan3-editing

## scripts/grid_search.py

Generates a script to do hyperparameter searching on the Euler cluster

