# reproduce our results
## standard, mixup, manifold mixup for cifar10

To reproduce the results for standard, mixup and manifold mixup use a variation of the following command:
--variant can be changed to a value in [standard, mixup, manifold_mixup]

```
python main.py --variant [standard | mixup | manifold_mixup] --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment [none | cifar10] --random_seed [42 | 4711 | 314159]
```

for example (standard, ciar10, 42):
```
python main.py --variant standard --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment cifar10 --random_seed 42
```

## stylegan 2 mixup
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
for example (mixup_gan, ciar10, 42):
```
python3.9 main.py --variant mixup_gan --dataset cifar10 --optim sgd --epochs 270 --batch_size 32 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --gamma 0.1 --augment cifar10 --random_seed 42
```

## Retrieve stylegan2 cifar10 latents
Warning: This will take a lot of computational time (encoder will probably take more than 5 days on a 3090, gradient step takes 30 seconds per image on a 1080 ti, you can parallelize this by manually changing the start and end index in the gradient_invert.py file and running it multiple times). You can use our precomputed latent vectors (grad_latents_00000_50000.npy)

Download models:
download the model from https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl
place it in stylegan3-editing-cifar10/pretrained_models/sg2c10-32.pkl

if it doesnt work directly convert it using https://github.com/NVlabs/stylegan3/edit/main/legacy.py

you might have to set pythonpath to the stylegan3-editing-cifar10 folder for the following inversion commands
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
improve the latents even further:
```
python stylegan3-editing-cifar10/inversion/scripts/gradient_invert.py
```

# files

## main.py
Contains the parser, use this to run experiments

## deepfool.py
Checks the adversarial score of a model

## model.py
Contains model architectures : classifier, VAEs

## train_test.py
Trains and evaluates the models and prints the report

## utils.py
Various functions

## augmented_datasets folder (generated by main.py)
each dataset has its own folder, containing a bunch of .npy files:
- vae_latent_codes: the latent codes of the train set, obtained using the VAE's encoder

## trained_models folder (generated by main.py)
contains the parameters for all the trained models (TODO add more details)

## stylegan3-editing-cifar10

modified code of https://github.com/yuval-alaluf/stylegan3-editing to invert stylegan2 model trained on cifar10 (https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl)

builds on https://github.com/yuval-alaluf/stylegan3-editing

