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

Download models:
download the model from https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl

if it doesnt work directly convert it using https://github.com/NVlabs/stylegan3/edit/main/legacy.py

you might have to set pythonpath to the stylegan3-editing-cifar10 folder
```
PYTHONPATH=.../stylegan3-editing-cifar10
```

## Train encoder
```
stylegan3-editing-cifar10/inversion/scripts/train_restyle_psp.py \
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

## get latents from encoder:
```
python stylegan3-editing-cifar10/inversion/scripts/inference_iterative.py \
--output_path experiments/cifar10_psp/inference \
--checkpoint_path experiments/cifar10_psp/checkpoints/best_model.pt \
--data_path /path/to/cifar10/train \
--test_batch_size 8 \
--test_workers 8 \
--n_iters_per_batch 3 \
```

## improve images using gradient descent
(edit the file directly if you want to only apply it on a subset) note that this will take a very long time:
```
stylegan3-editing-cifar10/inversion/scripts/gradient_invert.py
```
improve the latents even further:
```
stylegan3-editing-cifar10/inversion/scripts/gradient_invert.py
```
