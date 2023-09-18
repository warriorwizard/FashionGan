# FashionGan

![](https://github.com/warriorwizard/FashionGan/blob/main/output.png)

## Requirements

- tensorflow 1.14
- matplotlib
- pillow
- tqdm
- numpy
- pandas
- seaborn
- scikit-learn
- scipy
- jupyter
- opencv-python
- imageio
- h5py
- requests
- python 3.6

## Usage

### Training

To train the model, run `python train.py` with the following arguments:

```bash
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--sample_interval SAMPLE_INTERVAL]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR] [--dataset DATASET]
                [--data_dir DATA_DIR] [--image_size IMAGE_SIZE]
                [--num_workers NUM_WORKERS] [--latent_dim LATENT_DIM]
                [--lr LR] [--beta1 BETA1] [--beta2 BETA2] [--n_critic N_CRITIC]
                [--clip_value CLIP_VALUE] [--img_channels IMG_CHANNELS]
                [--img_shape IMG_SHAPE]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches
  --sample_interval SAMPLE_INTERVAL
                        interval between image sampling
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between saving model checkpoints
  --checkpoint_dir CHECKPOINT_DIR
                        directory for saving model checkpoints
  --dataset DATASET     dataset to train on
  --data_dir DATA_DIR   directory containing the dataset
  --image_size IMAGE_SIZE
                        size of each image dimension
  --num_workers NUM_WORKERS
                        number of workers for dataloader
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
  --lr LR               learning rate
  --beta1 BETA1         adam: decay of first order momentum of gradient
  --beta2 BETA2         adam: decay of first order momentum of gradient
  --n_critic N_CRITIC   number of training steps for discriminator per iter
  --clip_value CLIP_VALUE
                        lower and upper clip value for disc. weights
  --img_channels IMG_CHANNELS
                        number of image channels
  --img_shape IMG_SHAPE
                        shape of each image
```

### Testing

To test the model, run `python test.py` with the following arguments

