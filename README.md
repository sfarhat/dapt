# DAPT: Distillation as an Alternative to Pre-Training

This repo contains the original Pytorch implementation of the following paper:<br><br>
[On the Surprising Efficacy of Distillation as an Alternative to Pre-Training Small Models]()  
 [Sean Farhat](https://sfarhat.github.io/), [Deming Chen](https://dchen.ece.illinois.edu) <br>
 University of Illinois, Urbana-Champaign   
 
which appeared at the 5th Practical ML for Low Resource Settings ([PML4LRS](https://pml4dc.github.io/iclr2024/)) Workshop at ICLR 2024.

## Step 0: Setting Paths

Go into `paths.json` and edit the paths to point to where you'd like the datasets (`data_path`, `syn_data_path`) and checkpoints (`cpt_path`) to be saved and loaded.

## Step 1: Obtaining the Desired Teachers

To obtain the desired teachers, we must first train them on the task of interest. This can be achieved via 3 methods, depending on how we wish for the teacher to learn.

* Scratch (FR): The network is randomly initialized and fully trained on the task end-to-end.
* Linear Probed (LP): The network is initialized with a feature backbone pre-trained on ImageNet. Then, **only** it's task-specific head is trained on the task.
* Full Finetuning (FT): The network is initialized with a feature backbone pre-trained on ImageNet. Then, it is fully trained on the task end-to-end. It's task-specific head uses a higher learning rate than the body.

> Note: For our experiments, we take the ImageNet pre-trained weights from Pytorch's model hub.

To create these teachers, edit and use the appropriate `scripts/train_(fr|lp|ft).sh` scripts. 

All possible command line arguments can be found by running `python train_(fr|lp|ft).py --help`.

These will create and save the best and last model checkpoints in `<data_path>/<model>_<fr|lp|ft>_<optimizer>`

## Step 2a: Distill

Edit and run `scripts/distill.sh`. In this script, we have several options to control the assistance process.

* `--dataset` chooses the task. Options: `cifar100`, `cifar10`, `mit_indoor`, `cub_2011`, `caltech101`, `dtd`
* `--teacher_model` chooses which teacher model we use. Step 1 *must* be completed for this script to find the desired model. Options: `resnet50`, `vit-b-16`
* `--teacher_init` chooses the initialization of the teacher. Options: `fr`, `lp`, `ft`
* `--student_model` chooses the student model which is initialized randomly. Options: `mobilenetv2` or `resnet18`
* `--distill` chooses the distillation algorithm. Options: `align_uniform`, `crd`, `kd`, `srrl`

All possible command line arguments can be found by running `python distill.py --help`.

## Step 2b: Synthetic Distill

First, we have to generate the synthetic data. To do this, edit and run `scripts/generate.sh`.
This will save the synthetic data in the `syn_data_path` from `paths.json`.

Then, edit and run `scripts/gen_distill.sh`. It works similar to `distill.sh`, with the addition of the following options:

All possible command line arguments can be found by running `python generated_distill.py --help`.

* `--synset_size` chooses how much of the synthetic dataset we wish to use as a fraction of the training set size. Options: `1x`, `2x` (Note: enough synthetic images must be generated for these to work correctly.)
* `--aug` enables image augmentations
* `--aug_mode` chooses whether to apply a Singular or Multiple augmentations. Options: `S`, `M`

## Logging

Weights and Biases (wandb) integration is included for all scripts above. Assuming you have `wandb` set up on your machine, simply add the `--logging` flag to each script.

## Timing

We have included a convenient `--timing` flag for all scripts that will run the task for one epoch and report how long it took.

## Citation

```
```
