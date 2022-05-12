# BMI543/643 Machine Learning  --  Neural Network Tutorial with `pytorch` 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nathanieljevans/bmi543_nn_tutorial/HEAD)

## create and activate conda environment 

```bash 

$ conda env create -f environment.yml
$ conda activate bmi543
(bmi543) $ python -m ipykernel install --user --name bmi543 --display-name "bmi543"
(bmi543) $ jupyter notebook 

```

Please run `cell 28` prior to class on Tuesday to download the CIFAR10 dataset. 

`cell 28` should look like this: 

```python
# download the CIFAR10 dataset : https://www.cs.toronto.edu/~kriz/cifar.html
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10('./data/', download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10('./data/', download=False, train=False, transform=transform)
```
