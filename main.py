import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from imutils import paths
from torchsummary import summary
from typing import List, Union

#for better transformations
import albumentations as A

""" Trying transfer learning using Segmentation Models library
Segmentation Models library is widely used in the image segmentation competitions"""
# https://github.com/qubvel/segmentation_models.pytorch
# Notebook example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
import segmentation_models_pytorch as smp

# For optimization of hyperparameters
# pip install ray[tune]
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# from prepare_submission import *


@dataclass
class Param:
    """
    This class of classes allows to instantiate all the parameters needed in this project
    """

    @dataclass
    class Path:
        # base path
        base = os.path.dirname(os.path.abspath(__file__))
        # datset path
        dataset = os.path.join(base, "dataset")
        # define the path to the images and masks dataset
        image = os.path.join(dataset, "train_images")
        mask = os.path.join(dataset, "train_masks")
        test = os.path.join(dataset, "test_images")
        # define the path to the outputs
        model = os.path.join(base, "models")
        # define the path of the data analysis
        mydata = os.path.join(base, "mydata")
        # load the image and mask filepaths in a sorted manner
        list_image = sorted(list(paths.list_images(image)))
        list_mask = sorted(list(paths.list_images(mask)))
        list_test = sorted(list(paths.list_images(test)))

    @dataclass
    class Image:
        channels: int = None
        height: int = 1024 #new height to resize the high resolution image
        width: int = 1024 #new width to resize the high resolution image
        mean: torch.tensor = None
        std: torch.tensor = None
        classes = {0: 'Background',  # each key represents the pixel value of the class associated
                   1: 'Property Roof',
                   2: 'Secondary Structure',
                   3: 'Swimming Pool',
                   4: 'Vehicle',
                   5: 'Grass',
                   6: 'Trees / Shrubs',
                   7: 'Solar Panels',
                   8: 'Chimney',
                   9: 'Street Light',
                   10: 'Window',
                   11: 'Satellite Antenna',
                   12: 'Garbage Bins',
                   13: 'Trampoline',
                   14: 'Road/Highway',
                   15: 'Under Construction / In Progress Status',
                   16: 'Power Lines & Cables',
                   17: 'Water Tank / Oil Tank',
                   18: 'Parking Area - Commercial',
                   19: 'Sports Complex / Arena',
                   20: 'Industrial Site',
                   21: 'Dense Vegetation / Forest',
                   22: 'Water Body',
                   23: 'Flooded',
                   24: 'Boat'}

    @dataclass
    class Model:
        val_split: float = 0.15
        classes: int = 25
        lr: float = 0.001
        epochs: int = 20
        batch_size: int = 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion: str = 'Dice'
        optimizer: str = 'Adam'


class Transform:
    """
    This class allows to do the main transformations, augmentation and preprocessing on the images and mask dataset
    Using torchvision.transforms and abumentation
    """

    def __init__(self, param: Param = None, preprocessing_fn=None):
        self.param_image = param.Image
        self.preprocessing = preprocessing_fn #when using segmentation_models_pytorch models have pretrained encoders,
        # so have to prepare data the same way as during weights pretraining
        self.set_args()

    def set_args(self):
        #common transformation to load and resize image and mask with torchvision.transforms
        self.common = T.Compose([
            T.ToPILImage(),
            T.Resize((self.param_image.height, self.param_image.width), interpolation=cv2.INTER_NEAREST),
            T.ToTensor()
        ])
        #instanciation of training augmentation with albumentation
        self.train_augmentation = A.Compose([
            A.Resize(self.param_image.height, self.param_image.width, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomCrop(int(self.param_image.height/2), int(self.param_image.width/2)),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0,0.5),(0,0.5)),
            A.GaussNoise()
        ])
        #instanciation of validation augmentation with albumentation
        self.val_augmentation = A.Compose([
            A.Resize(self.param_image.height, self.param_image.width, interpolation=cv2.INTER_NEAREST)])
        #instanciation of prepocessing if needed
        if not self.preprocessing is None:
            self.preprocessing = A.Compose([A.Lambda(image=self.preprocessing)])
        #instanciation od normalization and denorm
        if not self.param_image.mean is None:
            #normalize an image
            self.normalize = T.Compose([
                T.ToTensor(),
                T.Normalize(self.param_image.mean, self.param_image.std)
            ])
            #denormalize an image
            self.denormalize = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0., 0., 0.], std=1 / self.param_image.std),
                T.Normalize(mean=-self.param_image.mean, std=[1., 1., 1.])
            ])


class SegmentationDataset(Dataset):
    def __init__(self, image_paths: List[str] = None, mask_paths: List[str] = None, common_transform=None,
                 augmentation=None, preprocessing=None, normalize_transform=None):
        """
        Store the image/mask filepaths, and transformers
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.common_transform = common_transform
        self.augmentation = augmentation
        self.normalize_transform = normalize_transform
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        # load the image from local disk and swap its channels from BGR to RGB
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load the mask associated in grayscale mode (if in training or validation, otherwise mask is just zero as unknown)
        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[index], 0)
        else:
            mask = torch.zeros(1)
        if self.common_transform:
            mask, image = self.common_transform(mask), self.common_transform(image)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.normalize_transform:
            image = self.normalize_transform(image)
        if isinstance(mask, np.ndarray): mask = torch.from_numpy(mask).long()
        return image, mask

    def __len__(self):
        # return the total number of data samples
        return len(self.image_paths)


class Analysis:
    """
    This class allows to do the main analysis of the input data
    """

    def __init__(self):
        self.main()

    @staticmethod
    def split(param: Param = Param, verbose: bool = True):
        """
        Partition the data into training and validation splits using the parameters (split, list_image,etc) from Param
        """
        if verbose: (f'Partition the data into training and validation splits using a split of {param.Model.val_split}')
        return train_test_split(param.Path.list_image, param.Path.list_mask, test_size=param.Model.val_split,
                                random_state=42)

    def listloading(self):
        """
        Load the raw training images in a list in order to analyse the shapes
        """
        shapes_train_images_path = os.path.join(Param.Path.mydata, f'shapes_train_images_{Param.Image.height}.pkl')
        try:
            shapes_train_images = pd.read_pickle(shapes_train_images_path)
        except FileNotFoundError:
            train_images = [cv2.imread(path) for path in self.path_train_images]
            shapes_train_images = pd.DataFrame([image.shape for image in train_images],
                                               columns=['height', 'width', 'channel'])
            shapes_train_images.to_pickle(shapes_train_images_path)
        return shapes_train_images

    def params_image(self):
        """
        Get the parameters of the training images, resize the parameters shapes to store them into Param
        """
        shapes_train_images = self.listloading()
        shapes_train_images['aspect_ratio'] = round(shapes_train_images["width"] / shapes_train_images["height"], 2)
        min_height = shapes_train_images['height'].min()  # 4000
        min_width = shapes_train_images['width'].min()  # 3000
        min_channel = shapes_train_images['channel'].min()
        Param.Image.channels = min_channel
        print(f'The minimum height and width of training images are {(min_height, min_width)} \nAll images '
              f'have been resize to {(Param.Image.height, Param.Image.width)} in order to avoid computational error')

    @staticmethod
    def get_metrics(loader):
        """
        Allows to compute the mean, std of the training images as well as looking for
        the number of classes in the training masks
        """
        for images, masks in loader:
            print(f'The training images (aka X) has a shape of {images.shape}')
            print(f'The training masks (aka y) has the same shape: {masks.shape}')
            min = torch.amin(images, dim=(0, 2, 3))
            max = torch.amax(images, dim=(0, 2, 3))
            print(f'The training images and mask are scaled per channel in {(min, max)}')
            mean = torch.mean(images, dim=[0, 2, 3])
            std = torch.std(images, dim=[0, 2, 3])
            print(f'The training images have a mean and std per channel of {(mean, std)}')
            classes = torch.unique(masks)
            classes_count = (masks).unique(return_counts=True)
            classes_count = pd.DataFrame(classes_count[1].numpy(), index=list(Param.Image.classes.values()),
                                         columns=['Count values'])
            print(f'The repartition of class in training is: \n{classes_count}')
        return mean, std, len(classes)

    def set_metrics(self):
        """
        Create the training set and loader with common transformation and with batch_size = len(all_data)
        in order to get the mean and std of the training images for further normalize transformation
        """
        transform = Transform(Param)

        self.trainset = SegmentationDataset(image_paths=self.path_train_images, mask_paths=self.path_train_masks,
                                            common_transform=transform.common)
        self.train_dataloader = DataLoader(self.trainset, shuffle=True,
                                           batch_size=len(self.trainset), num_workers=os.cpu_count())
        Param.Image.mean, Param.Image.std, Param.Model.classes = self.get_metrics(self.train_dataloader)
        print(f'The number of classes is {Param.Model.classes}')

    def main(self):
        print('*' * 50)
        print('*' * 20 + 'DataAnalysis' + '*' * 20)
        self.path_train_images, self.path_val_images, self.path_train_masks, self.path_val_masks = self.split()
        print('')
        self.listloading()
        self.params_image()
        self.set_metrics()
        print('*' * 50)


class Data:
    """
    This class allows to create the true datasets and dataloaders for both training and validation
    """

    def __init__(self, param: Param, transform: Transform, verbose: bool = True):
        self.param = param
        self.transform = transform
        self.verbose = verbose
        self.path_train_images, self.path_val_images, self.path_train_masks, self.path_val_masks = Analysis.split(self.param, self.verbose)
        self.path_test_images = self.param.Path.list_test
        self.main()

    def set_dataset(self):
        """
        Instanciation of trainset with augmentation, preprocessing and normalization transformation
        And the val set
        """
        if self.verbose: print('The training set is now augmented and transformed with normalization'
                               'The validation and testing set is just transformed with normalization')
        self.trainset = SegmentationDataset(image_paths=self.path_train_images, mask_paths=self.path_train_masks,
                                            augmentation=self.transform.train_augmentation,
                                            normalize_transform=self.transform.normalize,
                                            preprocessing=self.transform.preprocessing)
        self.valset = SegmentationDataset(image_paths=self.path_val_images, mask_paths=self.path_val_masks,
                                          augmentation=self.transform.val_augmentation,
                                          normalize_transform=self.transform.normalize,
                                          preprocessing=self.transform.preprocessing)
        self.testset = SegmentationDataset(image_paths=self.path_test_images,
                                           augmentation=self.transform.val_augmentation,
                                           normalize_transform=self.transform.normalize,
                                           preprocessing=self.transform.preprocessing)

    def set_dataloader(self):
        """
        Instanciation of train dataloader and val dataloader
        """
        self.train_dataloader = DataLoader(self.trainset, shuffle=True,
                                           batch_size=Param.Model.batch_size)  # , num_workers=os.cpu_count())
        self.val_dataloader = DataLoader(self.valset, shuffle=False,
                                         batch_size=Param.Model.batch_size)  # , num_workers=os.cpu_count())
        self.test_dataloader = DataLoader(self.testset, shuffle=False,
                                          batch_size=len(self.testset))  # , num_workers=os.cpu_count())

    def original_size(self):
        """
        Methods allowing to save the original size of each testing image in order to submit predicted mask having the same size
        """
        path = os.path.join(Param.Path.mydata, 'size_test_images.pkl')
        if self.verbose: print('Storing the original test image size for the prediction submission')
        try:
            self.test_size = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            self.test_size = [cv2.imread(path).shape for path in self.path_test_images]
            pickle.dump(self.test_size, open(path, 'wb'))

    def main(self):
        if self.verbose:
            print('*' * 50)
            print('*' * 20 + 'DataInstanciation' + '*' * 20)
        self.set_dataset()
        self.set_dataloader()
        self.original_size()
        if self.verbose:
            print('Plotting examples of X - Y from training set')
            for i in range(5):
                image, mask = self.trainset[i]
                Utils.plot(image, mask)
        if self.verbose: print('*' * 50)


class DiceLoss(nn.Module):
    """
    This class allows to compute the DiceLoss
    The leaderboard score is the mean of the Dice coefficients for each (Image, Label) pair in the test set.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = Utils.activation(inputs)
        targets = F.one_hot(targets, Param.Model.classes).permute(0, 3, 1, 2).contiguous()
        intersection = torch.sum(inputs * targets, (1, 2, 3))
        cardinality = torch.sum(inputs + targets, (1, 2, 3))
        dice_score = 2. * intersection / (cardinality)
        return torch.mean(1. - dice_score)


class EarlyStopping:
    """
    This class allow to :
    Stop the training when the validation loss doesn't decrease anymore
    Useful to reduce the number of epoch
    """

    def __init__(
            self,
            tolerance: int = 10):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = 1e5

    def __call__(self, val_loss):
        if round(val_loss, 3) >= round(self.best_val_loss, 3):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0


class ImageSegmentation:
    """
    This class allows to compute the main methods to fit a model and predict output on testing set
    """

    def __init__(self, model, param: Param, data: Data, wandb=None, save: bool = None, name: str = None,
                 verbose: bool = None, hyperopt: bool = None):
        self.param_model = param.Model
        self.data = data
        self.wandb = wandb
        self.save = save
        self.verbose = verbose
        self.hyperopt = hyperopt
        self.criterion = DiceLoss()
        self.best_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.path = None
        self.__instantiate_model(model)
        self.__instantiate_save(name, param)
        self.__instantiate_optimizer()

    def __instantiate_model(self, model):
        self.model = Utils.to_device(model, self.param_model.device)
        if not self.hyperopt: summary(model, next(iter(self.data.train_dataloader))[0].shape[1:])

    def __instantiate_save(self, name, param):
        if self.save:
            assert (name is not None)
            self.path = os.path.join(param.Path.model, name)
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def __instantiate_optimizer(self):
        if self.param_model.optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.param_model.lr)
        elif self.param_model.optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.param_model.lr)
        else:
            raise ValueError(f'{self.param_model.optimizer} has not been implemented')

    @staticmethod
    def __transf_batch(img_batch, lbl_batch):
        img_batch = img_batch.to(torch.float32)
        lbl_batch = Utils.squeeze_generic(lbl_batch, [0])
        return img_batch, lbl_batch

    def __train_model(self):
        # set the model in training mode
        self.model.train()
        train_losses, train_accuracy = [], []
        # loop over the training set
        for img_batch, lbl_batch in self.data.train_dataloader:
            # send input to device
            img_batch, lbl_batch = self.__transf_batch(img_batch, lbl_batch)
            img_batch, lbl_batch = Utils.to_device((img_batch, lbl_batch), self.param_model.device)
            # zero out previous accumulated gradients
            self.optimizer.zero_grad()
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(img_batch)
            train_accuracy.append(Utils.accuracy(outputs, lbl_batch))
            loss = self.criterion(outputs, lbl_batch.long())
            # perform backpropagation and update model parameters
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
        return np.mean(train_losses), np.mean(train_accuracy)

    @torch.no_grad()
    def __evaluate_model(self, data: Union[DataLoader, Dataset], index: int = None):
        # Allows to evaluate on dataloader or predict on datalaoder and dataset (for a given index)
        # set the model in eval mode
        self.model.eval()
        losses, accuracy, predictions = [], [], []
        # loop over the validation set
        if isinstance(data, Dataset): data = [list(data[index])]
        for batch in data:
            # send input to device
            img_batch, lbl_batch = batch
            img_batch, lbl_batch = self.__transf_batch(img_batch, lbl_batch)
            if len(img_batch.shape) == 3: img_batch = img_batch.unsqueeze(0)
            img_batch, lbl_batch = Utils.to_device((img_batch, lbl_batch), self.param_model.device)
            outputs = self.model(img_batch)
            predictions.append(Utils.predict(outputs))
            # if on validation calculate the loss
            if torch.count_nonzero(lbl_batch).item() != 0:
                accuracy.append(Utils.accuracy(outputs, lbl_batch))
                loss = self.criterion(outputs, lbl_batch.long())
                losses.append(loss.item())

        if torch.count_nonzero(lbl_batch).item() != 0:
            return np.mean(losses), np.mean(accuracy), predictions
        else:
            return None, None, predictions

    def __compute_early_stopping(self, epoch, my_es, val_loss_mean):
        break_it = False
        my_es(val_loss_mean)
        if my_es.early_stop:
            print(f'At epoch {epoch}, the second early stopping tolerance = {my_es.tolerance} has been reached,'
                  f' the loss of validation is not decreasing anymore -> stop it')
            break_it = True
        return break_it

    def __compute_verbose_train(self, epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean, val_acc_mean):
        print(
            "Epoch [{}] took {:.2f}s | train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, time.time() - start_time, train_loss_mean, train_acc_mean, val_loss_mean, val_acc_mean))
        self.wandb.log({"train_loss": train_loss_mean, "val_loss": val_loss_mean, "val_acc": val_acc_mean})

    def fit(self):
        if not self.hyperopt:
            my_es = EarlyStopping()

        for epoch in range(1, self.param_model.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model()
            val_loss_mean, val_acc_mean, _ = self.__evaluate_model(self.data.val_dataloader)

            if not self.hyperopt:
                break_it = self.__compute_early_stopping(epoch, my_es, val_loss_mean)
                if break_it:
                    break
                if self.verbose:
                    self.__compute_verbose_train(epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean,
                                                 val_acc_mean)
            else:
                # Send the current validation loss and accuration back to Tune for the hyperopt
                # Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the best results.
                tune.report(train_loss=train_loss_mean, train_acc=train_acc_mean, val_loss=val_loss_mean, val_acc=val_acc_mean)

            if self.save:
                torch.save(self.model.state_dict(), f'{self.path}/model_{epoch}.pt')

        if not self.hyperopt:
            if break_it:
                self.best_epoch = epoch - my_es.tolerance
            else:
                self.best_epoch = epoch
        self.train_loss = train_loss_mean
        self.val_loss = val_loss_mean

    def predict(self, data: Union[DataLoader, Dataset] = None, index_image: int = None):
        """
        Allows to predict all the masks from a dataloader or just one mask for a specific image from a dataset
        """
        if self.best_epoch is not None:
            epoch = self.best_epoch
        else:
            epoch = Utils.find_last_epoch(self.path)
        self.model = Utils.load_model(self.model, epoch, self.path, self.param_model.device)
        loss_mean, acc_mean, predictions = self.__evaluate_model(data, index_image)
        if self.verbose:
            if loss_mean is not None:
                print(f'Prediction on index {index_image} validation - loss: {loss_mean} | accuracy: {acc_mean}')
                return predictions
            else:
                """Resize the output to match the original mask dimensions"""
                """This method is not working we have to find something else"""
                resized_predictions = []
                for (pred_mask, original_size) in zip(predictions, self.data.test_size):
                    resized_predictions.append(
                        F.interpolate(pred_mask, (original_size[0], original_size[1])))
                return resized_predictions


@dataclass
class HPanalysis:
    """
    This class allows to get the wanted attributes from the analysis of the tune.run method
    Indeed saving (with a pickle) directly the analysis creates error when using other laptop with different paths
    """
    trial_dataframes: dict
    results: dict


class HyperOptImageSegmentation:
    """
    This class hyper-optimize the parameters on the validation set
    """

    def __init__(
            self,
            trials_hopt: int = 10,  # Number of Trials of hyper optimization
            tolerance_es: int = 10, # Tolerance of early stopping
            epochs: int = None
    ):
        self.trials_hopt = trials_hopt
        self.tolerance_es = tolerance_es
        self.epochs = epochs
        self.metric = "val_loss"

    @property
    def search_space(self):
        """
        Defining the search space of the hyper-parameters
        """
        return {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([2, 4, 8, 16]),
            "optimizer": tune.choice(['Adam', 'SGD']),
        }

    @property
    def scheduler(self):
        """
        ASHAScheduler terminate bad performing trials early
        Uses a metric as the training result objective value attribute
        """
        return ASHAScheduler(metric=self.metric, mode='min', max_t=self.epochs, grace_period=1,
                             reduction_factor=2)

    @property
    def reporter(self):
        """
        Reporter of the hyper-optimization results on the validation sample
        """
        return CLIReporter(
            metric_columns=["train_loss", "train_acc", "val_loss", "val_acc"])

    def instantiation(self, config, model, param, save: bool = False, name: str = None, transform=None,
                      verbose: bool=False, hyperopt: bool = True, wandb=None):
        """
        Instantiation of the model + data with the parameters given by the config
        save=False, Not saving each model tries by the hyperopt, takes too much memory
        Will just save the best model
        """
        param.Model.lr = config['lr']
        param.Model.batch_size = config['batch_size']
        param.Model.optimizer = config['optimizer']
        if 'epoch' in config: param.Model.epoch = config["epoch"]
        data = Data(param, transform, verbose=False)
        model = ImageSegmentation(model, param, data, wandb=wandb, save=save, name=name, verbose=verbose, hyperopt=hyperopt)
        return data, model

    def fit(self, config, model=None, param=None, save: bool = False, name: str = None, transform=None):
        data, model = self.instantiation(config, model, param, save, name, transform)
        model.fit()

    def find_epoch(self, analysis, metric, path):
        """Scrap the dataframe of results to find the best epoch"""
        epoch = analysis.trial_dataframes[path][metric].iloc[self.tolerance_es:].idxmin()
        return epoch, analysis.trial_dataframes[path][metric].loc[epoch]

    def loop_trials(self, analysis, metric, path_trials, name_trials):
        best_metric = 1
        best_trial, best_config, best_epoch = None, {}, 0
        for path, name in zip(path_trials, name_trials):
            try:
                epoch, metric_result = self.find_epoch(analysis, metric, path)
            except ValueError:
                epoch = 0  # this trial has not enough epochs for the early stopping
                metric_result = 1
            if metric_result < best_metric:
                best_metric = metric_result
                best_epoch = epoch
                best_trial = name
                best_config = analysis.results[name]['config']
        return best_metric, best_trial, best_config, best_epoch

    def get_best_trial(self, analysis=None):
        """
        Method that allows to browse all the trials of an analysis and in each trials browse all progress (epochs)
        The aim is to find the trial and the epoch which minimize the validation loss
        """
        path_trials = list(analysis.trial_dataframes.keys())
        name_trials = list(analysis.results.keys())
        best_metric, best_trial, best_config, best_epoch = self.loop_trials(analysis, self.metric,
                                                                            path_trials,
                                                                            name_trials)
        best_config["epoch"] = best_epoch

        print(f'Best trial: {best_trial}')
        print(f"Best trial final validation loss: {best_metric}")
        print(f'The configuration of this trial is: {best_config}')
        return best_config

    def main(self, model, param, save: bool = True, name: str = None, transform=None):
        path = os.path.join(param.Path.mydata, f'HyperOpt_{name}.pkl')
        """
        Training and Validation
        Running the package Ray Tune in order to iteratively search for hyperparameters that optimize 
        the validation loss
        """
        try:
            analysis = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            analysis = tune.run(
                tune.with_parameters(self.fit, model=model, param=param, save=save, name=name, transform=transform),
                num_samples=self.trials_hopt,
                config=self.search_space,
                scheduler=self.scheduler,
                progress_reporter=self.reporter,
                name=name
            )
            """Saving the results of he hyperoptimization"""
            analysis = HPanalysis(analysis.trial_dataframes, analysis.results)
            pickle.dump(analysis, open(path, 'wb'))

        """
        Testing with the best parameters
        """
        #getting the configuration
        config = self.get_best_trial(analysis)
        #give a name to the best hyperopt
        name = name + "__".join([str(k)+'_'+str(v) for k, v in config.items()])
        #connect wandb
        Utils.wandb_connect()
        hyperparams = {"Batch size": config['batch_size'],
                       "Learning rate": config['lr'],
                       "Epochs": config['epoch'],
                       "Optimizer": config['optimizer']}
        wandb.init(config=hyperparams, project="DL_project", name=name)
        #instantiate the model with this config
        data, model = self.instantiation(config, model, param, wandb=wandb, save=True, name=name, hyperopt=False, verbose=True)
        #Fitting of the model + saving it
        model.fit()
        """
        ***********Not working for now************
        ****Have to Resize correctly the output to match the original mask dimensions****
        pred_mask_test = model.predict(data.test_dataloader)
        # saving these final predictions
        pickle.dump(pred_mask_test, open(f'{model.path}/final_predictions.pkl', 'wb'))
        """


class Utils:
    """
    This class allows to contains all the utility function that will be needed in this project
    """

    @staticmethod
    def transform_image(image: np.ndarray):
        # undo the transformation of the image to plot it
        image = transform.denormalize(image)
        image = (image.permute(1, 2, 0)).numpy()
        return image

    @staticmethod
    def transform_mask(mask: torch.Tensor):
        return mask.numpy()

    @staticmethod
    def plot(image, mask):
        image = Utils.transform_image(image)
        mask = Utils.transform_mask(mask) #Utils.transform_mask(mask)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Image - Mask')
        ax1.imshow(image)
        ax2.imshow(mask)
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_pred(image, mask, pred_mask):
        image = Utils.transform_image(image)
        mask = Utils.transform_mask(mask)
        pred_mask = Utils.transform_mask(pred_mask)
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[1].imshow(mask)
        ax[2].imshow(pred_mask)
        ax[0].set_title("Image")
        ax[1].set_title("Original Mask")
        ax[2].set_title("Predicted Mask")
        fig.tight_layout()
        fig.show()

    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [Utils.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    @staticmethod
    def wandb_connect():
        wandb_api_key = "5dec2451d8f6b13859fed8d9d63cdfcb14b56816"  # here use your API key from WandB interface
        wandb_conx = wandb.login(key=wandb_api_key)
        print(f"Connected to Wandb online interface : {wandb_conx}")

    @staticmethod
    def squeeze_generic(array: Union[np.ndarray, torch.Tensor] = None, axes_to_keep: List[int] = None):
        out_s = [s for i, s in enumerate(array.shape) if i in axes_to_keep or s != 1]
        return array.reshape(out_s)

    @staticmethod
    def activation(outputs):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        return torch.exp(torch.log_softmax(outputs, dim=1))

    @staticmethod
    def predict(outputs):
        return torch.argmax(Utils.activation(outputs), dim=1).cpu().detach()

    @staticmethod
    def accuracy(pred_mask, mask):
        pred_mask = Utils.predict(pred_mask)
        correct = torch.eq(pred_mask, mask.cpu().int()
        return float(correct.sum()) / float(correct.numel())

    @staticmethod
    def find_last_epoch(path):
        return int([f for f in os.listdir(path)][-1].split('_')[1].split('.')[0])

    @staticmethod
    def load_model(model, epoch, path, device):
        path_ = f'{path}/model_{epoch}.pt' if epoch is not None else path
        if device.type == 'cpu':
            model.load_state_dict(torch.load(path_, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path_))
        return model


def main(model, param, name, transform):
    data = Data(param, transform, verbose=True)
    Utils.wandb_connect()
    hyperparams = {"Batch size": Param.Model.batch_size,
                   "Learning rate": Param.Model.lr,
                   "Optimizer": Param.Model.optimizer}
    wandb.init(config=hyperparams, project="DL_project", name=name)

    # Instanciation of the model
    Net = ImageSegmentation(model, param, data, wandb, name=name, save=True, verbose=True, hyperopt=False)
    # Fitting the model (training + validation)
    Net.fit()
    # Doing prediction on the first image of the validation
    pred_mask_val = Net.predict(data.valset, index_image=0)
    image, mask = data.valset[0]
    Utils.plot_pred(image, mask, pred_mask_val[0])
    """
    # Doing the final predictions on the testing dataloader
    # HAS TO RESIZE TO ORIGINAL HEIGHT AND WIDTH
    pred_mask_test = Net.predict(data.test_dataloader)
    # saving these final predictions
    pickle.dump(pred_mask_test, open(f'{Net.path}/final_predictions.pkl', 'wb'))
    """


def main_hyperopt(model, param, name, transform):
    HyperOptNet = HyperOptImageSegmentation(epochs=param.Model.epochs)
    HyperOptNet.main(model=model, param=param, name=name, transform=transform)


if __name__ == '__main__':
    param = Param
    analysis = Analysis()

    #choose encoder and pretrained weights
    encoder = 'mobilenet_v2'
    encoder_weights = 'imagenet'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=Param.Model.classes,
        activation=None,
    )

    #Gives access by putting in global: the function Utils.transform_image needs transform to denormalize
    global transform
    transform = Transform(param, smp.encoders.get_preprocessing_fn(encoder, encoder_weights))
    """ Using this model with hyperoptimisation of parameters (batch_size, optimizer, learning_rate)"""
    #main_hyperopt(model, param, encoder, transform)


    """ Using this model without hyperoptimisation"""
    main(model, param, encoder, transform)

    print('End')
