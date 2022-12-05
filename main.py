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
import torchvision.transforms as trans
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from imutils import paths
from torchsummary import summary
from typing import List, Union

#For optimization of hyperparameters
#pip install ray[tune]
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
        model = os.path.join(base, "output")
        # define the path of the data analysis
        mydata = os.path.join(base, "mydata")
        # load the image and mask filepaths in a sorted manner
        list_image = sorted(list(paths.list_images(image)))
        list_mask = sorted(list(paths.list_images(mask)))
        list_test = sorted(list(paths.list_images(test)))

    @dataclass
    class Image:
        channels: int = None
        height: int = None
        width: int = None
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
        classes: int = None
        lr: float = 0.01
        epochs: int = 3
        batch_size: int = 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion: str = 'Dice'
        optimizer: str = 'Adam'


@dataclass
class Transform:
    """
    This class allows to do the main transformations on the images and mask dataset
    Using torchvision.transforms
    For
    """

    def __init__(self, param: Param = None):
        self.param_image = param.Image
        self.main()

    def main(self):
        self.mask = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((self.param_image.height, self.param_image.width), interpolation=cv2.INTER_NEAREST),
            trans.ToTensor()
        ])
        if not self.param_image.mean is None:
            self.image = trans.Compose([
                trans.ToPILImage(),
                trans.Resize((self.param_image.height, self.param_image.width), interpolation=cv2.INTER_NEAREST),
                trans.ToTensor(),
                trans.Normalize(self.param_image.mean, self.param_image.std)
            ])
            self.denormalize = trans.Compose([
                trans.Normalize(mean=[0., 0., 0.], std=1 / self.param_image.std),
                trans.Normalize(mean=-self.param_image.mean, std=[1., 1., 1.])
            ])


class SegmentationDataset(Dataset):
    def __init__(self, image_paths: List[str] = None, mask_paths: List[str] = None, transform_image=None,
                 transform_mask=None):
        """
        Store the image/mask filepaths, and transformers
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, index):
        # load the image from local disk and swap its channels from BGR to RGB
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load the mask associated in grayscale mode (if in training or validation, otherwise mask is just zero)
        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[index], 0)
        else:
            mask = torch.zeros(1)
        # apply the transformations to both image and its mask if not None
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask * 255).int()  # as using ToTensor() convert data to [0,1] and we want mask pixels into [0,255]
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
    def split():
        """
        Partition the data into training and validation splits using the parameters (split, list_image,etc) from Param
        """
        print(f'Partition the data into training and validation splits using a split of {Param.Model.val_split}')
        return train_test_split(Param.Path.list_image, Param.Path.list_mask, test_size=Param.Model.val_split,
                                random_state=42)

    def listloading(self):
        """
        Load the raw training images in a list in order to analyse the shapes
        """
        shapes_train_images_path = os.path.join(Param.Path.mydata, 'shapes_train_images.pkl')
        try:
            shapes_train_images = pd.read_pickle(shapes_train_images_path)
        except FileNotFoundError:
            train_images = [cv2.imread(path) for path in self.path_train_images]
            shapes_train_images = pd.DataFrame([image.shape for image in train_images],
                                               columns=['height', 'width', 'channel'])
            shapes_train_images.to_pickle(shapes_train_images_path)
        return shapes_train_images

    @staticmethod
    def resize(measure, threshold=200):
        """
        Allow to find the factor which decrease the size (width or height) until a certain threshold
        Indeed too big size requires to much computation for the training
        Then max width and height = threshold = 200
        """
        factor = 1
        while measure / factor > threshold:
            factor += 1
        return 1 / factor

    @staticmethod
    def closest_number(n, m=32):
        """
        Expected image height and width must be divisible by 32 therefore this function allows to find the
        closest number divisible by 32
        """
        q = int(n / m)
        n1 = m * q
        if ((n * m) > 0):
            n2 = (m * (q + 1))
        else:
            n2 = (m * (q - 1))
        if (abs(n - n1) < abs(n - n2)):
            return n1
        return n2

    def params_image(self):
        """
        Get the parameters of the training images, resize the parameters shapes to store them into Param
        """
        shapes_train_images = self.listloading()
        shapes_train_images['aspect_ratio'] = round(shapes_train_images["width"] / shapes_train_images["height"], 2)
        min_height = shapes_train_images['height'].min()  # 4000
        min_width = shapes_train_images['width'].min()  # 3000
        min_channel = shapes_train_images['channel'].min()
        factor = self.resize(min_height)
        Param.Image.height = self.closest_number(int(min_height * factor))
        Param.Image.width = self.closest_number(int(min_width * factor))
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
        Create the training set and loader without normalize transformation and with batch_size = len(all_data)
        in order to get the mean and std of the training images for further normalize transformation
        """
        transform = Transform(Param)

        self.trainset = SegmentationDataset(image_paths=self.path_train_images, mask_paths=self.path_train_masks,
                                            transform_image=transform.mask, transform_mask=transform.mask)
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

    def __init__(self, param:Param):
        self.param = param
        self.path_train_images, self.path_val_images, self.path_train_masks, self.path_val_masks = Analysis.split()
        self.path_test_images = self.param.Path.list_test
        self.main()

    def set_dataset(self):
        """
        Instanciation of trainset and valset
        Put the transform variable in global in order to reuse it through all the script
        """
        global transform
        transform = Transform(self.param)
        print('The training and validation images set are now transformed with Normalization')
        self.trainset = SegmentationDataset(image_paths=self.path_train_images, mask_paths=self.path_train_masks,
                                            transform_image=transform.image, transform_mask=transform.mask)
        self.valset = SegmentationDataset(image_paths=self.path_val_images, mask_paths=self.path_val_masks,
                                          transform_image=transform.image, transform_mask=transform.mask)
        self.testset = SegmentationDataset(image_paths=self.path_test_images, transform_image=transform.image)

    def set_dataloader(self):
        """
        Instanciation of train dataloader and val dataloader
        """
        self.train_dataloader = DataLoader(self.trainset, shuffle=True,
                                           batch_size=Param.Model.batch_size)#, num_workers=os.cpu_count())
        self.val_dataloader = DataLoader(self.valset, shuffle=False,
                                         batch_size=Param.Model.batch_size)#, num_workers=os.cpu_count())
        self.test_dataloader = DataLoader(self.testset, shuffle=False,
                                          batch_size=len(self.testset))#, num_workers=os.cpu_count())

    def original_size(self):
        """
        Methods allowing to save the original size of each testing image in order to submit predicted mask having the same size
        """
        path = os.path.join(Param.Path.mydata, 'size_test_images.pkl')
        print('Storing the original test image size for the prediction submission')
        try:
            self.test_size = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            self.test_size = [cv2.imread(path).shape for path in self.path_test_images]
            pickle.dump(self.test_size, open(path, 'wb'))

    def main(self):
        print('*' * 50)
        print('*' * 20 + 'DataInstanciation' + '*' * 20)
        self.set_dataset()
        self.set_dataloader()
        self.original_size()
        """
        print('Plotting examples of X - Y from training set')
        for i in range(5):
            image, mask = self.trainset[i]
            Utils.plot(image, mask)
        """
        print('*' * 50)


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


class FirstEarlyStopping:
    """
    This class allow to :
    Stop the training when the validation loss increase while training loss decrease
    Useful to avoid overfitting
    """

    def __init__(
            self,
            tolerance: int = 10):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.previous_train_loss = 1e5
        self.previous_val_loss = 1e5

    def __call__(self, train_loss, val_loss):
        if ((train_loss <= self.previous_train_loss) and (val_loss >= self.previous_val_loss)):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        self.previous_train_loss = train_loss
        self.previous_val_loss = val_loss


class SecondEarlyStopping:
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
    def __init__(self, model, param: Param, data: Data, wandb=None, save: bool = True, name: str = None,
                 verbose: bool = True, hyperopt: bool = True):
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

    def __train_model(self):
        # set the model in training mode
        self.model.train()
        train_losses, train_accuracy = [], []
        # loop over the training set
        for img_batch, lbl_batch in self.data.train_dataloader:
            # send input to device
            lbl_batch = Utils.squeeze_generic(lbl_batch, [0])
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
            lbl_batch = Utils.squeeze_generic(lbl_batch, [0])
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

    def __compute_early_stopping(self, epoch, my_first_es, my_snd_es, train_loss_mean, val_loss_mean):
        break_it = False
        my_first_es(train_loss_mean, val_loss_mean)
        if my_first_es.early_stop:
            print(f'At epoch {epoch}, the first early stopping tolerance = {my_first_es.tolerance} has been reached,'
                  f' the model is overfitting -> stop it')
            break_it = True
        my_snd_es(val_loss_mean)
        if my_snd_es.early_stop:
            print(f'At epoch {epoch}, the second early stopping tolerance = {my_snd_es.tolerance} has been reached,'
                  f' the loss of validation is not decreasing anymore -> stop it')
            break_it = True
        return break_it

    def __compute_verbose_train(self, epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean, val_acc_mean):
        print(
            "Epoch [{}] took {:.2f}s | train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, time.time() - start_time, train_loss_mean, train_acc_mean, val_loss_mean, val_acc_mean))
        self.wandb.log({"train_loss": train_loss_mean, "val_loss": val_loss_mean, "val_acc": val_acc_mean})

    def fit(self, checkpoint_dir=None):
        if not self.hyperopt:
            my_first_es = FirstEarlyStopping()
            my_snd_es = SecondEarlyStopping()

        for epoch in range(1, self.param_model.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model()
            val_loss_mean, val_acc_mean, _ = self.__evaluate_model(self.data.val_dataloader)

            if not self.hyperopt:
                break_it = self.__compute_early_stopping(epoch, my_first_es, my_snd_es, train_loss_mean, val_loss_mean)
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
                if not self.hyperopt:
                    torch.save(self.model.state_dict(), f'{self.path}/model_{epoch}.pt')
                else:
                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

        if not self.hyperopt:
            if break_it:
                self.best_epoch = epoch - my_first_es.tolerance
            else:
                self.best_epoch = epoch
        self.train_loss = train_loss_mean
        self.val_loss = val_loss_mean

    def predict(self, data: Union[DataLoader, Dataset] = None, index_image: int = None):
        """
        Allows to predict all the masks from a dataloader or just one mask for a specific image from a dataset
        """
        assert (isinstance(data, Dataset) and isinstance(index_image, int))
        if not self.hyperopt:
            if self.best_epoch is not None:
                epoch = self.best_epoch
            else:
                epoch = Utils.find_last_epoch(self.path)
            self.model = Utils.load_model(self.model, epoch, self.path, self.param_model.device)
        else:
            self.model = Utils.load_model(self.model, None, self.path, self.param_model.device)
        loss_mean, acc_mean, predictions = self.__evaluate_model(data, index_image)
        if self.verbose:
            if loss_mean is not None:
                print(f'Prediction on index {index_image} validation - loss: {loss_mean} | accuracy: {acc_mean}')
                return predictions
            else:
                """Resize the output to match the original mask dimensions"""
                resized_predictions = []
                for (pred_mask, original_size) in zip(predictions, self.data.test_size):
                    resized_predictions.append(
                        F.interpolate(pred_mask, (original_size.shape[0], original_size.shape[1])))
                return resized_predictions


class HyperOptImageSegmentation:
    """
    This class hyper-optimize the parameters on the validation set
    """

    def __init__(
            self,
            trials_hopt: int = 2,  # Number of Trials of hyper optimization
            tolerance_es: int = 1,  # Tolerance of early stopping
    ):
        self.trials_hopt = trials_hopt
        self.tolerance_es = tolerance_es

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
        return ASHAScheduler(metric="val_loss", mode='min', max_t=Param.Model.epochs, grace_period=1,
                             reduction_factor=2)

    @property
    def reporter(self):
        """
        Reporter of the hyper-optimization results on the validation sample
        """
        return CLIReporter(
            metric_columns=["train_loss", "train_acc", "val_loss", "val_acc"])

    def instantiation(self, config, model, param, save: bool = True, name: str = None):
        param.Model.lr = config['lr']
        param.Model.batch_size = config['batch_size']
        param.Model.optimizer = config['optimizer']
        data = Data(param)
        model = ImageSegmentation(model, param, data, wandb=None, save=save, name=name,
                                  verbose=False, hyperopt=True)
        return data, model

    def fit(self, config, checkpoint_dir=None, model=None, param=None, save: bool = True, name: str = None):
        data, model = self.instantiation(config, model, param, save, name)
        model.fit(checkpoint_dir)

    def get_best_trial(self, analysis=None):
        """
        Method that allows to find the trial which minimize the validation loss
        """
        best_trial = analysis.get_best_trial("val_loss", "min", "last")
        config = best_trial.config
        loss = best_trial.last_result["val_loss"]
        acc = best_trial.last_result["val_acc"]
        checkpoint = best_trial.checkpoint.value

        print(f"Best trial config: {config}")
        print(f"Best trial final validation loss: {loss}")
        print(f"Best trial final validation accuracy: {acc}")
        return config, checkpoint

    def call(self, model, param, save: bool = True, name: str = None):
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
                tune.with_parameters(self.fit, model=model, param=param, save=save, name=name),
                num_samples=self.trials_hopt,
                config=self.search_space,
                scheduler=self.scheduler,
                progress_reporter=self.reporter,
                name=name
            )
            pickle.dump(analysis, open(path, 'wb'))
        """
        Testing with the best parameters
        """
        config, checkpoint = self.get_best_trial(analysis)
        print(checkpoint)
        data, model = self.instantiation(config, model, param, save, name)
        model.path = checkpoint
        pred_mask_test = model.predict(data.test_dataloader)
        # saving these final predictions
        pickle.dump(pred_mask_test, open(f'{model.path}/final_predictions.pkl', 'wb'))


class Utils:
    """
    This class allows to contains all the utility function that will be needed in this project
    """

    @staticmethod
    def transform_image(image):
        # undo the transformation of the image to plot it
        image = transform.denormalize(image)
        image = (image.permute(1, 2, 0)).numpy()
        return image

    @staticmethod
    def transform_mask(mask):
        return (mask.permute(1, 2, 0)).numpy()

    @staticmethod
    def plot(image, mask):
        image = Utils.transform_image(image)
        mask = Utils.transform_mask(mask)
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
        correct = torch.eq(pred_mask, mask).int()
        return float(correct.sum()) / float(correct.numel())

    @staticmethod
    def find_last_epoch(path):
        return int([f for f in os.listdir(path)][-1].split('_')[1].split('.')[0])

    @staticmethod
    def load_model(model, epoch, path, device):
        path_ = f'{path}/model_{epoch}.pt' if epoch is None else path
        if device.type == 'cpu':
            model.load_state_dict(torch.load(path_, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path_))
        return model

def main(model, param, name):
    data = Data(param)
    Utils.wandb_connect()
    hyperparams = {"Batch size": Param.Model.batch_size,
                   "Learning rate": Param.Model.lr,
                   "Optimizer": Param.Model.optimizer}
    wandb.init(config=hyperparams, project="DL_project", name=name)

    # Instanciation of the model
    Net = ImageSegmentation(model, param, data, wandb, name=name)
    # Fitting the model (training + validation)
    Net.fit()
    # Doing prediction on the first image of the validation
    pred_mask_val = Net.predict(data.valset, index_image=0)
    image, mask = data.valset[0]
    Utils.plot_pred(image, mask, pred_mask_val[0])
    # Doing the final predictions on the testing dataloader
    # HAS TO RESIZE TO ORIGINAL HEIGHT AND WIDTH
    pred_mask_test = Net.predict(data.test_dataloader)
    # saving these final predictions
    pickle.dump(pred_mask_test, open(f'{Net.path}/final_predictions.pkl', 'wb'))

def main_hyperopt(model, name, param):
    HyperOptNet = HyperOptImageSegmentation()
    HyperOptNet.call(model=model, param=param, name=name, save=True)


if __name__ == '__main__':

    param = Param
    analysis = Analysis()

    """ Trying transfer learning using Segmentation Models library
    Segmentation Models library is widely used in the image segmentation competitions"""
    # https://github.com/qubvel/segmentation_models.pytorch
    import segmentation_models_pytorch as smp

    encoder = 'mobilenet_v2'
    encoder_weights = 'imagenet'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=Param.Model.classes,
        activation=None,
    )
    """ Using this model with hyperoptimisation of parameters (batch_size, optimizer, learning_rate)"""
    main_hyperopt(model, encoder, param)

    """ Using this model without hyperoptimisation"""
    main(model, encoder, param)

    print('end')
