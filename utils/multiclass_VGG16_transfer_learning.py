"""
A class for loading the preprocessed data into a pre-trained model and start training and testing.
"""

# pylint: disable=relative-beyond-top-level

import torch
# Import torch.nn which contains all functions necessary to define a convolutional neural network
import torch.nn as nn
import sys

# Generate confusion matrixes
from sklearn.metrics import confusion_matrix

# Generate summary statistics
from .metrics import summary_statistics

class multiclass_VGG16_transfer_learning():
    """
    Parameters
    ----------
    model:                The pre-trained model that is already initialized.
    train_dataloader:     A already defined Python iterable over the training dataset.
    test_dataloader:      A already defined Python iterable over the testing dataset.
    criterion:            The loss function used for training the model.
    optimizer:            The optimizer used for training the model.

    Output
    ------
    running loss:         The running loss when training the model.
    accuracy:             The accuracy of the trained model.

    Created by Danni Chen on 09/27/2021.
    """

    def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer,
                 device = None, writer = None, verbose = False):

        # Auto-select device if none provided.
        if not device:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set if class closed.
        self.closed = False
        # Set if class verbose.
        self.verbose = verbose

        # Assign device
        self.device = device
        # Assign writer
        self.writer = writer

        # train_dataloader attribute 
        self.train_dataloader = train_dataloader
        # test_dataloader attribute
        self.test_dataloader = test_dataloader
        
        self.model = model

        # Auto-assign model to device when setting.
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = criterion

        # Optimizer
        self.optimizer = optimizer

    def close(self):
        self.closed = True
        self.writer.close()
        torch.cuda.empty_cache()

    def check_closed(self):
        if self.closed:
            raise RuntimeError("Cannot perform operations on a closed instance.")

    def save(self, fp):
        self.check_closed()
        torch.save(self.model.state_dict(), fp)

    def model_training(self, numOfEpoch):
        
        # Ensure no resources have been released yet
        self.check_closed()

        pct = 1/numOfEpoch

        metrics_dict = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_mcc': [],
            'train_auc': [],
            'test_loss': [],
            'test_acc': [],
            'test_mcc': [],
            'test_auc': []
        }

        for epoch in range(numOfEpoch): 

            # Train the model.
            self.model.train()

            # initiate running_loss
            running_loss = 0.0 

            # all model predictions in current epoch
            epoch_preds = []
            # all ground-truth labels in current epoch
            all_labels = []

            for ith_batch,batch_data in enumerate(self.train_dataloader): 
                # obtain the images and labels
                img_batch,labels_batch = batch_data['image'],batch_data['label']
                img_batch = img_batch.float().to(self.device)
                labels_batch = labels_batch.to(self.device)

                # torch.Size([batch_size, 3, size_h, size_w])

                # zero the parameter gradients (necessary because .backward() accumulate gradient for each parameter after each iteration)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # feed the img_batch (input) into the network
                # Record the predictions and ground-truth
                outputs = self.model(img_batch)
                epoch_preds.append(outputs)
                all_labels.append(labels_batch)

                # calculate the cross-entropy loss
                loss = self.criterion(outputs, labels_batch)
                # backward
                loss.backward()
                # perform parameter update based on current gradient (stored in .grad) and update rule of SGD
                self.optimizer.step()

                # print statistics
                running_loss += loss.item() # .item() extracts loss values as floats
                if self.verbose:
                    sys.stdout.write('\rBatch loss (batch {:d}/{:d}): {:.3f}'.format(
                        ith_batch + 1, len(self.train_dataloader), loss.item()))
                else:
                    sys.stdout.write('\r{:.2f}% complete.'.format(
                        (epoch + 1 + (ith_batch + 1) / len(self.train_dataloader)) * pct
                    ))

            print()

            # form all label and all pred vectors.
            all_labels = torch.cat(all_labels, dim=0)
            epoch_preds = torch.softmax(torch.cat(epoch_preds, dim=0), dim=1)

            # generate summary training metrics
            train_acc, train_mcc, train_auc = summary_statistics(all_labels.detach().cpu(), epoch_preds.detach().cpu())
            train_loss = running_loss / len(self.train_dataloader)
            
            # report epoch-wide metrics information
            if self.verbose:
                print('Average epoch {:d} training loss/acc: {:.3f}/{:.3f}'.format(epoch + 1, train_loss, train_acc))
                print(confusion_matrix(all_labels.detach().cpu(), epoch_preds.argmax(-1).detach().cpu()))

            # Generate summary testing metrics
            test_acc, test_mcc, test_auc, test_loss = self.model_testing()

            # Update the metrics.
            list(map(lambda x, y: metrics_dict[x].append(y),
                ['epoch', 'train_loss', 'train_acc', 'train_mcc', 'train_auc',
                    'test_loss', 'test_acc', 'test_mcc', 'test_auc'],
                [epoch, train_loss, train_acc, train_mcc, train_auc,
                    test_loss, test_acc, test_mcc, test_auc]
            ))
   
            if self.writer:
                self.writer.add_scalars('Loss', {'training': train_loss, 'testing': test_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'training': train_acc, 'testing': test_acc}, epoch)
                self.writer.add_scalars('MCC', {'training': train_mcc, 'testing': test_mcc}, epoch)
                self.writer.add_scalars('AUC', {'training': train_auc, 'testing': test_auc}, epoch)

        print('Done.')

        return metrics_dict

    def model_testing(self):

        # Ensure no resources have been released yet
        self.check_closed()

        # Set the eval mode flag on the model (not important here but good practice)
        self.model.eval()

        # all model predictions
        epoch_preds = []
        # all ground-truth labels
        all_labels = []
        # running loss!
        running_loss = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader: # iterate through the data
                images, labels = data['image'],data['label']
                images = images.float().to(self.device)
                labels = labels.to(self.device)

                # calculate outputs by running images through the network 
                outputs = self.model(images)
                running_loss += self.criterion(outputs, labels).item()
                epoch_preds.append(outputs)
                all_labels.append(labels)

            # form all label and all pred vectors.
            all_labels = torch.cat(all_labels, dim=0)
            epoch_preds = torch.softmax(torch.cat(epoch_preds, dim=0), dim=1)

            # return summary statistics
            metrics = summary_statistics(all_labels.detach().cpu(), epoch_preds.detach().cpu())
            return list(metrics) + [running_loss / len(self.test_dataloader)]

