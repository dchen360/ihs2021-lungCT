#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
A class for loading the preprocessed data into a pre-trained model and start trainig and testing.
"""
import torch
import torch.optim as optim
# Import torch.nn which contains all functions necessary to define a convolutional neural network
import torch.nn as nn
import sys

class four_category_VGG16_transfer_learning():
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

    def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer):

        # train_dataloader attribute 
        self.train_dataloader = train_dataloader
        # test_dataloader attribute
        self.test_dataloader = test_dataloader
        
        self.model = model
        # Freeze training for all layers
        # To save computation time and that the network would already 
        # be able to extract generic features from the dataset.
        for param in self.model.features.parameters():
            param.requires_grad = False  
        
        # https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
        # Remove the original fully-connected layer (the last layer) and create a new one
        # Newly created modules have requires_grad=True by default
        num_features = self.model.classifier[6].in_features
        classifier_layers = list(self.model.classifier.children())[:-1] # Remove the last layer
        classifier_layers.extend([nn.Linear(in_features = num_features, out_features= 4)]) # Add the new layer with outputting 4 categories
        self.model.classifier = nn.Sequential(*classifier_layers) # Replace the model classifier, Overwriting the original
        
        # Loss function
        self.criterion = criterion

        # Optimizer
        self.optimizer = optimizer

    def model_training(self, numOfEpoch):
        
        self.model.train()

        for epoch in range(numOfEpoch): 

            # initiate running_loss
            running_loss = 0.0 

            for ith_batch,batch_data in enumerate(self.train_dataloader): 
                # obtain the images and labels
                img_batch,labels_batch = batch_data['image'],batch_data['label']

        #         print(img_batch.shape) #torch.Size([64, 3, 512, 512])     

                # zero the parameter gradients (necessary because .backward() accumulate gradient for each parameter after each iteration)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # feed the img_batch (input) into the network
                outputs = self.model(img_batch.float())
        #         print(outputs)
        #         print(outputs.shape) 
        #         print(labels_batch.shape)

                # calculate the cross-entropy loss
                loss = self.criterion(outputs, labels_batch)
                # backward
                loss.backward()
                # perform parameter update based on current gradient (stored in .grad) and update rule of SGD
                self.optimizer.step()

                # print statistics
                running_loss += loss.item() # .item() extracts loss values as floats
                sys.stdout.write('\rRunning loss ({:d}/10): {:.3f}'.format((ith_batch % 10)+1, running_loss))

                # print every 10 mini-batches
                if ith_batch % 10 == 9:
                    print('\nTen-batch statistics: [%d, %5d] loss: %.3f' %
                        (epoch + 1, ith_batch + 1, running_loss / 10))
                    running_loss = 0.0

                # print every 10 mini-batches
                # if ith_batch % 10 == 9:
                #    print('[%d, %5d] loss: %.3f' %
                #          (epoch + 1, ith_batch + 1, running_loss / 10))
                #    running_loss = 0.0

        finished = '\nFinished Training\n'
        return finished

    def model_testing(self):
        # Set the eval mode flag on the model (not important here but good practice)
        self.model.eval()

        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader: # iterate through the data
                images, labels = data['image'],data['label']

                # we are doing binary classification here
                labels[labels != 0] = 1

                # calculate outputs by running images through the network 
                outputs = self.model(images.float())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                # increment total and correct
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # return the accuracy
        accuracy = ('Accuracy of the networks: %d %%' % (
            100 * correct / total))
        return accuracy

