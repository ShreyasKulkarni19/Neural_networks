clc;
clear all;

%use imresize to change dimension of all images to be same in training set

a = imread('C:\Users\Nandu\Documents\MATLAB\CNN Proj\bloodcelltest.jpeg');
%figure,imshow(a)

matlabroot = 'C:\Users\Nandu\Documents\MATLAB';
DataSetPath = fullfile(matlabroot,'CNN Proj','Dataset');
%DataSetPathTest = fullfile(matlabroot,'CNN Proj','Validation');

Data = imageDatastore(DataSetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
%DataTest = imageDatastore(DataSetPathTest,'IncludeSubfolders',true,'LabelSource','foldernames');

%Need to understand what Layers and Options data mean
%Flatten layer after maxpooling
%flattenLayer in deeplearning toolbox
%avoiding overfitting of data

%numTrainFiles = 40;
[dataTrain,dataValidation] = splitEachLabel(Data,0.7,'randomize');

%dataTrain = Data;
%dataValidation = DataTest;

%Layers = [imageInputLayer([240 320 3])
%    convolution2dLayer(5,20)
%    reluLayer
%    maxPooling2dLayer(2,'stride',2)
%    convolution2dLayer(5,20)
%    reluLayer
%    maxPooling2dLayer(2,'stride',2)
%    fullyConnectedLayer(4)
%    softmaxLayer
%    classificationLayer()];
inputSize = [240 320 3];
numClasses = 2;

%Layers = [
 %   imageInputLayer(inputSize)
    
  %  convolution2dLayer(5,20)
   % reluLayer
    
   % maxPooling2dLayer(2,'stride',2)
    
   % fullyConnectedLayer(numClasses)
   % softmaxLayer
   % classificationLayer];
Layers = [
    imageInputLayer(inputSize);
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'stride',2);
    fullyConnectedLayer(numClasses,'BiasLearnRateFactor',2);
    softmaxLayer()
    classificationLayer()];

Options = trainingOptions('sgdm','MaxEpochs',10,'LearnRateSchedule','piecewise','initialLearnRate',0.001','LearnRateDropFactor',0.1,'LearnRateDropPeriod',8,'MiniBatchSize',100,'ValidationData',dataValidation,'ValidationFrequency',30,'Verbose',false,'Plots','training-progress');
Convnet = trainNetwork(dataTrain,Layers,Options);
%Output = classify(Convnet,a);
YPred = classify(Convnet,dataValidation);
YValidation = dataValidation.Labels;
accuracy = mean(YPred == YValidation)
