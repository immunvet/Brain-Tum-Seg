%%
clear;clc;close all;





%%
%directories with png images
%place folders with images to the working directory or specify the
%appropriate path
dataSetDir = fullfile('./');
imageDir = fullfile(dataSetDir,'images_train');
labelDir = fullfile(dataSetDir,'masks_train');

outFolder = './out';


%%
%creating datastore for images
imds = imageDatastore(imageDir);

%%
%assigning classes names to numerical values
classNames = ["not_a_tumor" "nonenhcore" "edema" "enhancing"];
           
pxlabelIDs = [0 1 2 4];
       
pxds = pixelLabelDatastore(labelDir,classNames,pxlabelIDs);

%%
%demonstrations of applying masks on images

%creating our own colormap in order to use it for visualizations 
cmap = [1 1 1;0.8 0.4 0.8; 0.4 0.8 1; 1 0.2 0];


I = readimage(imds,6789);
C = readimage(pxds,6789);


B = labeloverlay(I,C,'Colormap',cmap);


figure
imshow(B);
%colorbar adding
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap);


%%
%spliting data for training and validation datasets
[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionData(imds,pxds,pxlabelIDs);
numTrainingImages = numel(imdsTrain.Files);
numTestingImages = numel(imdsTest.Files);


%%
augmenter = imageDataAugmenter('RandScale', [1.2 1.2], 'RandRotation', [-10 10]);

pximds = pixelLabelImageDatastore(imds,pxds, 'DataAugmentation', augmenter); 

%counting pixels with different values(labels)

tbl = countEachLabel(pxds);


%plotting labels` values 
frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classNames),frequency)
xticks(1:numel(classNames)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency');


%%
%normalizing weights and creating new final classification layer

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);


%%
%assigning input size and creating network with name 'lgraph'
imageSize = [240 240 3];
numClasses = numel(classNames);
lgraph = segnetLayers(imageSize, numClasses, 'vgg16');


lgraph = removeLayers(lgraph,'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');
plot(lgraph);





%%
options = trainingOptions('adam', ...
    'GradientDecayFactor', 0.9, ... 
    'SquaredGradientDecayFactor', 0.99, ...
    'Epsilon', 0.001, ...
    'InitialLearnRate',0.002, ...
    'L2Regularization',0.0005, ...  
    'MaxEpochs',10, ...  
    'MiniBatchSize',22, ...
    'CheckpointPath', tempdir, ...
    'Shuffle', 'every-epoch', ... #shuffling every epoch to prevent overfitting
    'VerboseFrequency',30, ...
    'Plots','training-progress');      


%%
%training proceedure

[net, info] = trainNetwork(pximds,lgraph,options);


%%
%saving network and environment

vgg16_final = net;
save vgg16_final;

%%

pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',10, ...
    'WriteLocation',tempdir, ...  
    'Verbose',true );
 
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);

%%
metrics.DataSetMetrics
metrics.ClassMetrics




%%

%Test network on one image

I = readimage(imdsTest, 1245);
C = semanticseg(I, net);

B = labeloverlay(I,C);
imshow(B)
%pixelLabelColorbar('jet', classes);

%%
%creating our own colormap in order to use it for visualizations 
cmap = [1 1 1; 1 0 0; 1 .5 0; 1 1 0];

%%
%demonstrations of applying masks on images
Im1 = readimage(imdsTest,10);
C1 = readimage(pxdsTest,10);



B1 = labeloverlay(Im1,C1,'Colormap',cmap);


figure
%imshow(B);
%Test network on one image

Im2 = readimage(imdsTest, 10);
C2 = semanticseg(Im2, net);

B2 = labeloverlay(Im2,C2,'Colormap',cmap);
%imshow(B);
%pixelLabelColorbar('jet',classNames);
subplot(131), imshow(Im1), title('Scan Without Labels');
subplot(132), imshow(B1), title('Ground Truth Labels');
subplot(133), imshow(B2), title('Our model` created Labels');

% Add a colorbar 
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap);

%%
%Validation Visualization

%confusion matrix
normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';

%MeanIoU Visualization
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')
