%% Training Network
imds = imageDatastore('data_augmented(3 classes)', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
net = googlenet;
analyzeNetwork(net)
net.Layers(1)
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
miniBatchSize = 5;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',12, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%% Validate 4 Images
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% Confusion Matrix
plotconfusion(imdsValidation.Labels, YPred)
title(['overall per image accuracy ', num2str(round(100*accuracy)), '%'])

%% Face detector
data2 = load('fasterRCNNVehicleTrainingData.mat');
load('-mat', '3111');
rng(0);
shuffledIdx = randperm(height(trainingData3111));
trainingData4 = trainingData3111(shuffledIdx,:);
imds = imageDatastore(trainingData4.imageFilename);
blds = boxLabelDatastore(trainingData4(:,2:end));
ds = combine(imds, blds);
lgraph = layerGraph(data2.detector.Network);
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 12, ...
    'VerboseFrequency', 200, ...
    'CheckPointPath', tempdir);
detector5 = trainFasterRCNNObjectDetector(trainingData4, ...
    lgraph, options, 'NegativeOverlapRange', [0 0.3], ...
    'PositiveOverlapRange', [0.6 1]);

%% Load NET and DETECTOR
load('-mat','net_5_12');
load('-mat', 'detector5');

%% Webcam + Bar graph (Top 3 classes)
camera = webcam("Logitech BRIO");
camera.Resolution = '320x240';

inputSize2 = net.Layers(1).InputSize(1:2)
 
h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,3,1);
ax2 = subplot(1,3,2);
ax3 = subplot(1,3,3);
ax2.ActivePositionProperty = 'position';
t = [0 0 0];
alert = 0;

while ishandle(h)
    im = snapshot(camera);
    [bbox, score, label] = detect(detector5, im);
    image(ax1, im);
    if (bbox ~= [0 0 0 0])
        [m, n] = size(bbox);
        if (m >1)
            score = score';
            [m, n] = size(score);
            [M, I] = max(score);
            temp = I(1);
            bbox = bbox(temp, :);
        end
        ficOut = insertShape(im, 'rectangle', bbox);
        image(ax1, ficOut);
        im = imcrop(im, bbox);
        image(ax3, im);
    end
    im = imresize(im, inputSize2);
    [label, score] = classify(net, im);
    title(ax1, {char(label), num2str(max(score), 2)});
    [~, idx] = sort(score, 'descend');
    idx = idx(3:-1:1);
    scoreTop = score(idx);
    if (label == "pain" && scoreTop(3) > 0.8)
        alert = 1;
    elseif (label == "cry" && scoreTop(3) > 0.8)
        alert = 1;
    elseif (label == "calm" && scoreTop(3) > 0.75)
        alert = 0;
    end
            
    classes = net.Layers(end).Classes;
    classNamesTop = string(classes(idx));
    if alert == 1
        barh(ax2, scoreTop, 'r');
        title(ax2, 'Top 3')
        xlabel(ax2, 'Probability') 
        xlim(ax2, [0 1])
        yticklabels(ax2, classNamesTop)
        ax2.YAxisLocation = 'right';
    elseif alert == 0
        barh(ax2, scoreTop, 'b');
        title(ax2, 'Top 3')
        xlabel(ax2, 'Probability')
        xlim(ax2, [0 1])
        yticklabels(ax2, classNamesTop)
        ax2.YAxisLocation = 'right';
    end
    drawnow;
end