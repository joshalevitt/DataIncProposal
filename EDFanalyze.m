
%set sampling frequency, initialize some variables
fs = 250;
fullText = '';

labels = 0;
features = zeros(1, 129);

CurrentDir = pwd;

% traverse the folder containing the data
cd(strcat(CurrentDir, '\train'))
list = ls();
folderNum = 3;
while folderNum <= length(list)
    fold = list(folderNum, :)
    cd(fold)
    subList = ls();
    edfName = {};
    recName = {};
    j = 1;
    %find the edf, txt and rec file from each folder
    while j <= length(subList)
        if ~isempty(strfind(subList(j,:),'.edf'))
            edfName = [edfName, {subList(j, :)}];
        elseif ~isempty(strfind(subList(j,:),'.rec'))
            recName = [recName, {subList(j, :)}];
        elseif ~isempty(strfind(subList(j,:),'.txt'))
            txtName = subList(j, :);
        end           
        j = j + 1;
    end
    %open our files
    report = fileread(txtName);
    fullText = strcat(fullText, report);
    
    if length(edfName) ~= length(recName)
        display('unequal number of rec and edf filed in this folder')
    end
    fileNum = 1;
    while fileNum <= length(edfName)
    
        [~, data] = edfread(edfName{fileNum});
        f = fopen(recName{fileNum});
        text = textscan(f, '%s');
        fclose(f);

        txtData = text{1};
        numEvents = length(txtData);
        eventData = zeros(numEvents, 4);
        i = 1;
        %parse event data
        while i <= numEvents
            eventSpecs = strsplit(txtData{i}, ',');
            eventData(i,1) = str2double(eventSpecs{1});
            eventData(i,2) = str2double(eventSpecs{2});
            eventData(i,3) = str2double(eventSpecs{3});
            eventData(i,4) = str2double(eventSpecs{4});
            i = i+1;
        end

        channel = eventData(1,1) + 1;
        event = 1;

        eventLabels = zeros(numEvents, 1);
        eventFeatures = zeros( numEvents, 129);

        %match each event with the corresponding EEG, and calculate power
        %spectrum

        while event <= numEvents


            %remove leading and trailing zeros
            signal=transpose(data(channel,:));
            signal=signal(1:find(signal,1,'last'));
            signal=signal(find(signal,1,'first'):length(signal));

            %normalize EEG amplitude, by subtracting mean and dividing by standard
            %deviation after ignoring outliers
            sor=sort(signal(1:length(signal)));
            a=round(.2*length(sor));
            sor=sor(a:(length(sor)-a));
            ma=mean(sor);
            sta=std(sor);
            signal=(signal-ma)/sta;

            % loop through all event for this channel
            while  event <= numEvents && eventData(event,1) + 1 == channel
                startTime = eventData(event, 2);
                i = round(startTime * fs + 1);
                sigEvent = signal(i : i + fs - 1);

                %calculate PSD
                psdEvent = periodogram(sigEvent, [], [], fs);

                eventLabels(event) = eventData(event, 4);
                eventFeatures(event, :) = psdEvent;

                event = event + 1;
            end

            if event <= numEvents
                channel = eventData(event,1) + 1;
            end
        end
        labels = cat(1, labels, eventLabels);
        features = cat(1, features, eventFeatures);
        fileNum = fileNum + 1;
    end
    cd(strcat(CurrentDir, '\train'))
    folderNum = folderNum + 1;
end


%Repeat this entire process for the eval folder
testlabels = 0;
testfeatures = zeros(1, 129);

cd(strcat(CurrentDir, '\eval'))
list = ls();
folderNum = 3;
while folderNum <= length(list)
    fold = list(folderNum, :)
    cd(fold)
    subList = ls();
    edfName = {};
    recName = {};
    j = 1;
    while j <= length(subList)
        
        if ~isempty(strfind(subList(j,:),'.edf'))
            edfName = [edfName, {subList(j, :)}];
        elseif ~isempty(strfind(subList(j,:),'.rec'))
            recName = [recName, {subList(j, :)}];
        elseif ~isempty(strfind(subList(j,:),'.txt'))
            txtName = subList(j, :);
        end           
        j = j + 1;
    end
    report = fileread(txtName);
    fullText = strcat(fullText, report);
    if length(edfName) ~= length(recName)
        display('unequal number of rec and edf filed in this folder')
    end
    fileNum = 1;
    while fileNum <= length(edfName)
    
        [~, data] = edfread(edfName{fileNum});
        f = fopen(recName{fileNum});
        text = textscan(f, '%s');
        fclose(f);

        txtData = text{1};
        numEvents = length(txtData);
        eventData = zeros(numEvents, 4);
        i = 1;
        while i <= numEvents
            eventSpecs = strsplit(txtData{i}, ',');
            eventData(i,1) = str2double(eventSpecs{1});
            eventData(i,2) = str2double(eventSpecs{2});
            eventData(i,3) = str2double(eventSpecs{3});
            eventData(i,4) = str2double(eventSpecs{4});
            i = i+1;
        end

        channel = eventData(1,1) + 1;
        event = 1;

        eventLabels = zeros(numEvents, 1);
        eventFeatures = zeros( numEvents, 129);

        while event <= numEvents


            %remove leading and trailing zeros
            signal=transpose(data(channel,:));
            signal=signal(1:find(signal,1,'last'));
            signal=signal(find(signal,1,'first'):length(signal));

            %normalize EEG amplitude, by subtracting mean and dividing by standard
            %deviation after ignoring outliers
            sor=sort(signal(1:length(signal)));
            a=round(.2*length(sor));
            sor=sor(a:(length(sor)-a));
            ma=mean(sor);
            sta=std(sor);
            signal=(signal-ma)/sta;

            while  event <= numEvents && eventData(event,1) + 1 == channel
                startTime = eventData(event, 2);
                i = round(startTime * fs + 1);
                sigEvent = signal(i : i + fs - 1);

                psdEvent = periodogram(sigEvent, [], [], fs);

                eventLabels(event) = eventData(event, 4);
                eventFeatures(event, :) = psdEvent;

                event = event + 1;
            end

            if event <= numEvents
                channel = eventData(event,1) + 1;
            end
        end
        testlabels = cat(1, testlabels, eventLabels);
        testfeatures = cat(1, testfeatures, eventFeatures);
        fileNum = fileNum + 1;
    end
    cd(strcat(CurrentDir, '\eval'))
    folderNum = folderNum + 1;
end

% remove the dummy entries from our labels and feature sets
labels(1) = [];
testlabels(1) = [];
features(1,:) = [];
testfeatures(1,:) = [];

% calculate the frequency band from the power spectrum, in order to reduce
% our dimensionality
normfeatures = [mean(features(:, 2:4), 2), mean(features(:, 4:9), 2), mean(features(:, 9:14), 2), mean(features(:, 15:32), 2), mean(features(:, 33: 60), 2)];
normTestfeatures = [mean(testfeatures(:, 2:4), 2), mean(testfeatures(:, 4:9), 2), mean(testfeatures(:, 9:14), 2), mean(testfeatures(:, 15:32), 2), mean(testfeatures(:, 33: 60), 2)]; 

%concatenate all our data together
fullData = cat(1, cat(2, labels, normfeatures), cat(2, testlabels, normTestfeatures));

% do 5 - fold cross validaiton, and build a confusion matrix
Confusion = zeros(6);
c = cvpartition(length(fullData), 'KFold', 5);
i = 1;
t = templateKNN('NumNeighbors', 10, 'Standardize', 1);
% loop through 5 folds
while i <= 5
    idx = training(c, i);
    % train knn on the train data
    Classifier = fitcecoc(fullData(idx, 2:6), fullData(idx, 1), 'Verbose', 2, 'Coding', 'onevsall', 'Learners', t);
    idx = test(c, i);
    testSet = fullData(idx, 2:6);
    testSetLabels = fullData(idx, 1);
    %test oon test data
    testpredictions = predict(Classifier, testSet);
    % add to confusion matrix
    Confusion = Confusion + crosstab(testpredictions, testSetLabels);

    i = i + 1;
end

% convert our confusion matrix to percentages
numCorrect = 0;
i = 1;
while i <= 6
    numCorrect = Confusion(i, i);
    Confusion(i, :) = Confusion(i, :) / sum(Confusion(i, :));
    i = i + 1;
end

%make our confusion matrix figure
figure
imagesc(Confusion)
colorbar
colormap(gray)
caxis([0 1])
xlabel('Predicted Class')
ylabel('True Class')
xticklabels({'spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg'});
yticklabels({'spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg'});

acc = numCorrect / length(fullData);
display('The classification accuracy is: ')
display(acc)

% make our wordcloud figure
figure
wordcloud(fullText);