% ----------------------
% This code shows how to perform unsupervised hierarchical dynamic parsing and encoding (HDPE) for action recognition on the ChaLearn dataset
% This code is adapted from the code "Chalearn_VideoDarwin.m" by Basura Fernando; website: https://bitbucket.org/bfernando/videodarwin
% Relevant paper:
% Modeling Video Evolution for Action Recognition
% Basura Fernando, Efstratios Gavves, Jose Oramas M., Amir Ghodrati, Tinne Tuytelaars; 
% The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 5378-5387

% Following "Chalearn_VideoDarwin.m", the skeleton data provide with
% chalearn http://gesture.chalearn.org/ are used
% In this experiment the vector quantized features (dictionary of 100) are employed.
% Train data is used for training and val data is used for testing. 

% The function "getVideoDarwin_HDPE" in this code is used to extract
% HDPE representations; Copyright: Bing Su
%
% Instructions
% ------------
% Note : Please check TODOs : change the paths etc..
% Dependency : vlfeat-0.9.18, liblinear-1.93, libsvm-3.18 
%
%
function [Pre,Recall,Fscore] = ChaLearn_HDPE()    
    % TODO
%     addpath('~/lib/vlfeat/toolbox');
%     vl_setup();		
%     % TODO
% 	% add lib linear to path
% 	addpath('~/lib/liblinear/matlab');
% 	% TODO
% 	% add lib svm to path
%     addpath('~/lib/libsvm/matlab');
    
    
    % TODO
    % add lib svm to path
    addpath('lib/liblinear-1.96/matlab');
    addpath('lib/libsvm-3.20/matlab');
    addpath('lib/vlfeat-0.9.18/toolbox');  %E:\bing\Action\TransDarwin\vlfeat-0.9.20-bin\vlfeat-0.9.20
    vl_setup();
    
    %data downloads and pre-process;
    % we employ the same frame-wide features in "Chalearn_VideoDarwin.m" 
    if exist('ChaLearn_train_test_split.mat','file') ~= 2            
        urlwrite('http://users.cecs.anu.edu.au/~basura/data/ChaLearn_train_test_split.mat','ChaLearn_train_test_split.mat');
    end
    load('ChaLearn_train_test_split.mat');    
    if exist('scaledExamples','dir') == 7 && numel(dir(sprintf('scaledExamples/*.mat'))) == 13883
        fprintf('Dataset exists..\n');
    else
        fprintf('Start downloading dataset..\n');
        urlwrite('http://users.cecs.anu.edu.au/~basura/data/scaledExamples.tar.gz','scaledExamples.tar.gz');
        system('tar -zxvf scaledExamples.tar.gz');
    end
    
    videoname = fnames;	 
    feats = {'.\scaledExamples\scaledExamples'}; 
    
    % TODO: the parameters
    template_length = 7;
    band_factor = 2;
    
    for f = 1 : numel(feats)
        file = sprintf('%s.mat',feats{f});
        
        % Extract the HDPE representation for the video
        % ALL_Data: the HDPE representation
        % Ori_Data: the rank pooling representation for the whole sequence
        % Local_Data: the mean pooling representation for the whole
        % sequence
        % We use ALL_Data only; the last two can be used for fusion
        [ALL_Data,Ori_Data,Local_Data] = getVideoDarwin_HDPE(feats{f},videoname,template_length,band_factor);
        
        ALL_Data_cell{f} = [ALL_Data];
        
    end    
    Options.KERN = 0;    % non linear kernel
    Options.Norm = 2;     % L2 normalization

	if Options.KERN == 6        
        for ch = 1 : size(ALL_Data_cell,2)                
            x = vl_homkermap(ALL_Data_cell{ch}', 2, 'kchi2') ;  
            ALL_Data_cell{ch} = x';
        end
    end
    
    if Options.KERN == 5        
        for ch = 1 : size(ALL_Data_cell,2)    
            x = ALL_Data_cell{ch};            
            ALL_Data_cell{ch} = sqrt(abs(x));
        end
    end 
    
	
	if Options.Norm == 2       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL2(ALL_Data_cell{ch});
        end
    end  
    
    if Options.Norm == 1       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL1(ALL_Data_cell{ch});
        end
    end  
    
    if size(ALL_Data_cell,2) == 1
        weights = 1;
    end

    if size(ALL_Data_cell,2) == 2 || size(ALL_Data_cell,2) == 6 
        weights = [0.5 0.5];
    end

    if size(ALL_Data_cell,2) > 2 && size(ALL_Data_cell,2) ~= 6
        nch = size(ALL_Data_cell,2) ;
        weights = ones(1,nch) * 1/nch;
    end      
    
    classid = labels2;  
    trn_indx = [cur_train_indx]; % [cur_train_indx  cur_val_indx]; 
    test_indx = [cur_val_indx];  % cur_test_indx     
    TrainClass_ALL = classid(trn_indx,:);
    TestClass_ALL = classid(test_indx,:);   
   [~,TrainClass] = max(TrainClass_ALL,[],2);
   [~,TestClass] = max(TestClass_ALL,[],2);   		
      
    for ch = 1 : size(ALL_Data_cell,2)        
        ALL_Data = ALL_Data_cell{ch};
        TrainData = ALL_Data(trn_indx,:);        
        TestData = ALL_Data(test_indx,:);

        TrainData_Kern_cell{ch} = [TrainData * TrainData'];    
        TestData_Kern_cell{ch} = [TestData * TrainData'];                        
        clear TrainData; clear TestData; clear ALL_Data;            
    end
    
    for wi = 1 : size(weights,1)
        TrainData_Kern = zeros(size(TrainData_Kern_cell{1}));
        TestData_Kern = zeros(size(TestData_Kern_cell{1}));
            for ch = 1 : size(ALL_Data_cell,2)     
                TrainData_Kern = TrainData_Kern + weights(wi,ch) * TrainData_Kern_cell{ch};
                TestData_Kern = TestData_Kern + weights(wi,ch) * TestData_Kern_cell{ch};
            end
            [precision(wi,:),recall(wi,:),acc(wi) ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass);       
    end          
            
    [~,indx] = max(acc);            
    precision = precision(indx,:);
    recall = recall(indx,:); 
    F = 2*(precision .* recall)./(precision+recall);
    fprintf('Mean F score = %1.4f\n',mean(F));
        
    Pre = mean(precision)
    Recall = mean(recall)
    Fscore = mean(F)
end






function [ALL_Data,Ori_Data,Local_Data] = getVideoDarwin_HDPE(featType,Videos,template_length,band_factor) 
% This is the major function for HDPE
    CVAL = 1; % C value for the ranking function or SVR    
	TOTAL = size(Videos,2);  
    max_iteration_num_ini = 15;
    %template_length = 20;
    dim = 100;
    %band_factor = 1.5;
    for i = 1:TOTAL
        name = Videos{i};         
        MATFILE = fullfile(featType,sprintf('%s.mat',name));        
        load(MATFILE);
        data  = clustDist';  clear clustDist;
        temp_align_path = selfclustering(data,max_iteration_num_ini,template_length,band_factor);
        first_layer_data = zeros(template_length,dim);
        first_layer_data_rev = zeros(template_length,dim);
        for temp_align_count = 1:template_length
            temp_start = temp_align_path(temp_align_count,1);
            temp_end = temp_align_path(temp_align_count,2);
            % Use mean pooling in the first layer
            first_layer_data(temp_align_count,:) = mean(data([temp_start:temp_end],:),1); 
            %first_layer_data_rev(temp_align_count,:) = W_rev';           
        end
        
        % Use rank pooling in the second layer
        Wh = genRepresentation(first_layer_data,CVAL);
        W = genRepresentation(data,CVAL);
        Wo = mean(data,1);
        Wo = vl_homkermap(Wo',2,'kchi2');
        
        if i == 1
             ALL_Data =  zeros(TOTAL,size(Wh,1)) ;          
             Ori_Data =  zeros(TOTAL,size(W,1)) ;
             Local_Data =  zeros(TOTAL,size(Wo,1)) ;
        end
        if mod(i,100) == 0
            fprintf('.')
        end
        ALL_Data(i,:) = Wh';
        Ori_Data(i,:) = W';
        Local_Data(i,:) = Wo';
    end
    fprintf('Complete...\n')
end

function W = genRepresentation(data,CVAL)
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end                            
    Data = vl_homkermap(Data',2,'kchi2');
    %Data = (sqrt(abs(Data')));
    Data = Data';

    W_fow = liblinearsvr(Data,CVAL,2); 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end            
    Data = vl_homkermap(Data',2,'kchi2');
    %Data = (sqrt(abs(Data')));
    Data = Data';            
    W_rev = liblinearsvr(Data,CVAL,2); 			              
    W = [W_fow ; W_rev];  


end


function X = normalizeL2(X)
	for i = 1 : size(X,1)
		if norm(X(i,:)) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:));
		end
    end	   
end

function [trn,tst] = generateTrainTest(classid)
    trn = zeros(numel(classid),1);
    tst = zeros(numel(classid),1);
    maxC = max(classid);
    for c = 1 : maxC
        indx = find(classid == c);
        n = numel(indx);
        tindx = indx(1:4);
        testindx = indx(5:end);
        trn(tindx,1) = 1;
        tst(testindx,1) = 1;
    end
end

function [X] = getLabel(classid)
    X = zeros(numel(classid),max(classid))-1;
    for i = 1 : max(classid)
        indx = find(classid == i);
        X(indx,i) = 1;
    end
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end
    
    if normD == 1
        Data = normalizeL1(Data);
    end
    
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -q',C) );
    w = model.w';
end

function [precision,recall,acc ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
         nTrain = 1 : size(TrainData_Kern,1);
         TrainData_Kern = [nTrain' TrainData_Kern];         
         nTest = 1 : size(TestData_Kern,1);
         TestData_Kern = [nTest' TestData_Kern];         
         %C = [1 10 100 500 1000 ];
         C = 1000;
         for ci = 1 : numel(C)
             model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q ',C(ci)));               
         end        
         
         [~,max_indx]=max(model);
         
         C = C(max_indx);
         
         for ci = 1 : numel(C)
             model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C(ci)));
             [predicted, acc, scores{ci}] = svmpredict(TestClass, TestData_Kern ,model);	                 
             [precision(ci,:) , recall(ci,:)] = perclass_precision_recall(TestClass,predicted);
             accuracy(ci) = acc(1,1);
         end        
         
        [acc,cindx] = max(accuracy);   
        scores = scores{cindx};
        precision = precision(cindx,:);
        recall = recall(cindx,:);
end

function [precision , recall] = perclass_precision_recall(label,predicted)

    
    
    
    for cl = 1 : 20
        true_pos = sum((predicted == cl) .* (label == cl));
        false_pos = sum((predicted == cl) .* (label ~= cl));
        false_neg = sum((predicted ~= cl) .* (label == cl));
        precision(cl) = true_pos / (true_pos + false_pos);
        recall(cl) = true_pos / (true_pos + false_neg);
        
    end


end
