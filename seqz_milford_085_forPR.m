% Hacking a simple filter for deep learning results from Zetao

close all;
clc;
clear;
load('10.mat'); % load the confusion matrix, where each column is the eucidean distance between each testing image and all the reference/training images

full_file = dir(fullfile('lack/','*.mat')); % the folder 'lack' contains files with each file contains a confusion matrix constructed from each layer of the CNN
num_file = numel(full_file);

% Matching parameters
flen = 5;   % Filter length
dt = 3;     % Vibration threshold
st = 0.75;   % Slope matching tolerance plus minus of the desired slope of 1 or -1.

% st is the parameter we sweep to generate the precision recall curve
for file_idx = 1:1:num_file  % here I iterate over different layers, 

temp_file = full_file(file_idx).name;
temp_file2 = fullfile('lack',temp_file);
load(temp_file2);
precision_all = [];
recall_all = [];


for flen = 5:1:5
    for dt = 3:1:3
for st = 0.15:0.2:2.35
    
d = diff_matrix(1:4789,4790:9575); % diff_matrix is the confusion matrix loaded from '10.mat'(see the command above). the whole eynsham datasets (including training/reference and testing data) contains 9575 frames, out of which the first 4789 frames are for training/reference, and 4790~9575 are 
                                   % for testing, the diff_matrix is a
                                   % 9575* 9575 matrix (compare each image
                                   % to any other image in the data).
                                   % therefore, we cut
                                   % diff_matrix(1:4789,4790:9575),
                                   % considering only the confusion matrix
                                   % constructed from matching testing
                                   % images to traininge images.

[a b] = min(d); % the min score is the best match

inlier_fraction = 5/6;  % Percentage of top matches used in the vibration calculation, allows the occasional outlier
p = zeros(1, length(b));
cy = zeros(1,length(b)-flen+round(flen/2));
% Go through all frames
for i = 1:length(b) - flen
% for i = 1:100
% for i = 3326:3326
    
%     s = b(i);
    maxdiff = 0;
    
%     Index of match
    ci = i + round(flen / 2);   
    
%     Analyze vibrations
    vibrations = abs(diff(b(i:i+flen - 1)));
    [sort_vib_val sort_vib_indices] = sort(vibrations);    
    maxdiff = max(sort_vib_val(1:round(inlier_fraction * flen)));
    
%     maxdiff = max(abs(diff(b(i:i+flen - 1))));
    
%     for j = 0:flen - 2
%         maxdiff = max(abs(b(i + j + 1) - b(i + j)));
%     end
    
%     Linear regression
    pt = polyfit(0:flen - 1, b(i:i + flen - 1), 1);   
    p(ci) = pt(1);
    
    cx(ci) = ci;
%     cy(ci) = b(i) + p(ci) * 0.5 * flen;    

    
    if maxdiff <= dt && (abs(p(ci) - 1) < st || abs(p(ci) - -1) < st) %Forward and reverse matching
% if (maxdiff <= dt) 
%     if maxdiff <= dt & (abs(p(ci) - 1) < st) %Forward matching only
%         q(ci) = b(i);  
        cy(ci) = pt(2) + pt(1) * 0.5 * flen;
%         plot([cx(ci) - flen / 2 cx(ci) + flen / 2], [cy(ci) - p(ci) * flen / 2 cy(ci) + p(ci) * flen / 2], 'b-');
        
        
        % check previous points and if they are not confident, add them
        for inner_idx = 1:round(flen/2)
            check_idx = ci-inner_idx;
            if(cy(check_idx) == 0) % not confident
                cy(check_idx) = b(check_idx);
            end
        end
        
        for inner_idx = 1:(flen - round(flen/2)-1)
            check_idx = ci+inner_idx;
            if(check_idx<=4783)
               cy(check_idx) = b(check_idx);
            end
        end
        
    end
%     else        
%        if(cy(ci)==0)
%           cy(ci) = 0;
%        end
%     end
end


% check the ground truth.
load('GroundTruth_Eynsham.mat');
start_first = 1;
end_first = 4789;
len_first = end_first-start_first+1;
start_second = 4790;
end_second = 9575;
len_second = end_second-start_second+1;
half_matrix = 4785; 

ground_matrix = zeros(len_second,len_first); % each row represents the matching result of one image in the second traverse

for ground_idx = start_second:end_second
    value_ground = ground_truth(ground_idx,:);
    value_fit = find(value_ground == 1);
    
    value_fit2 = value_fit(find(value_fit<end_first));% only store those in first round
    value_fit3 = value_fit2 - start_first + 1; % '16' here is the consistent shift between the ground truth
    value_fit4 = value_fit3(find(value_fit3>0));
    
    matris_idx = ground_idx - start_second +1;
    ground_matrix(matris_idx,value_fit4) = 1;
end

tp_num = 0;
tp_value = [];
fp_num = 0;
fp_value = [];
for truth_idx = 1:length(cy)
    
    ground_row = ground_truth(truth_idx+end_first,:);
    ground_row_idx = find(ground_row == 1);
    
    if(cy(truth_idx)~=0) % means we consider it to be a confident match       
         truth_va = cy(truth_idx);
         truth_va2 = round(truth_va);
         if(any(ground_row_idx == round(truth_va2)))
            tp_num = tp_num + 1;
            tp_value = [tp_value;truth_idx];
         else
             fp_num = fp_num + 1;
            fp_value = [fp_value;truth_idx];
             truth_x = ones(1,length(ground_row_idx))*truth_idx;
%  plot(truth_x,ground_row_idx,'ws','LineWidth',1);
         end       
    else
        
   
    end
end

precision = tp_num/(tp_num+fp_num);
recall = tp_num/length(b); % 0.8571 is currently the highest rate.
precision_all = [precision_all;precision];
recall_all = [recall_all;recall];
end
    end
end


save_file = ['pr_',temp_file];
save(save_file,'precision_all','recall_all');
end