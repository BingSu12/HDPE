function [distance,path] = computeWarpingPathtoTemplate_Eud_band_addc(sequence_sample, template, band_factor)
% Compute the modified DTW alignment path from a sequence to a template

% -------------
% INPUT:
% -------------
% sequence_sample: a num_frames * d matrix, representing the input sequence consists of of N
% d-dimensional vectors, where num_frames is the number of instances (vectors) in this sequence,
% and d is the dimensionality of instances;
%
% tempalte: a template_length * d matrix, representing the input template sequence
% consists of of template_length d-dimensional vectors, , where template_length is the number of instances (vectors) in
% the template, and d is the dimensionality of instances;
%
% band_factor: the band factor controlling the degree of warping

% -------------
% OUTPUT
% -------------
% distance: the modified DTW distance between the sequence and the template
% path: the alignment path between the sequence and the template

% -------------
% Copyright (c) 2017 Bing Su
% -------------


num_frames = size(sequence_sample, 1);
template_length = size(template, 1);
max_local_length = num_frames-template_length+1;

band_size = band_factor*(num_frames/template_length);
max_local_length = max(ceil(band_size),1);

scores = ones(num_frames, template_length, max_local_length) * -10^20;
factor = template_length/num_frames;
offset = zeros(num_frames, template_length);

scores(1,1,1) = -norm((sequence_sample(1,:) - template(1,:)),2)^2; 

for j = 1:template_length
    for i = j:num_frames
            c_ij = -norm((sequence_sample(i,:) - template(j,:)),2)^2;
            for l = 1
                if j>1
                    
                    scores(i,j,l) = c_ij + max(scores(i-1,j-1,:)); 
                    indextemp = find(scores(i-1,j-1,:)==max(scores(i-1,j-1,:)));
                    offset(i,j) = indextemp(1);
                end
            end
            for l = 2:min(i-j+1,max_local_length)                
                scores(i,j,l) = c_ij + scores(i-1,j,l-1);
            end
    end
end
distance = max(scores(num_frames,template_length,:));
match_frame = template_length;
f = num_frames;
path = zeros(template_length,2);
maxvalue = max(scores(f,match_frame,:));
ltemp = find(scores(f,match_frame,:)==maxvalue);
ltemp = ltemp(1);

while (f>0)        
    path(match_frame,:) = [f-ltemp+1 f];
    f = f - ltemp;
    match_frame = match_frame - 1;
    ltemp = offset(f+1,match_frame+1);

end

if f~=0
    f
    path
    disp('Not match!');
end

end