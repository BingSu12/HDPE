function partemp_align_path = selfclustering(features,max_iteration_num_ini,template_length,band_factor)
% Perform unsupervised parsing for a sequence by temporal clustering

% -------------
% INPUT:
% -------------
% features: a feanum * d matrix, representing the input sequence consists of of feanum
% d-dimensional vectors, where feanum is the number of instances (vectors) in features,
% and d is the dimensionality of instances;
% 
% max_iteration_num_ini: the maximum number of iterations
% 
% template_length: the length of the essential sequence, can be viewed as
% the number of divisions to be parsed;
%
% band_factor: the band factor controlling the degree of warpingy)

% -------------
% OUTPUT
% -------------
% partemp_align_path: the alignment path between the sequence "features" and the
% essentail sequence


% -------------
% Copyright (c) 2017 Bing Su
% -------------


feanum = size(features,1);
partemp_align_path = zeros(template_length,2);
if feanum>=template_length
    partemp_ave_align_num = [floor(feanum/template_length) floor(feanum/template_length)];
    temp_start = 1;
    for temp_align_count = 1:template_length-1
        temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
        partemp_align_path(temp_align_count,:) = [temp_start temp_end];
        temp_start = temp_end + 1;
    end
    temp_end = feanum;
    partemp_align_path(template_length,:) = [temp_start temp_end];
else
    partemp_ave_align_num = [max(floor(template_length/feanum),1) max(floor(template_length/feanum),1)];
    temp_start = 1;
    for temp_align_count = 1:feanum-1
        temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
        for temp_tem_count = temp_start:temp_end
            partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
        end
        temp_start = temp_end + 1;
    end
    temp_end = template_length;
    temp_align_count = feanum;
    for temp_tem_count = temp_start:temp_end
        partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
    end
end


dim = size(features,2);

for ite = 1:max_iteration_num_ini 
    mean_sequence = zeros(template_length, dim);
    temp_align_path = partemp_align_path;
    for temp_align_count = 1:template_length
        temp_start = temp_align_path(temp_align_count,1);
        temp_end = temp_align_path(temp_align_count,2);
        mean_sequence_num = temp_end - temp_start + 1;
        if mean_sequence_num > 0
            mean_sequence(temp_align_count,:) = sum(features([temp_start:temp_end],:),1)/mean_sequence_num;
        end
    end
    feanum = size(features,1);
    if feanum>=template_length
        [dis,temp_align_path] = computeWarpingPathtoTemplate_Eud_band_addc(features, mean_sequence, band_factor);
        partemp_align_path = temp_align_path;
    else
        [dis,temp_align_path] = computeWarpingPathtoTemplate_Eud_band_addc(mean_sequence, features, band_factor);
        for temp_align_count = 1:feanum
            temp_start = temp_align_path(temp_align_count,1);
            temp_end = temp_align_path(temp_align_count,2);
            for temp_tem_count = temp_start:temp_end
                partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
            end
        end
    end
end                                

end