clear;

file = dir('mv/*.bin');                        %%bin文件目录
[num_sequence ~] = size(file);
resolution = '854x480_HW_Q3_CRF10';            %%分辨率选择
num_s = 0;
num_w = 0;
for j = 1:num_sequence
    a = strfind(file(j).name,resolution);
    if isempty(a) == 0
        num_s = num_s+1;

    %%
        %%ffmpeg
        copyfile(sprintf('mv/%s',file(j).name),'test/1.bin');            %%复制文件在/test/..
        system('ffmpeg -i test/1.bin test/out.yuv');
        system('ffmpeg -s 854x480 -i test/out.yuv test/out_%d.png');
        load video_thumbnail_model.mat; 
        num_frame = dir('test/*.png');
        [n_frame ~] = size(num_frame);
        kd = 8;
        for k=1:n_frame
           im = imread(sprintf('test/%s_%d.png', 'out', k));
           im_data = single(rgb2gray(im));
           [h,w] = size(im_data);
           im_data_thumbnails = imresize(im_data,[12,16]);
           x_a(k,:) = im_data_thumbnails(:)'*A(:,1:kd); 
           
        end
        f_a = diff(x_a);      
        for k = 1:n_frame-1
           FSIG_a(k) = norm(f_a(k,:),2);
        end
        [~,n] = size(FSIG_a);
        FSIG_a_mean(num_s) = mean(FSIG_a);
        FSIG_a_var(num_s) = var(FSIG_a);
        FSIG_front_mean(num_s) = mean(FSIG_a(1:round(n/2)));
        FSIG_front_var(num_s) = var(FSIG_a(1:round(n/2)));
        FSIG_back_mean(num_s) = mean(FSIG_a(round(n/2):n));
        FSIG_back_var(num_s) = var(FSIG_a(round(n/2):n));
        
        sequence_name(num_s) = string(file(j).name);
    else
        num_w = num_w+1;
    end
    
    delete('test/*.yuv','test/*.png','test/*.bin');

end
FSIG_a_mean = FSIG_a_mean';
FSIG_a_var = FSIG_a_var';
FSIG_front_mean = FSIG_front_mean';
FSIG_front_var = FSIG_front_var';
FSIG_back_mean = FSIG_back_mean';
FSIG_back_var = FSIG_back_var';
sequence_name = sequence_name';
datacolumns = {'Name','FSIG_a_mean','FSIG_a_var','FSIG_front_mean','FSIG_front_var','FSIG_back_mean','FSIG_back_var'};
data = table(sequence_name,FSIG_a_mean,FSIG_a_var,FSIG_front_mean,FSIG_front_var,FSIG_back_mean,FSIG_back_var);
writetable(data, 'feature_FSIG.csv');
