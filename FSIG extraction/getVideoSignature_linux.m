

file = dir('mv/*.bin');
[num_sequence a] = size(file);
resolution = '854x480_HW_Q3_CRF10';
num_s = 0;
num_w = 0;
for j = 1:num_sequence
    a = strfind(file(j).name,resolution);
    if isempty(a) == 0
        num_s = num_s+1;

    %%
        %%ffmpeg
        copyfile(sprintf('mv/%s',file(j).name),'test/1.bin');
        system('ffmpeg -i test/1.bin test/out.yuv');
        system('ffmpeg -s 854x480 -i test/out.yuv test/out_%d.png');

        % function FSIG_a_mean = getVideoSignature()
        %%
        %%FSIG
        load video_thumbnail_model.mat; 
        num_frame = dir('test/*.png');
        [n_frame aa] = size(num_frame);
        kd = 8;
        for k=1:n_frame
           im = imread(sprintf('test/%s_%d.png', 'out', k));
           im_data = single(rgb2gray(im));
           [h,w] = size(im_data);
           im_data_thumbnails_lefttop = imresize(im_data(1:h/2,1:w/2),[12,16]);
           im_data_thumbnails_righttop = imresize(im_data(1:h/2,w/2+1:w),[12,16]);
           im_data_thumbnails_leftbot = imresize(im_data(h/2+1:h,1:w/2),[12,16]);
           im_data_thumbnails_rightbot = imresize(im_data(h/2+1:h,w/2+1:w),[12,16]);
           im_data_thumbnails = imresize(im_data,[12,16]);
           x_a(k,:) = im_data_thumbnails(:)'*A(:,1:kd); 
           x_lt(k,:) = im_data_thumbnails_lefttop(:)'*A(:,1:kd); 
           x_lb(k,:) = im_data_thumbnails_leftbot(:)'*A(:,1:kd); 
           x_rt(k,:) = im_data_thumbnails_righttop(:)'*A(:,1:kd); 
           x_rb(k,:) = im_data_thumbnails_rightbot(:)'*A(:,1:kd); 
        end
        f_a = diff(x_a);
        f_lt = diff(x_lt);
        f_lb = diff(x_lb);
        f_rt = diff(x_rt);
        f_rb = diff(x_rb);
        for k = 1:n_frame-1
            FSIG_a(k) = norm(f_a(k,:),2);
            FSIG_lt(k) = norm(f_lt(k,:),2);
            FSIG_lb(k) = norm(f_lb(k,:),2);
            FSIG_rt(k) = norm(f_rt(k,:),2);
            FSIG_rb(k) = norm(f_rb(k,:),2);
        end
        FSIG_a_mean(num_s) = mean(FSIG_a);
        FSIG_lt_mean(num_s) = mean(FSIG_lt);
        FSIG_lb_mean(num_s) = mean(FSIG_lb);
        FSIG_rt_mean(num_s) = mean(FSIG_rt);
        FSIG_rb_mean(num_s) = mean(FSIG_rb);
        FSIG_a_var(num_s) = var(FSIG_a);
        FSIG_lt_var(num_s) = var(FSIG_lt);
        FSIG_lb_var(num_s) = var(FSIG_lb);
        FSIG_rt_var(num_s) = var(FSIG_rt);
        FSIG_rb_var(num_s) = var(FSIG_rb);
        sequence_name(num_s) = string(file(j).name);
    

%     % k = 1:n_frame-1;
%     % figure(1);
%     % plot(k,FSIG_a);
%     % figure(2);
%     % plot(k,FSIG_lt);
%     % figure(3);
%     % plot(k,FSIG_lb);
%     % figure(4);
%     % plot(k,FSIG_rt);
%     % figure(5);
%     % plot(k,FSIG_rb);
%     %return;
%     % end
    else
        num_w = num_w+1;
    end
    
    delete('test/*.yuv','test/*.png','test/*.bin');

end
% 
FSIG_a_mean = FSIG_a_mean';
FSIG_lt_mean = FSIG_lt_mean';
FSIG_lb_mean = FSIG_lb_mean';
FSIG_rt_mean = FSIG_rt_mean';
FSIG_rb_mean = FSIG_rb_mean';
FSIG_a_var = FSIG_a_var';
FSIG_lt_var = FSIG_lt_var';
FSIG_lb_var = FSIG_lb_var';
FSIG_rt_var = FSIG_rt_var';
FSIG_rb_var = FSIG_rb_var';
sequence_name = sequence_name';
datacolumns = {'Name','FSIG_a_mean','FSIG_a_var','FSIG_lt_mean','FSIG_lt_var','FSIG_lb_mean','FSIG_lb_var','FSIG_rt_mean','FSIG_rt_var','FSIG_rb_mean','FSIG_rb_var'};
data = table(sequence_name,FSIG_a_mean,FSIG_a_var,FSIG_lt_mean,FSIG_lt_var,FSIG_lb_mean,FSIG_lb_var,FSIG_rt_mean,FSIG_rt_var,FSIG_rb_mean,FSIG_rb_var);
writetable(data, 'feature.csv');
