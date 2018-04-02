clear;

clear;

file = dir('mv/*.bin');
[num_sequence a] = size(file);
resolution = '854x480';
num_s = 0;
num_w = 0;

for j = 1:num_sequence
    a = strfind(file(j).name,resolution);
    if isempty(a) == 0
        num_s = num_s+1;
        
        copyfile(sprintf('mv/%s',file(j).name),'test/1.bin');
        
        system('./TAppDecoderStatic -b test/1.bin >1.txt');
        sequence_name(num_s) = string(file(j).name);
    else
        num_w = num_w+1;
    end
    
    delete('test/*.bin');
end

datacolumns = {'Name'};
data = table(sequence_name);
writetable(data,'second_order_features_name');