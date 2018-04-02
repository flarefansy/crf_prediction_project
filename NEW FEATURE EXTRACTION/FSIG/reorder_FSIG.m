clear; clc;

clear;
fid = fopen('AvgFeaturesAdded.csv'); 
dcells = textscan(fid,'%s');  
fclose(fid);  

B = sortrows(dcells{1,1});
b={};
se = {};
a = {}
fid=fopen('AvgFeaturesAdded_record.csv','w');
for i=1:size(B,1)-1
    str_s = strsplit(B{i},'_');
    loc = find(strcmp(str_s(),'000')==1|strcmp(str_s(),'001')==1|strcmp(str_s(),'002')==1|strcmp(str_s(),'003')==1|strcmp(str_s(),'004')==1|strcmp(str_s(),'005')==1|strcmp(str_s(),'006')==1);
    if strcmp(str_s(1:loc,1),se(1,1:loc))
    start_f{i} = str2num(char(str_s{loc+1}));
    se{i} = strcat(str_s{1:loc});
end
for j=1:size(B,1)-1
    loc_se = find(ismember(se,se{j})==1);
    [~,loc]=sort(cell2mat(start_f(loc_se)),'ascend');
    b(loc_se)=B(loc+loc_se(1)-1);
end
b = b';
% %     for j=1:size(B,1)
% %         se = strsplit(B{j},'_')
% %         if strcmp(str_s(1:loc-1,1),se(1,1:loc-1))
% %             num = j;
% %         end
% for j=1:size(b,1)
%     aa((j-1)*33+1:j*33) = b(j);
% end
% aa=aa';
% for j=1:size(aa,1)
%     a = aa(j);
%     a = cell2mat(a);
%     fprintf(fid,'%s\n',a);
% end
for j=1:size(b,1)
    a = b(j);
    a = cell2mat(a);
    fprintf(fid,'%s\n',a);
end
