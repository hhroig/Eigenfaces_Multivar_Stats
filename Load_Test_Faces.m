function [subjectID, data, file_name] = Load_Test_Faces
% ------------------------------------------------
% "Load_Test_Faces": Loads independent data set in folder "test". This one
% is thought to evaluate the classifier through a new data set of
% independent images with the same name structure and pixel ratio than the
% original database.
% 
% Harold Antionio Hernández Roig 
% hahernan@est-econ.uc3m.es
% ------------------------------------------------


% Load Test Data
 root = cd;
 cd 'test'

names = dir('*.bmp');
data=[];
subjectID = cell(1,length(names));
file_name = cell(1,length(names));

for k=1:length(names)
        disp(['Reading file ' names(k).name]); 
        subjectID{k} = names(k).name(1:5);        
        file_name{k} = names(k).name;
        
        Im = imread(names(k).name);
        
        % working without colors
        Im_gray = rgb2gray(Im);        
        Im_gray = mat2gray(Im_gray); % converts the same image into a [0,1] gray-scale!
                 
        data(:,k) = Im_gray(:); % every column is a face
end
cd(root)
end

