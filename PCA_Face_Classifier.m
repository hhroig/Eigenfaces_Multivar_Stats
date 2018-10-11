% ------------------------------------------------
% "PCA_Face_Classifier": It calls function "EigFacePCA_thresholds" to get
% the classifiers resulting from a "leave one out" learning splitting. It
% classifies new images from an independet data set. This code calls
% function "Load_Test_Faces" to acquire this independet data set. The face
% recognition procedure here presented calculates distances of every new
% image to the face-space in order to determine wether it is a face or not.
% It also computes the distances to each face-class in order to check if
% new individuals are known or unknonw.
% 
% Harold Antionio Hernández Roig 
% hahernan@est-econ.uc3m.es
% ------------------------------------------------

clear
close all
%% Initialize EigFaceBCA_thresholds function
show_ave = 0;  % show the average face...
show_4eig = 0; % show the first 4 eigenfaces...
show_hist = 0; % show histograms for distances from real classes and face-space
show_roc = 0;  % show the ROC curve for each leave one out

[threshold, TPR, FPR, All_Classes, All_Eigenfaces, class_name, Psi, All_Dist_Fspace] = ...
    EigFacePCA_thresholds(show_ave, show_4eig, show_hist, show_roc);

% After Analysis we choose the following set for clasification:

op_set = 1; % fixes which classifier (from all 6 leave-a-face-out) to use
U = All_Eigenfaces{op_set};
W_classes = All_Classes{op_set};

theta = threshold(3,op_set); % in row, we put the number for corresponding percentile (3 for 90%)
sigma = max(All_Dist_Fspace(1,:)); % maximum distance from face-sapace attained in "leave one out"

%% Load Test Data Set of New Faces

[subjectID, newFs, file_name] = Load_Test_Faces;
[p,s] = size(newFs); % p: pixels, s: # of subjects in Test

% Mean-Adjustment of new faces
Test = newFs - repmat(Psi(:, op_set), [1,s]);

%% Processing New Faces... dealing with the test set "Test"
% We already have the (normalize by Psi) test data of faces in Test

% Extract eigenface components of test face:
W_test = []; % eigenface components of each test face 
for i = 1:s
    W_test = [W_test U'*Test(:,i)];
end

% Distances to Face-Space
dist_fspace = [];
for face = 1:s
    dist_fspace = [dist_fspace norm( Test(:,face) - U*W_test(:,face) )]; % U*W_test(:,sub) is projection of face "sub" onto face-space    
end

% Out the non-faces!
out_no_face = dist_fspace > sigma;
no_faces = file_name(out_no_face);  % then we can have the complete file name of those "no-faces"
file_name(out_no_face) = [];
real_subjectsIDs = subjectID(~out_no_face);
s_real = length(real_subjectsIDs);
no_faces_rate = 1- s_real/s;
W_test(:,out_no_face) = [];

% Builiding the Matrix D of distances from each individual in 
% W_test to each of the 100 classes:

D = zeros(s_real,100); % each d_ij is distance of subject i to class j
for i = 1:s_real
    for j = 1:100
        D(i,j) = norm(W_test(:,i)-W_classes(:,j));
    end
end

% Find the face class that minimizes de Eucl. Dist:
[eps_k, location] = min(D,[],2);

% Once we have minumn distances we take out the unknown indiv. (eps_k >= theta)
thresholding = eps_k < theta;
eps_k(~thresholding) = [];
location(~thresholding) = [];
unknown_faces = file_name(~thresholding);
known_subjects = real_subjectsIDs(thresholding);

estimated_subject = class_name(location);

count_TP = 0;
for m = 1:length(location)
    if sum(known_subjects{m} == estimated_subject{m}) == 5
        count_TP = count_TP + 1;
    end
end

well_classiifed_rate = count_TP/s_real;
unknown_faces_rate = length(unknown_faces)/s_real;
    
%% Summary of Results:
disp('------------------------------------------------------')
disp('             ----- SUMMARY -----')
disp(['* There were ', num2str(s),' new images to test the algorithm;'])
disp(['and ', num2str(no_faces_rate*100),'% were not detected as faces.'])
disp(['* This reduces the classification to ', num2str(s_real),' faces.'])
disp(['* The rate of unknown faces is ', num2str(unknown_faces_rate*100),'% and;'])
disp(['* The rate of well classified subjects is ', num2str(well_classiifed_rate*100),'%.'])
disp('------------------------------------------------------')
