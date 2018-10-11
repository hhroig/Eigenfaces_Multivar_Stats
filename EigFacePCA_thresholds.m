function [threshold, TPR, FPR, All_Classes, All_Eigenfaces, class_name,...
   all_Psi, All_Dist_Fspace] = EigFacePCA_thresholds(show_ave, show_4eig, show_hist, show_roc)

% ------------------------------------------------
% "EigFacePCA_thresholds": Calculates, via leave one out, 6 different
% blocks for face recognition. It divides the data into train and test sets
% in order to correctly determine de PCA of the matrix of training faces.
% This splitting also alows the user to study the capacity of the
% Eigenfaces to classify the test set of images as real faces and return 
% the class to which they belong. The accuracy of the classification is
% studied through the distances of each image to the face-space and to
% every face-class.
% The data must be in a Folder with name "caras"!
% 
% Harold Antionio Hernández Roig 
% hahernan@est-econ.uc3m.es
% ------------------------------------------------

%% Loading Data
 root = cd;
 cd 'caras'

names = dir('*.bmp');
data=[];
for k=1:length(names)
        disp(['Reading file ' names(k).name]); 
         Im = imread(names(k).name);
         
        % working without colors
        Im_gray = rgb2gray(Im);        
        Im_gray = mat2gray(Im_gray); % converts the same image into a [0,1] gray-scale!
        
%         working with colors (not used: grey-scale reduces computational cost and improves accuracy)
%         Im = double(Im);   
%                 
%         red = Im(:,:,1);
%         green = Im(:,:,2);
%         blue = Im(:,:,3);
%         
%         temp = [red(:); green(:); blue(:)];
%         data(:,k) = temp; % every column is a face        
                 
        data(:,k) = Im_gray(:); % every column is a face
end

%% Construction of 3D matrix of faces...

G = data; % matrix of training faces: [G1 G2, G3, ..., GM], every column Gi is a face.
[N,M] = size(G);  % for me this N pixels of image and M subjects times (6) pics per subjects

A3D = zeros(N,6,M/6); % Similiar as A: every row (p) is a pixel, every column in 2nd dimension (t) is a type of face and every number in 3rd dim. (s) represents a subject
for i=0:99
    A3D(:,:,i+1) = G(:,6*i+1:6*(i+1));
end

[p,t,s] = size(A3D); % p = pixels in rows, t = type of face, s = subject

% names of files:
images_names = {names.name};
images_names = reshape(images_names, t, s); % name of files: rows for each face type, cols. for each subject
class_name = cell(1,s);
for i = 1:s
class_name{i} = images_names{1,i}(1:5);
end

%% Leave One Face Out!!!

All_Classes = cell(1,6);
All_Eigenfaces = cell(1,6);

quant_dist_fspace = 1:-.01:.95;
All_Dist_Fspace = zeros(length(quant_dist_fspace),t);

quant_thr = 1:-0.05:0;
threshold = zeros(length(quant_thr),t); % maimum allowable distance from any face class
TPR = zeros(size(threshold));
FPR = zeros(size(threshold));
all_Psi = zeros(p,t);

for ind_test = 1:t % "leave one face out"
    
    Test = reshape(A3D(:,ind_test,:),p,s); % test set of faces: is a vector due to leave one out
    
    A_train = A3D;
    A_train(:,ind_test,:) = []; % train set of faces in 3D
    A = reshape(A_train, p, (t-1)*100); % train set of faces in 2D
    
    Psi = mean(A,2); % average face of training set
    all_Psi(:,ind_test) = Psi;
     
    if show_ave % optional plot of ave-face
    figure
    ave_train_face = reshape(mat2gray(Psi),165,120);
    imshow(ave_train_face)
    title(['Average Face of Training Set: Leave Out Face No.',num2str(ind_test)])
    end
    
    A = A - repmat(Psi,[1,(t-1)*100]);
    A_train = A_train - repmat(Psi,[1,t-1,s]);
    Test = Test - repmat(Psi,[1,s]);

%% Eigenvalues/vectors of Transpose Matrix

L = A'*A; % MxM matrix from which we extract the eigenvectors...

[v, mu_D] = eig(L);

mu = diag(mu_D); % the corresponding eigenvalues
aux = find(mu > 0.00001);

% let's take out the corresponding to mu = 0, using the "aux" vector:
mu = mu(aux);
v = v(:,aux);

u = A*v; % then each column u_i of u is an eigenvector of C = A*A'

% normalize the eigenvectors?
for i = 1:size(u,2)
    u(:,i) = u(:,i)/sqrt(u(:,i)'*u(:,i)); 
end

% let's sort the eigenvalues (then the corresponding eigenvectors)
[mu,I] = sort(mu,'descend');
u = u(:,I);

%% Decide how many Principal Components to keep (number k)

% We fixed the approach of retaining those who explain the bigger amount of
% variance:
trace = sum(mu);
k = 0; 
variance_kept = 0;
while variance_kept < 0.95
    k = k+1;
    variance_kept = sum(mu(1:k))/trace;
end
% Second Option would be to decide graphically with a scree plot, but is
% not practical for a learning algoritm...
% % scree plot:
% variance_explained = zeros(size(mu));
% for kk = 1:length(mu)
%     variance_explained(kk) = sum(mu(1:kk))/trace;
% end
% plot(variance_explained)

U = u(:,1:k); % we keep in U = [u1, ..., uk] the "k" significant eigenfaces
All_Eigenfaces{ind_test} =  U;

% Lets check out the firsts eigenfaces:
if show_4eig % optional plot of first 4 eigenfaces...
figure;
eigenface1 = reshape(mat2gray(U(:,1)),165,120);
subplot(2,2,1),imshow(eigenface1)
title(['1st Eigenface. Leave Out Face No.',num2str(ind_test)])
eigenface2 = reshape(mat2gray(U(:,2)),165,120);
subplot(2,2,2),imshow(eigenface2)
title(['2nd Eigenfaces Leave Out Face No.',num2str(ind_test)])
eigenface3 = reshape(mat2gray(U(:,3)),165,120);
subplot(2,2,3),imshow(eigenface3)
title(['3rd Eigenface. Leave Out Face No.',num2str(ind_test)])
eigenface4 = reshape(mat2gray(U(:,4)),165,120);
subplot(2,2,4),imshow(eigenface4)
title(['4th Eigenface. Leave Out Face No.',num2str(ind_test)])
end

%% Construct the Face Classes (Omega_i)
% note: we have 5 test images per person due to leave one out, 100 indiv. and we can
% calculate Omega_i, i = 1,..., 100 by averaging the results of the eigenface
% representation over a small number of face images of each individual
% (those in the training set)

% I already have the normalized train faces in A3D, A, A_train, etc...

W = zeros(k,t-1,s); % projection of each face in "face space"
for slice = 1:s
for i = 1:t-1
    W(:,i,slice) = U'*A_train(:,i,slice);
end
end

% Each column of aveW is an Omega_i, i=1,...,100
W_classes = zeros(k, s);
for slice = 1:s
    W_classes(:,slice) = mean(W(:,:,slice),2);
end

All_Classes{ind_test} =  W_classes;

%% Processing New Faces... dealing with the test set "Test"
% We already have the (normalize by Psi) test data of faces in Test

% Extract eigenface components of test face:
W_test = []; % eigenface components of each test face 
for i = 1:s
    W_test = [W_test U'*Test(:,i)];
end

% Builiding the Matrix D(100 x 100) of distances from each individual in 
% W_test to each of the 100 classes:

D = zeros(s,s); % each d_ij is distance of subject i to class j
for i = 1:s
    for j = 1:s
        D(i,j) = norm(W_test(:,i)-W_classes(:,j));
    end
end

real_dist = diag(D);

if show_hist % optional plot of histogram of dist. to real class...
figure
histogram(real_dist)
title(['Histogram of Distances from Real Classes: Leave Out Face No.',num2str(ind_test)])
end

% Decision threshold
threshold(:,ind_test) = quantile(real_dist, quant_thr); % we fixed a % of well classified!

for q = 1:length(quant_thr)
logical_D = (D <= threshold(q, ind_test));
TPR(q,ind_test) = mean(diag(logical_D)); % Well classified:  same as threshold(ind_test) and is fixed!
FPR(q,ind_test) = ( sum(logical_D(:)) - sum(diag(logical_D)) )/(100^2) ; % Falsely classified.
end

if show_roc
figure
plot(FPR(:,ind_test),TPR(:,ind_test), FPR(:,ind_test),TPR(:,ind_test), 'o')
legend('ROC Curve', 'Actual rates for diferent thresholds')
title(['ROC Curve for Different Thresholds: Leave Out Face No.',num2str(ind_test)])
xlabel('FPR')
ylabel('TPR: fixed by user!')
end

% Distances to Face-Space
dist_fspace = [];
for face = 1:s
    dist_fspace = [dist_fspace norm( Test(:,face) - U*W_test(:,face) )]; % U*W_test(:,sub) is projection of face "sub" onto face-space    
end

if show_hist % optional plot of histogram of dist. to face-space ...
figure
histogram(dist_fspace)
title(['Histogram of Distances from Face-Space: Leave Out Face No.',num2str(ind_test)])
end

All_Dist_Fspace(:,ind_test) = quantile(dist_fspace, quant_dist_fspace); % we only save the quantiles

end % of leave one out
cd(root)
end % of function !!!