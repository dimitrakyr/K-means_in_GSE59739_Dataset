clc; clear all; close all;

fprintf('K-means in GSE59739 Dataset \n\n')
pause(1)
fprintf('by Dimitra Kyriakouli \n\n')
pause(1)

% GSE59739 Analysis
fprintf('GSE59739 Analysis \n\n')

GEO = load("GSE59739.mat");
GEOT = tabulate(GEO.class)
data = GEO.data;
clear GEO

% K-means comparison using different distance metrics
fprintf('K-means comparison using different distance metrics\n\n')

rng(45)
idx_sqe = kmeans(data,3,'Distance','sqeuclidean');
figure(1)
subplot(2,2,1)
[eval_sqe, h1] = silhouette(data, idx_sqe);
title('Sqeuclidean Distance')
xlabel('Silhouette Value')
ylabel('Cluster')
res_sqe = mean(eval_sqe);
fprintf("Mean of Silhouette Values - Sqeuclidean Distance: %f\n", res_sqe)

rng(45)
idx_cit = kmeans(data,3,'Distance','cityblock');
subplot(2,2,2)
[eval_cit, h2] = silhouette(data, idx_cit);
title('Cityblock Distance')
xlabel('Silhouette Value')
ylabel('Cluster')
res_cit = mean(eval_cit);
fprintf("Mean of Silhouette Values - Cityblock Distance: %f\n", res_cit)

rng(45)
idx_cos = kmeans(data,3,'Distance','cosine');
subplot(2,2,3)
[eval_cos, h3] = silhouette(data, idx_cos);
title('Cosine Distance')
xlabel('Silhouette Value')
ylabel('Cluster')
res_cos = mean(eval_cos);
fprintf("Mean of Silhouette Values - Cosine Distance: %f\n", res_cos)

rng(45)
idx_cor = kmeans(data,3,'Distance','correlation');
subplot(2,2,4)
[eval_cor, h4] = silhouette(data, idx_cor);
title('Correlation Distance')
xlabel('Silhouette Value')
ylabel('Cluster')
res_cor = mean(eval_cor);
fprintf("Mean of Silhouette Values - Correlation Distance: %f\n", res_cor)

% K-means comparison using different number of clusters
fprintf('\nK-means comparison using different number of clusters \n\n')

for k = 2:10
rng(45)
idx = kmeans(data,k);
figure(2)
subplot(3,3,k-1)
[eval, h] = silhouette(data, idx);
title(num2str(k),' Clusters')
xlabel('Silhouette Value')
ylabel('Cluster')
res(k-1) = mean(eval);
fprintf("Mean of Silhouette Values for %d Clusters: %f \n", k, res(k-1))
end

% Comparison with dataset after PCA
fprintf('\nComparison with dataset after PCA  - 3 clusters\n\n')

rng(45)
idx4a = kmeans(data,3);
figure(3)
subplot(1,2,1)
[eval4a, h4a] = silhouette(data, idx4a);
title('3 Clusters - Original Dataset')
xlabel('Silhouette Value')
ylabel('Cluster')
res4a = mean(eval4a);
fprintf("Mean of Silhouette Values for 3 Clusters - Original Dataset: %f \n", res4a)

data_norm = zscore(data);
[pc_coeff, pc, ~, ~, explained, ~] = pca(data_norm);
pcadata = pc(:,1:2);

rng(45)
idx4b = kmeans(pcadata,3);
subplot(1,2,2)
[eval4b, h4b] = silhouette(data, idx4b);
title('3 Clusters - PCA Dataset')
xlabel('Silhouette Value')
ylabel('Cluster')
res4b = mean(eval4b);
fprintf("Mean of Silhouette Values for 3 Clusters - PCA Dataset: %f \n\n", res4b)

plotdata = [pcadata idx4a];
figure(4);
gscatter(plotdata(:,1), plotdata(:,2), plotdata(:,3));
title('Plotting First 2 Principal Components - Kmeans in Original Dataset')
xlabel('1st Principal Component');
ylabel('2nd Principal Component');

plotdata = [pcadata idx4b];
figure(5);
gscatter(plotdata(:,1), plotdata(:,2), plotdata(:,3));
title('Plotting First 2 Principal Components - Kmeans in PCA dataset')
xlabel('1st Principal Component');
ylabel('2nd Principal Component');

fprintf('\nPercentage of variance attributed to the 1st Principal Component: %f\n', explained(1))
fprintf('Percentage of variance attributed to the 2st Principal Component: %f', explained(2))

fprintf('\n\nEnd Of Programme')


