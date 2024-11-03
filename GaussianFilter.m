% 读取图像
img = imread('example_image.jpg');

% 定义高斯滤波器的大小和标准差
filter_size = 5; % 滤波器大小
sigma = 1.5; % 标准差

% 创建高斯滤波器
gaussian_filter = fspecial('gaussian', filter_size, sigma);

% 应用高斯滤波器
smoothed_img = imfilter(img, gaussian_filter, 'replicate');

% 显示原始图像和平滑后的图像
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(smoothed_img);
title('Smoothed Image');
