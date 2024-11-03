% 读取引导图像和目标图像
guide_img = imread('guide_image.jpg');
target_img = imread('target_image.jpg');

% 将图像转换为双精度浮点类型
guide_img = im2double(guide_img);
target_img = im2double(target_img);

% 定义引导滤波器的参数
radius = 5; % 滤波器半径
epsilon = 0.1^2; % 控制平滑度的参数

% 应用引导滤波器
filtered_img = imguidedfilter(target_img, guide_img, 'Radius', radius, 'DegreeOfSmoothing', epsilon);

% 显示原始图像和引导滤波器处理后的图像
figure;
subplot(1, 2, 1);
imshow(target_img);
title('Original Image');

subplot(1, 2, 2);
imshow(filtered_img);
title('Filtered Image');
