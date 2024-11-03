% 读取图像
f = im2double(imread("C:\Users\Lenovo\Desktop\images\009.png"));

% 设置参数
sigma_s = 20;  % 替换为您的值
sigma_r = 0.4;  % 替换为您的值

% 调用自适应流形滤波器函数
tic
[g, tilde_g] = adaptive_manifold_filter(f, sigma_s, sigma_r);

% 显示锐化后的图像
result = f+5*(f-g)
toc

imshow(g);
imshow(result);  % 或者 imshow(tilde_g);


% 等待用户关闭图像窗口
waitfor(gcf);