function vif_value = VIF(image1, image2)
    % 将图像转换为灰度图像
    gray_image1 = rgb2gray(image1);
    gray_image2 = rgb2gray(image2);
    
    % 计算图像的局部对比度
    contrast_image1 = local_contrast(gray_image1);
    contrast_image2 = local_contrast(gray_image2);
    
    % 确保两个图像具有相同的大小
    %[rows, cols] = size(contrast_image1);
    %contrast_image2 = imresize(contrast_image2, [rows, cols]);
    
    % 计算图像的局部对比度和局部结构相似性的乘积
    product_contrast_ssim = contrast_image1 .* contrast_image2;
    
    % 计算局部对比度和局部结构相似性的均值
    mean_product_contrast_ssim = mean(product_contrast_ssim(:));
    
    % 计算局部对比度的均值
    mean_contrast_image1 = mean(contrast_image1(:));
    mean_contrast_image2 = mean(contrast_image2(:));
    
    % 计算 VIF 值
    vif_value = mean_product_contrast_ssim / (mean_contrast_image1 * mean_contrast_image2);
end

function contrast_image = local_contrast(gray_image)
    % 定义局部对比度的滤波器
    filter_size = 7; % 设置滤波器的大小
    filter_sigma = 7 / 6; % 设置滤波器的标准差
    filter = fspecial('gaussian', filter_size, filter_sigma);
    
    % 对图像应用局部对比度滤波器
    gray_image_double = im2double(gray_image);
    blurred_image = imfilter(gray_image_double, filter, 'replicate');
    diff_image = gray_image_double - blurred_image;
    contrast_image = sqrt(imfilter(diff_image.^2, filter, 'replicate'));
end


