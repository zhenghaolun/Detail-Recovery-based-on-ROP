function M2
% Rank1 - Rank-One Prior: toward Real-Time Scene Recovery
% DEMO of RANK1; Matlab 2018a and higher verision
% @ ImageType is the type of the input image
% @ 1: hazy image; 
% @ 2: sandstorm image; 
% @ 3: underwaterimage
%            
%   The Code is created based on the method described in the following paper 
%   [1] "Rank-One Prior: Toward Real-Time Scene Recovery", Jun Liu, Ryan Wen Liu, Jianing Sun, Tieyong Zeng, IEEE International Conference on Computer Vision (CVPR), 2021, 14802-14810.
%  
%   Author: Jianing Sun (sunjn118@nenu.edu.cn), Jun Liu (liuj292@nenu.edu.cn)
%   Version : 1.0 
%
% The code and the algorithm are for non-commercial use only. 
%
% If you are inteseted in commercial purpose, please comply with the patent: ZL 202011281893.3 and contact the authors, thank you!

omega    = 0.8;
% 设置参数
sigma_s = 20;  % 替换为您的值
sigma_r = 0.4;  % 替换为您的值

%dir = 'images';
imgname = "C:\Users\Lenovo\Desktop\images\009.png";
%img     = im2double(imread([dir '/' imgname]));
img = im2double(imread(imgname));

%if gpuDeviceCount
    %img = gpuArray(img); % use GPU to save much more time for an image of big size 
%end

% 读取图像
%originalImage = im2double(imread("C:\Users\Lenovo\Desktop\Test\Underwater.jpg"));

% 定义雾气厚度
%hazeStrength = 0.5; % 调整此值以控制雾气效果的强度

% 创建全白图像
%whiteImage = ones(size(originalImage));

% 创建全绿图像
%greenImage = ones(size(originalImage));
%greenImage(:,:,2) = 1; % 设置绿色通道为1，其它通道为0

% 创建全黄图像
%yellowImage = ones(size(originalImage));
%yellowImage(:,:,3) = 0; % 将蓝色通道设置为0，保持红色和绿色通道为1

% 混合原始图像和全白图像
%hazedImage = originalImage * (1 - hazeStrength) + greenImage * hazeStrength;

 
tic
f = @(x) adaptive_manifold_filter(x, sigma_s, sigma_r);
P = img;
for n=1:1
  f1P = f(P);
  f2P = f(f1P);
  f3P = f(f2P);
  f4P = f(f3P);
  P = 4*P - 6*f1P + 4*f2P - f3P; % 4-polyblur
end
[J,s_tildeT,tildeT] = rank1_enhancement(img,omega);
r=CT(double(J)/255,double(P)/255);
toc

figure('NumberTitle','off','name','Enhancement');
imshow(r);
truesize();

% 读取图像
%TestImage = im2double(imread("C:\Users\Lenovo\Desktop\Test\Haze.png"));

% 获取第一张图片的尺寸
%targetSize = size(TestImage);

% 调整第二张图片的大小为第一张图片的尺寸
%ProcessedImage = imresize(ProcessedImage, targetSize(1:2));

% 计算峰值信噪比（PSNR）
%psnrValue = psnr(r, originalImage);

% 计算结构相似性指标（SSIM）
%ssimValue = ssim(r, originalImage);

% 计算视觉感知模型（VIF）指标
%vifValue = VIF(originalImage, r);

% IMMSE
%mseValue = immse(r, originalImage);

% BRISQUE
brisqueValue = brisque(r);

% NIQE
niqeValue = niqe(r);

% PIQE
piqeValue = piqe(r);


% 显示结果
%fprintf('PSNR值: %.2f dB\n', psnrValue);
%fprintf('SSIM值: %.4f\n', ssimValue);
%fprintf('VIF值: %.4f\n', vifValue);
%fprintf('MSE值: %.4f\n', mseValue);
fprintf('BRISQUE值: %.4f\n', brisqueValue);
fprintf('NIQE值: %.4f\n', niqeValue);
fprintf('PIQE值: %.4f\n', piqeValue);

%  save the results
%if gpuDeviceCount
    %imwrite(gather(J),['results/' imgname(1:end-4) '-Rank1.png']);
%else 
    %imwrite( J,['results/'  imgname(1:end-4) '-Rank1.png'],'png'); % recovered scene
    %imwrite( s_tildeT,['results/'  imgname(1:end-4) '-sT-Rank1.png'],'png'); % resampled T
    %imwrite( tildeT,['results/'  imgname(1:end-4) '-T-Rank1.png'],'png'); % T0
%end


function [Jr_ini,s_tildeT,tildeT] = rank1_enhancement(img,omega)
% for CVPR 2021
% % %  Optimize the scattering light

imgvec    = reshape(img, size(img,1)*size(img,2), 3);

x_RGB(1 ,1, 1:3) =  mean(imgvec,1); % unified spectrum
% %%%%%%  direction difference
x_mean   = repmat( x_RGB,[ size(img,1) size(img,2) 1 ] ); % unified spectrum in each pixel
%%%%%%%%%%%
scat_basis = x_mean ./max( sqrt(sum(x_mean.^2,3)), 0.001); % normalization
fog_basis  = img ./max( sqrt(sum(img.^2,3)), 0.001); % normalization
cs_sim   = repmat( (((sum( scat_basis .* fog_basis,3) ))),[1 1 3] ); % cos similarity
% %%%%%% scattering_light_estimation
scattering_light  = (cs_sim) .* (sum(img,3)./max( sum(x_mean,3), 0.001)).*x_mean;

intial_img =  img;
%%%  get_atmosphere
[ atmosphere, scattering_light ]   = get_atmosphere(  intial_img, scattering_light);
%%%  get_transmission_estimate
T       =  1 - omega * scattering_light ; % T = 1 - \tilde{t}
T_ini = scattering_mask_sample( T );
 
%%%  dehaze
ShowR_d      = ( intial_img - atmosphere )   ./max( T_ini ,0.001 ) + atmosphere; % Winv.*

 

% %    Postprocessing;  module for luminance adjustment,
% %    For some thick-fog scenes, this operation is not recommended;
mi = prctile2019(ShowR_d,1,[1 2]);
ma = prctile2019(ShowR_d,99,[1 2]);
Jr_ini = ( ShowR_d - mi)./(ma-mi);


Jr_ini = double(gamma0( uint8( Jr_ini*255 ) ));

tildeT = 1 - T;
s_tildeT = 1 - T_ini;

 
    function [ atmosphere, scatterlight ] = get_atmosphere( image,scatterlight )

        for i = 1:1
            scatter_est = sum(scatterlight,3);
            n_pixels = numel(scatter_est);
            n_search_pixels = floor( n_pixels * 0.001);
            image_vec = reshape(image, n_pixels, 3);
            [~, indices] = sort(scatter_est(:), 'descend');
            atmosphere = mean( image_vec( indices(1:n_search_pixels), : ), 1);

            atmos(1,1,:) = atmosphere;
            atmosphere = repmat( atmos, [ size(scatter_est) 1 ] );

            sek = scatter_est(indices(n_search_pixels));

            scatterlight = scatterlight .* repmat( scatter_est <= sek, [ 1 1 3] ) + ...
                ( 2/3 * sek -scatterlight ) .* repmat( scatter_est > sek, [ 1 1 3] );
        end
    end
end

 
%% 

function img = gamma0(img)
i = 0 : 255;
f = ((i + 0.5)./256 ).^(5/6);
LUT(i+1) = uint8( f.*256 -0.5 );

%%%%  rgb2hsv  - hsv2rgb            rgb2ycbcr-ycbcr2rgb
img = rgb2ycbcr(img);
img(:,:,1)    = LUT( img(:,:,1) + 1 );
img = ycbcr2rgb(img);

end


function F = scattering_mask_sample(I)
mSize = min(size(I,1),size(I,2));
if mSize < 800
    r = 0.02;
elseif mSize >= 800 && mSize < 1500
    r = 0.01;
else
    r = 0.005;
end
I0  = imresize(I,r);
F = imresize(I0,[size(I,1),size(I,2)],'bicubic');
end


    function y = prctile2019(varargin)
    par = inputParser();
    par.addRequired('x');
    par.addRequired('p');
    par.addOptional('dim',1,@(x) isnumeric(x) || validateDimAll(x));
    par.addParameter('Delta',1e3);
    par.addParameter('RandStream',[]);

    par.parse(varargin{:});

    x = par.Results.x;
    p = par.Results.p;
    dim = par.Results.dim;
    delta = par.Results.Delta;
    rs = par.Results.RandStream;


    % Figure out which dimension prctile will work along.
    sz = size(x);


    % Permute the array so that the requested dimension is the first dim.
    if ~isequal(dim,1)
        nDimsX = ndims(x);
        dim = sort(dim);
        perm = [dim setdiff(1:max(nDimsX,max(dim)),dim)];
        x = permute(x, perm);
    end
    sz = size(x);
    dimArgGiven = true;



    % Drop X's leading singleton dims, and combine its trailing dims.  This
    % leaves a matrix, and we can work along columns.
    work_dim = 1:numel(dim);

    work_dim = work_dim(work_dim <= numel(sz));
    nrows = prod(sz(work_dim));
    ncols = numel(x) ./ nrows;
    x = reshape(x, nrows, ncols);

    x = sort(x,1);
    n = sum(~isnan(x), 1); % Number of non-NaN values in each column

    % For columns with no valid data, set n=1 to get nan in the result
    n(n==0) = 1;

    % If the number of non-nans in each column is the same, do all cols at once.
    if all(n == n(1))
        n = n(1);
        if isequal(p,50) % make the median fast
            if rem(n,2) % n is odd
                y = x((n+1)/2,:);
            else        % n is even
                y = (x(n/2,:) + x(n/2+1,:))/2;
            end
        else
            y = interpColsSame(x,p,n);
        end

    else
        % Get percentiles of the non-NaN values in each column.
        y = interpColsDiffer(x,p,n);
    end


    % Reshape Y to conform to X's original shape and size.
    szout = sz;
    szout(work_dim) = 1;
    szout(work_dim(1)) = numel(p);
    y = reshape(y,szout);

    % undo the DIM permutation
    if dimArgGiven && ~isequal(dim,1)
        y = ipermute(y,perm);
    end



        function y = interpColsSame(x, p, n)

            if isrow(p)
                p = p';
            end

            % Form the vector of index values (numel(p) x 1)
            r = (p/100)*n;
            k = floor(r+0.5); % K gives the index for the row just before r
            kp1 = k + 1;      % K+1 gives the index for the row just after r
            r = r - k;        % R is the ratio between the K and K+1 rows

            % Find indices that are out of the range 1 to n and cap them
            k(k<1 | isnan(k)) = 1;
            kp1 = bsxfun( @min, kp1, n );

            % Use simple linear interpolation for the valid percentages
            y = (0.5+r).*x(kp1,:)+(0.5-r).*x(k,:);

            % Make sure that values we hit exactly are copied rather than interpolated
            exact = (r==-0.5);
            if any(exact)
                y(exact,:) = x(k(exact),:);
            end

            % Make sure that identical values are copied rather than interpolated
            same = (x(k,:)==x(kp1,:));
            if any(same(:))
                x = x(k,:); % expand x
                y(same) = x(same);
            end
        end

        function y = interpColsDiffer(x, p, n)
            %INTERPCOLSDIFFER A simple 1-D linear interpolation of columns that can
            %deal with columns with differing numbers of valid entries (vector n).

            [nrows, ncols] = size(x);

            % Make p a column vector. n is already a row vector with ncols columns.
            if isrow(p)
                p = p';
            end

            % Form the grid of index values (numel(p) x numel(n))
            r = (p/100)*n;
            k = floor(r+0.5); % K gives the index for the row just before r
            kp1 = k + 1;      % K+1 gives the index for the row just after r
            r = r - k;        % R is the ratio between the K and K+1 rows

            % Find indices that are out of the range 1 to n and cap them
            k(k<1 | isnan(k)) = 1;
            kp1 = bsxfun( @min, kp1, n );

            % Convert K and Kp1 into linear indices
            offset = nrows*(0:ncols-1);
            k = bsxfun( @plus, k, offset );
            kp1 = bsxfun( @plus, kp1, offset );

            % Use simple linear interpolation for the valid percentages.
            % Note that NaNs in r produce NaN rows.
            y = (0.5-r).*x(k) + (0.5+r).*x(kp1);

            % Make sure that values we hit exactly are copied rather than interpolated
            exact = (r==-0.5);
            if any(exact(:))
                y(exact) = x(k(exact));
            end

            % Make sure that identical values are copied rather than interpolated
            same = (x(k)==x(kp1));
            if any(same(:))
                x = x(k); % expand x
                y(same) = x(same);
            end
        end

        function bool = validateDimAll(dim)
            bool = ((ischar(dim) && isrow(dim)) || ...
                (isstring(dim) && isscalar(dim) && (strlength(dim) > 0))) && ...
                strncmpi(dim,'all',max(strlength(dim), 1));
        end
    end
end