im = im2double(imread("C:\Users\Lenovo\Desktop\images\009.png"));
sigma_s = 20; % 示例的空间标准差
sigma_r = 0.4; % 示例的范围标准差
f = @(x) adaptive_manifold_filter(x, sigma_s, sigma_r);
P = im;
for n=1:1
  f1P = f(P);
  f2P = f(f1P);
  f3P = f(f2P);
  f4P = f(f3P);
  P = 4*P - 6*f1P + 4*f2P - f3P; % 4-polyblur
end

imshow(P);