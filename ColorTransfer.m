%Demo
source="C:\Users\Lenovo\Desktop\images\084.png";
target="C:\Users\Lenovo\Desktop\2.png";

S=imread(source);
T=imread(target);

figure; imshow(S); title('source');
figure; imshow(T); title('target');
tic
r=CT(double(S)/255,double(T)/255);
toc
figure; imshow(r); title('result');

imwrite(r,'results.jpg');