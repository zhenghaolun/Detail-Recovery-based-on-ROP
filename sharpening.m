% ��ȡͼ��
f = im2double(imread("C:\Users\Lenovo\Desktop\images\009.png"));

% ���ò���
sigma_s = 20;  % �滻Ϊ����ֵ
sigma_r = 0.4;  % �滻Ϊ����ֵ

% ��������Ӧ�����˲�������
tic
[g, tilde_g] = adaptive_manifold_filter(f, sigma_s, sigma_r);

% ��ʾ�񻯺��ͼ��
result = f+5*(f-g)
toc

imshow(g);
imshow(result);  % ���� imshow(tilde_g);


% �ȴ��û��ر�ͼ�񴰿�
waitfor(gcf);