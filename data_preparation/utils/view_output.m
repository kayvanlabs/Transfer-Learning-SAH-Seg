% Make Annotation
i = 25;
imgs = PatientsData_sel(i).adjustImgs_static;
masks = PatientsData_sel(i).masks;

new = uint8(zeros([512,512,size(imgs,3),3]));
new(:,:,:,1) = imgs;
new(:,:,:,2) = imgs;
new(:,:,:,3) = imgs;

[I,J,K] = ind2sub(size(masks),find(masks==255));

for i = 1:length(I)
    new(I(i),J(i),K(i),:) = [255;0;0];
end
new_mask = permute(new, [1,2,4,3]);

for i = 1:size(new_mask,4)
    imwrite(new_mask(:,:,:,i), ['/Users/apple/Documents/' num2str(i) '.jpg'])
end
%%
% Make Annotation
i = 11;
imgs = PatientsData_sel(i).adjustImgs_static;
masks = Pred;
masks = permute(masks, [3,2,1]);
new = uint8(zeros([512,512,size(imgs,3),3]));
new(:,:,:,1) = imgs;
new(:,:,:,2) = imgs;
new(:,:,:,3) = imgs;

[I,J,K] = ind2sub(size(masks),find(masks==1));

for i = 1:length(I)
    new(I(i),J(i),K(i),:) = [255;0;0];
end
new_pred = permute(new, [1,2,4,3]);

%% Merge
new = uint8(zeros([512,512,size(imgs,3),3]));
new(:,:,:,1) = imgs;
new(:,:,:,2) = imgs;
new(:,:,:,3) = imgs;
new = permute(new, [1,2,4,3]);

%x = cat(2,new, new_mask, new_pred);

%x = cat(2,x, new_pred);
%%


%% Show the images
row = 4;
col = size(new_mask,4)/4;
for i = 1:size(new_mask,4)
    subplot(row,col,i, 'align'); imshow(new_mask(:,:,:,i))
end


%%
i = 15;
info= PatientsData_sel(i).meta.info;
%%
for i = 1:size(new,4)
    dicomwrite(new(:,:,:,i), ['/Users/apple/Developer/output' num2str(i) '.dicom'], 'ObjectType', 'CT Image Storage')
end

%%
test =uint8(zeros(512,512,3));
test(:,:,1) = img;
test(:,:,2) = img;
test(:,:,3) = img;


%%
stats = regionprops3(masks,'PixelIdxList', 'area');
