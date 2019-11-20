clear all;
%Mat = csvread('train.csv',1,2,[1 2 1000 4]);
Mat = csvread('train.csv',1,2);
typechar=zeros(1000,1); %initialize typechar

fileID = fopen('train.csv');%read train.csv
type=textscan(fileID,'%s %s %s %s %s %s','Delimiter',',','headerlines',1);

%celldisp(BldgType{2});
old=["1Fam" "2FmCon" "Duplx" "TwnhsE" "TwnhsI"]; %define building type
new=["1" "2" "3" "4" "5"]; %replace building type with no
for i =1: 5
    type{2}=strrep(type{2},old(i),new(i)); %string replace with number
end

typechar=cell2mat(type{1,2}); %get character
typeint=str2num(typechar); %convert character to number
x_data_original=[typeint Mat(:,1:3)]; %combine matrix as input
x_data=x_data_original;
y_data=Mat(:,4);

%plot3(x_data(:,3),x_data(:,4), y_data, '*');
title('training data');
xlabel('Ground living area');
ylabel('Garage area');
zlabel('Price $');
grid on;
format long g;
mean_train=mean(x_data);
var_train=var(x_data);

m = length(x_data(:,1));
x_data = [ones(m,1), x_data]; % x0 =1

%normalization
x_data(:,2) = (x_data(:,2) - mean(x_data(:,2))) ./ std(x_data(:,2)); %data normalization
x_data(:,3) = (x_data(:,3) - mean(x_data(:,3))) ./ std(x_data(:,3));
x_data(:,4) = (x_data(:,4) - mean(x_data(:,4))) ./ std(x_data(:,4));
x_data(:,5) = (x_data(:,5) - mean(x_data(:,5))) ./ std(x_data(:,5));

%gradient descent
iter = 1000; % iteration
theta = zeros(1,5); 
alpha = 0.004;
J = zeros(m,1);
for i = 1:iter     
    h = x_data * theta'; 
    J(i) = 1/2*m * sum((h - y_data).^2); % 
    % 
    theta(1,1) = theta(1,1) - alpha*(1/m)* sum((h - y_data) .* x_data(:,1)); % x(:,1)=x_0
    theta(1,2) = theta(1,2) - alpha*(1/m)* sum((h - y_data) .* x_data(:,2));
    theta(1,3) = theta(1,3) - alpha*(1/m)* sum((h - y_data) .* x_data(:,3));
    theta(1,4) = theta(1,4) - alpha*(1/m)* sum((h - y_data) .* x_data(:,4));
    theta(1,5) = theta(1,5) - alpha*(1/m)* sum((h - y_data) .* x_data(:,5));

end
theta;
%plot(1:iter, J);
title(['Loss function (Learning rate = ',num2str(alpha),')']);
%b = regress(y_data,x_data);
%import test data
[x_test]=testing11_14;
x_test_original=x_test;
x_test_original_1=x_test;
n=length(x_test(:,1));
x_test=[ones(n,1) x_test];
mean_test=mean(x_test);
var_test=var(x_test);
%normalization
x_test(:,2) = (x_test(:,2) - mean(x_test(:,2))) ./ std(x_test(:,2)); %data normalization
x_test(:,3) = (x_test(:,3) - mean(x_test(:,3))) ./ std(x_test(:,3));
x_test(:,4) = (x_test(:,4) - mean(x_test(:,4))) ./ std(x_test(:,4));
x_test(:,5) = (x_test(:,5) - mean(x_test(:,5))) ./ std(x_test(:,5));

y_test=round(x_test*theta');

%plot(x_test_original(:,4), y_test, '*');

%write into csv file
%csvwrite('test.csv',y_test,1,5);
%read submission
Mat = csvread('submission original.csv',1,0);
Mat=Mat(:,1);

%find index of price more than 250000
[row,~]=find(y_test>250000);
y_test_index_1=row; 
numbermorethan=length(y_test_index_1);

%find index of GarageArea > 700, 
x_test_original=[Mat x_test_original];
morethan250k = x_test_original(y_test_index_1,:); %combine id with testing data

findgarageArea_index = find(morethan250k(:,5)>700);
findgarageArea_index=morethan250k(findgarageArea_index,1);
numgreaterthan700 = length(findgarageArea_index);
%find GrLivArea > 2000
findgrlivarea_index=find(morethan250k(:,4)>2000);
numgreaterthan2000=length(findgrlivarea_index);
%find OverallQual > 8? 
findoverallqual_index=find(morethan250k(:,3)>8);
numgreaterthan8=length(findoverallqual_index);


fid=fopen('submission.csv','a'); %write into submission file
if fid<0
	errordlg('File creation failed','Error');
end

for i=1:259
	fprintf(fid,'%d, %d\n',Mat(i),y_test(i));
end
fclose(fid); 