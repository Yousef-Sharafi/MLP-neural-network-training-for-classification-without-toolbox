%Programmer: Yousef Sharafi

clc;
clear all;
close all;

data=xlsread('classification-iris.xlsX');

n=size(data,1);
m=size(data,2);

n1=m-1;
n2=30;
n3=20;
n4=1;

eta=0.02;
max_epoch=30;
a=-1;
b=+1;

rate_train=0.75;

data_train=[];
data_test=[];

for i=1:3
    index=find(data(:,n1+1)==i);
    num_train_class=round(numel(index)*rate_train);
    num_test_class=numel(index)-num_train_class;
    data_train=[data_train; data(index(1:num_train_class),:)];%#ok
    data_test=[data_test; data(index(num_train_class+1:end),:)];%#ok
end

num_train=size(data_train,1);
num_test=size(data_test,1);

w1=unifrnd(a,b,[n2 n1]);
o1=zeros(n2,1);
net1=zeros(n2,1);

w2=unifrnd(a,b,[n3 n2]);
o2=zeros(n3,1);
net2=zeros(n3,1);

w3=unifrnd(a,b,[n4 n3]);
o3=zeros(n4,1);
net3=zeros(n4,1);

error_train=zeros(num_train,1);
error_test=zeros(num_test,1);

output_train=zeros(num_train,1);
output_test=zeros(num_test,1);


mse_train=zeros(max_epoch,1);
mse_test=zeros(max_epoch,1);

for i=1:max_epoch

    data_train=data_train(randperm(num_train),:);
    
    for j=1:num_train
        input=data_train(j,1:n1);
        target=data_train(j,1+n1);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=net3;
        
        error_train(j)=target-o3;
        A=diag(o2.*(1-o2));
        B=diag(o1.*(1-o1));
        
        w1=w1-eta*error_train(j)*-1*1*(w3*A*w2*B)'*input;
        w2=w2-eta*error_train(j)*-1*1*(w3*A)'*o1';
        w3=w3-eta*error_train(j)*-1*1*o2';
    end
    
    for j=1:num_train
        input=data_train(j,1:n1);
        target=data_train(j,1+n1);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=round(net3);
        output_train(j)=o3;
        error_train(j)=target-o3;
    end
    
    mse_train(i)=mse(error_train);
    
    for j=1:num_test
        input=data_test(j,1:n1);
        target=data_test(j,1+n1);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=round(net3);
        output_test(j)=o3;
        error_test(j)=target-o3;
    end
    
    mse_test(i)=mse(error_test);
    
    
    figure(1);
    subplot(2,2,1),plot(data_train(:,n1+1),'-sr');
    hold on;
    subplot(2,2,1),plot(output_train,'-*b');
    hold off;
    
    subplot(2,2,3),plot(data_test(:,n1+1),'-sr');
    hold on;
    subplot(2,2,3),plot(output_test,'-*b');
    hold off;
    
    subplot(2,2,2),plot(mse_train(1:i),'-r');
    subplot(2,2,4),plot(mse_test(1:i),'-r');
    
    %     subplot(2,2,2)
    %     subplot(2,2,3)
    %     subplot(2,2,4)
    
        pause(0.5);
end

mse_test=max(mse_test)
mse_train=max(mse_train)

otuput_test_nn=output_test;
otuput_test_target=data_test(:,n1+1);


number_test1=num_test;
d_t1=zeros(number_test1,2);
d_t2=zeros(number_test1,2);

for i=1:number_test1
    if(otuput_test_target(i)==0)
        d_t1(i,:)=[1 0];
    end
    if(otuput_test_target(i)==1)
        d_t1(i,:)=[0 1];
    end
end

for i=1:number_test1
    if(otuput_test_nn(i)==0)
        d_t2(i,:)=[1 0];
    end
    if(otuput_test_nn(i)==1)
        d_t2(i,:)=[0 1];
    end
end

figure(5);
plotconfusion(d_t1',d_t2');

otuput_train_nn=output_train;
otuput_train_target=data_train(:,n1+1);


number_train=num_train;
d_t1=zeros(number_train,2);
d_t2=zeros(number_train,2);

for i=1:number_train
    if(otuput_train_target(i)==0)
        d_t1(i,:)=[1 0];
    end
    if(otuput_train_target(i)==1)
        d_t1(i,:)=[0 1];
    end
end

for i=1:number_train
    if(otuput_train_nn(i)==0)
        d_t2(i,:)=[1 0];
    end
    if(otuput_train_nn(i)==1)
        d_t2(i,:)=[0 1];
    end
end

figure(6);
plotconfusion(d_t1',d_t2');

figure(2);
plotregression(otuput_train_target,otuput_train_nn,'Regression Test');

figure(3);
plotregression(otuput_test_target,otuput_test_nn,'Regression Train');
