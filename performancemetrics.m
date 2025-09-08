close all;
clear all;
clc;

N = 40
st = 1;
Fsc=[];
MIU=[];
PA=[];
bestfsc=0;
bestmiu=0;
bestpa=0;
bestep = 0;

for i = st:st+N-1
    
   %gname = strcat('./Brain_test/',num2str(i,'%04d'),'.png');
   
   tname = '"C:\Users\27612\Desktop\KiU-Net-pytorch-master (2)\KiU-Net-pytorch-master\u_net\test_result"';%path of predicting results
   imgname = strcat(tname,'\',num2str(i,'%02d'),'.png');
   lname = 'C:\Users\27612\Desktop\KiU-Net-pytorch-master\KiU-Net-pytorch-master\Glas\testB\label';
   labelname = strcat(lname,'\', num2str(i,'%02d'),'.png');
   
   %predict
   I = double(imread(imgname));
   tmp=zeros(128,128);
   tmp(I>130) = 255;tmp(I<131) = 0;
   
   %label
   tmp2 = double(imread(labelname));%128x128x3
   tmp2=tmp2(:,:,1);%三个通道相同，保留一个
   tmp2(tmp2<=0)=0;tmp2(tmp2>0)=255;

   tp=0;fp=0;fn=0;tn=0;uni=0;ttp=0;lab=0;
   
   for p =1:128
       for q =1:128
           if tmp(p,q)==0 
               if tmp2(p,q) == tmp(p,q)
                   tn = tn+1;
               else
                   fp = fp+1;
                   uni = uni+1;
                   ttp = ttp+1;
               end
           elseif tmp(p,q)==255
               lab = lab +1;
               if tmp2(p,q) == tmp(p,q)
                   tp = tp+1;
                   ttp = ttp+1;
               else
                   fn = fn+1;
               end
               uni = uni+1;
           end
           
       end
   end
   
   if (tp~=0)
       F = (2*tp)/(2*tp+fp+fn);
       Fsc=[Fsc;F];
       MIU=[MIU,(tp*1.0/uni)];
       PA=[PA,(tp*1.0/ttp)];
       
   % elseif (lab==0)
    %   MIU=[MIU,1];
    %   PA=[PA,1];
    %   Fsc=[Fsc;[i,1]];
   else
       MIU=[MIU,1];
       PA=[PA,1];
       Fsc=[Fsc;1];
   
   end
   

   

    if bestfsc <= mean(Fsc) & (mean(Fsc) ~= 1)
    bestfsc = mean(Fsc) %F1分数,dice coefficient
    bestmiu = mean(MIU,2) %jacarrd index
    bestpa = mean(PA,2) %准确度
   
    end

   mean(Fsc)
   
   end
% bestfsc 
% bestmiu 
% bestpa 


% plot(Fsc(:,1),Fsc(:,2),'-*')
% hold on
% plot(Fsc(:,1),Fsc1(:,2),'-s')
% hold off
% figure();plot(Fsc(:,1),PA,'-*');hold on
% plot(Fsc(:,1),PA1,'-s');hold off
% Fsc1=Fsc;
% MIU1=MIU;
% PA1=PA;