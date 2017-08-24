%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: 
%       Yong Li
%Email:
%       liyongforevercas@163.com
%Department:
%       National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences
%
%Description:
%    to solve the robust face representation learning problem with classwise block-Diagonal Structure
%    with the inexact ALM algorithm
%    min |Z|_*+alpha/2*sum(\| \bar{Z}_{-i} \|_F^2) +lambda*|E|_1+\beta*||Z||_1+gamma/2*||D||_F^2
%     s.t., X = DZ+E
% inputs:
%        X -- d*N data matrix, d is the data dimension, and N is the number
%        of total samples.
%        D -- dictionary initialized with the training samples
%        param-- a struct contains weights of different items 
%Reference:
    %Y. Li, J. Liu, H. Lu, and S. Ma, ¡°Learning Robust Face Representation With Classwise Block-Diagonal Structure¡± 
    %Information Forensics and Security, IEEE Transactions on, 2014, pp.2051¨C2062.
    
% RCBD Copyright 2014, Yong Li (liyongforevercas@163.com)
% RCBD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% You should have received a copy of the GNU General Public License
% along with RCBD.  If not, see <http://www.gnu.org/licenses/>.
function varargout = RCBD(X, D, param)
gamma = param.gamma;
beta = param.beta;
lambda = param.lambda;
alpha =param.alpha; 
trainSamPerCls = ones(1, param.clsNum)*(param.trainNumPerCls);
m = size(D,2); % the number of bases of dictionary to be learnt
n = size( X, 2 ); % the number of total samples
trainSamTal = param.clsNum*param.trainNumPerCls;
tol = 1e-6;
maxIter = 5000; 
rho = 1.1;
mu= 1e-5;
max_mu= 1e8;

%% Initializing optimization variables
Z=zeros( size(D,2), size(X,2) );
E= zeros( size(X) );
Q=sparse( zeros(m,n) );
J = zeros(m,n); L =J;
Y1 = zeros( size(X) ); Y2 = zeros(m,n); Y3= Y2;
%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;
    %update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    %udpate Z 
    Z_left = (D)'*(D)+(alpha/mu+2 )*eye(m);
    Z = Z_left \ ( (D)'*(X-E)+J +L +( (D)'*Y1-Y2 -Y3 +alpha*Q )/mu );
    
    %update L
    L_temp = Z+Y3/mu;
    L= solve_l1_norm( L_temp, beta/mu );
    
    %update E
    xmaz = X-D*Z;
    temp = xmaz+Y1/mu;
    E = solve_l1_norm(temp,lambda/mu);

    %update D
    D_trans= ( Z*Z'+gamma/mu*eye(m) ) \ ( Y1*Z'/mu - (E-X)*Z' )';
    D = D_trans';
    
    Z_block = cell(numel(trainSamPerCls),1);
    for k = 1:numel(trainSamPerCls)
        Z_block{k} = Z ( (k-1)*param.dicNumPerCls+1: k*param.dicNumPerCls,...
            sum(trainSamPerCls(1:k-1))+1:sum( trainSamPerCls(1:k) ) );
    end
    
    Z_right =Z( : , trainSamTal+1 : n );    
    Q = [blkdiag( Z_block{:} ), Z_right];
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    leq3=Z-L;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max( stopC, max(max(abs(leq3))) );
    if (iter==1 || mod(iter, 20 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
    %update lagrange multipliers
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;
    mu = min(max_mu,mu*rho);
    end
end
varargout{1} = Z;
varargout{2} = E;
varargout{3} = D;
end

function [E] = solve_l1_norm(x,varepsilon)
     E = max(x- varepsilon, 0);
     E = E+min( x+ varepsilon, 0);   
end
