function [W, funcVal] = Least_AGM_MTL_LSA(X,  Y,  rho1, rho2, rho3, alpha, opts)

if nargin < 7
    opts = [];
end



sig_max = max_singular_MTL(X) ^ 2;


H = LS_adjustment(alpha, length(X));
% H = MLS_adjustment(alpha, length(X));



for t = 1 : length(X)
    X{t} = X{t}';
end
%¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­


opts = init_opts(opts);
task_num  = length (X);


dimension = size(X{1}, 1); 


funcVal = [];



if opts.init==2
    W0 = zeros(dimension, task_num);  
elseif opts.init== 0
    W0 = randn(dimension, task_num);  
else
    if isfield(opts,'W0')    
        W0  = opts.W0;
       
        % n = nnz(X) returns the number of nonzero elements in matrix X.
        % nnz means Number of NonZero

        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else

        W0 = zeros(dimension, task_num);
    end
end


bFlag=0; % this flag tests whether the gradient step only changes a little


Wz = W0; % Wz is search point
Wz_old = W0; % Wz_old is last search point

t = 1; % parameter to do convex optimization to Nesterov's method
t_old = 0; % last parameter

iter = 0; % the count of iteration



%% line search version
% gamma = 1;  % gmama means 1 / stepsize 

%% constant stepsize version
gamma = sig_max;  % stepsize is 1 / gamma
    

gamma_inc = 2; % modify the gamma, exponential increase 


while iter < opts.maxIter
    
    % alpha =  (t(i-2)-1)/t(i-1);
    alpha = (t_old - 1) /t;
      
    % Si = Xi + alpha(Xi - Xi-1) 
    % Si = Ws s: search point
  
    
    
    %% nondescent version
     Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    
   
     
  

    gWs  = gradVal_eval(Ws);  
    Fs   = funVal_eval(Ws);  
 
    
    while true

        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma, rho2 / gamma, rho3 / gamma, alpha);
        % Wzp means the projection of Sz
        
        Fzp = funVal_eval(Wzp);
    

        delta_Wzp = Wzp - Ws;   

        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        r_sum = nrm_delta_Wzp;

        Fzp_gamma = Fs   ...                
                  + sum(sum(delta_Wzp .* gWs))...
                  + gamma/2 * nrm_delta_Wzp;
        
        % this flag tests whether the gradient step only changes a little
        if (r_sum <= 1e-20)   
            bFlag = 1; % this shows that, the gradient step makes little improvement
            break;
        end
        

        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;  %% gamma ÓÐÉÏÏÞ
        end
    end
    

    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1, rho2, rho3, H));
    
    if (bFlag)

        break;
    end
    
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter >= 2
                % the function value almost does not change
                if (abs( funcVal(end) - funcVal(end-1)) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter >= 2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol * funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end) <= opts.tol)
                break;
            end
        case 3
            if iter >= opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [Wp] = FGLasso_projection (W,  lambda1, lambda2, lambda3, alpha)
        

        temp_opts = [];
        temp_opts = init_opts(temp_opts);
        temp_opts.maxIter = 3000; 
        temp_opts.rho = 0.001;  

        temp_opts.tFlag = 0;  
        temp_opts.tol = 1e-9;  
        task_num_temp = length(X);
        X_temp = cell(task_num_temp, 1);
        W_temp = cell(task_num_temp, 1);
        for i = 1 : task_num_temp
            X_temp{i} = eye(size(W, 1));
            W_temp{i} = W(:, i);
        end
        
        [Wp, temp_out] = Least_ADMM_MTL_LSA(X_temp, W_temp, lambda1, lambda2, lambda3, alpha, temp_opts);
        if mod(iter, 1000) ==0 
            disp([num2str(iter), '-th  Middle Iteration:   ', num2str(temp_out.itr)]);
        end



    end

% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor i = 1 : task_num
                grad_W(:, i) = X{i}*( ( (X{i}' * W(:,i) -Y{i} ) ) );
            end
        else
            grad_W = [];
            for i = 1 : task_num

                grad_W = cat(2, grad_W, X{i}*( ( (X{i}' * W(:,i) -Y{i} ) ) ));
            end
        end
    end

% smooth part gradient.
    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm ((Y{i} - X{i}' * W(:, i)), 2)^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm ((Y{i} - X{i}' * W(:, i)), 2)^2;
            end
        end
    end

    function [non_smooth_value] = nonsmooth_eval(W, lambda1, lambda2, lambda3, H)
%         non_smooth_value = 0;
  
        % \lambda_1||W||_1 + \lambda_2||W||_{2,1}, row decouple
        part1 = 0;
        for i = 1 : size(W, 1)
            row_temp = W(i, :); 
            row_vector = row_temp'; 
            part1 = part1 + lambda1 * norm(row_vector, 1) ...
                + lambda2 * norm(row_vector, 2);
        end
        
        % + \lambda_3||WH||_1, column decouple
        part2 = 0;
        fused_W = W * H;
        for i = 1 : size(fused_W, 2)
            col_temp = fused_W(:, i);
            part2 = part2 + lambda3 * norm(col_temp, 1);
        end
        
        non_smooth_value = part1 + part2;
        
        
        F = H';
        non_smooth = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth = non_smooth ...
                + lambda1 * norm(w, 1) + lambda3 * norm(F * w', 1) ...
                + lambda2 * norm(w', 2);
        end
    end

end