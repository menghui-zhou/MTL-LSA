function [W, out] = Least_ADMM_MTL_LSA(X, Y, lambda1, lambda2, lambda3, alpha, opts)

if nargin < 7
    opts = [];
    opts = init_opts(opts);
end

H = LS_adjustment(alpha, length(X));


if ~isfield(opts, 'p_residual'); opts.p_residual = 1;  end  % primal residual tolerance
if ~isfield(opts, 'd_residual'); opts.d_residual = 1;  end  % dual residual tolerance
if ~isfield(opts, 'rho');       opts.rho = 0.01;       end  % the parameter of quadratic penalty
if ~isfield(opts, 'tao');       opts.tao = 1.618;      end  % adjust the step size,  0<tao<(1+sqrt(5)/2) 
if ~isfield(opts, 'verbose');   opts.verbose = 0;      end  % display middle result




iter = 1; 
tt = tic;
W = zeros(size(X{1}, 2), length(X));
out = struct();


fp = inf;   
primal_residual = inf; % primal_residual
dual_residual = inf; % dual_residual
f = func_value(X, Y, W, lambda1, lambda2, lambda3, H); 
f0 = f;        
out.fvec = f0;  



% Lagragian variables
A = zeros(size(W));
B = zeros(size(W,1), size(H,2));
% Dual variables
C = zeros(size(A));
D = zeros(size(B));

F = H * H';
G = H';



%% cholesky decompostion
Cho_coe = cell(length(X), 1); 
for i = 1 : length(X)
    need_inv = X{i}'*X{i} + opts.rho * (1 + F(i,i))  * eye(size(X{i}, 2)); 
    temp = chol(need_inv); % temp' * temp = need_inv
    Cho_coe{i} = temp;
end




       
while iter < opts.maxIter ...
       && ...
       ~( primal_residual < opts.p_residual && dual_residual < opts.d_residual) 
     
   

    W_new = zeros(size(W));
    % series version
    for i = 1 : length(X)  % task num = length(X)
        F_temp = 0;
        for j = 1 : size(W, 2)
            if j ~= i
                F_temp = F_temp + W(:, i) * F(j, i);
            end
        end
        equ_right = X{i}' * Y{i} - C(:, i) + opts.rho * A(:, i) ...
                  + (opts.rho * B - D) * G(:, i) - opts.rho * F_temp;
        
         W_new(:, i) = Cho_coe{i} \ (Cho_coe{i}' \ equ_right);
    end
    W = W_new;
    
    
    
    
    %% update ADMM slack variable
    % row decouple
    A_old = A;
    A_target = W + C / opts.rho;  
    A = prox_SGL(A_target, lambda1 / opts.rho, lambda2 / opts.rho);
    
    % column decouple
    B_old = B;
    B_target = W * H + D/ opts.rho;
    B = prox_FL(B_target, lambda3 / opts.rho);
    

    
    %% dual residual
    dr = opts.rho * ((A - A_old) + (B - B_old) * H');
    dual_residual = norm(dr, 'fro');
    
    
    
    %% update Lagrangian dual multiplier
    C = C + opts.tao * opts.rho * (W - A);
    D = D + opts.tao * opts.rho * (W * H - B);

    
    
    

    fp = f;  
    f = func_value(X, Y, W, lambda1, lambda2, lambda3, H);
    
    
    
    %% primal residual
    pr1 = norm(W - A, 'fro');
    pr2 = norm(W * H - B, 'fro');
    primal_residual = pr1 + pr2;
    
    
    if opts.verbose
        fprintf('iter: %d \t fval: e%e \t feasi:%.1e \n', iter, f, primal_residual);
    end
    

    iter = iter + 1;
    

    out.fvec = [out.fvec; f]; 
%     funcValue = out.fvec;

    

    switch(opts.tFlag)
        case 0
            if iter >=2 
                % the function value almost does not change
                if (abs(f - fp ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter >= 2
                if (abs( f- fp ) <= fp * opts.tol )
                    break;
                end
            end
        case 2
            if ( f <= opts.tol)
                break;
            end
        case 3
            if iter >= opts.maxIter
                break;
            end
    end
    
end



out.W_candidate = A;
out.C = C;
out.D = D;

out.fval = f; 
out.itr = iter; 
out.tt = toc(tt); 
out.primal_residual = primal_residual; % primal feasibility
out.dual_residual = dual_residual; % dual feasibility
out.f_delta = abs(fp - f); 

end

% 1/2||XW-Y||_F^2 
% + \lambda_1||W||_1 + \lambda_2||W||_{2,1}
% + \lambda_3||WH||_1
function val = func_value(X, Y, W, lambda1, lambda2, lambda3, H)
    % 1/2||XW-Y||_F^2
    part1 = 0; 
    for i =  1 : length(X)
        part1 = part1 + norm(X{i} * W(:, i) - Y{i}, 2)^2;
    end
    part1 = part1 * 0.5; 
    
    
% + \lambda_1||W||_1 + \lambda_2||W||_{2,1}, row decouple
    part2 = 0;
    for i = 1 : size(W, 1)
        row_temp = W(i, :); 
        row_vector = row_temp'; 
        part2 = part2 + lambda1 * norm(row_vector, 1) ...
                      + lambda2 * norm(row_vector, 2);
    end  
    
% + \lambda_3||WH||_1, column decouple
    part3 = 0;
    fused_W = W * H;
    for i = 1 : size(fused_W, 2)
        col_temp = fused_W(:, i);
        part3 = part3 + lambda3 * norm(col_temp, 1);
    end
    
    val = part1 + part2 + part3;
end





% 1/2||W-V||_F^2 + \lambda_1||W||_1 + \lambda_2||W||_{2,1}
function Out = prox_SGL(V ,lambda1, lambda2)
    Out = zeros(size(V));

    for row = 1 : size(V, 1)

        target = V(row, :); 
        t_Lasso = max(abs(target) - lambda1, 0) .* sign(target);
             
        

        out_temp = norm(t_Lasso);  
        if out_temp  == 0
            out = 0;
        else
            out = max(out_temp - lambda2, 0) / out_temp * t_Lasso;
        end
        Out(row, :) = out;
    end
end



% 1/2||W-V||_F^2 + \lambda_3||W||_1
function Out = prox_FL(V, lambda)
    Out = zeros(size(V));
  
    for col = 1 : size(V, 2)
        target = V(:, col); 
        out_temp = max(abs(target) - lambda, 0); 
        out = sign(target) .* out_temp; 
        Out(:, col) = out;
    end
end


