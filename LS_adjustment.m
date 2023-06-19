function [F] = LS_adjustment(convex, task_num)




F = zeros(task_num, task_num-1);
% for example: one case if task_num = 5
% F = [1  0  0  0;
%     -1  1  0  0;
%     0  -1  1  0;
%     0  0  -1  1;
%     0  0  0  -1];
F( 1 : (task_num + 1) : end) = 1;
F( 2 : (task_num + 1) : end) = -1;




% combination = eye(4,4); % convex combination 
combination = eye(task_num-1);
for i = 1 : task_num - 2
    com = combination;
    com(i,i+1) = convex;
    com(i+1, i+1) = 1-convex;
    F = F * com;
end


end

