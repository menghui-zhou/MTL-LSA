function [sig_max] = max_singular_MTL(X)

sig_max = 0;  % perserve the max sigular value
for in = 1 : length(X)
    [~, s ,~] = svd(X{in});
    sig_max = max(sig_max, s(1,1));
end

end

