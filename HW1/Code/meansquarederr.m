function MSE = meansquarederr( T, Tdash )
t = size(T);
MSE = (sum((T - Tdash).^2))/(t(1)*t(2));
end