function d = discriminant(X,Sigma, mu,PiJ)
d = X*pinv(Sigma)*(mu') - 0.5*mu*pinv(Sigma)*mu' + log(PiJ);
end