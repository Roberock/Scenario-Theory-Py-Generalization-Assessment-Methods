function beta=getConfidence_ConvexDiscard(N,epsilon,k,d)
 %Give reliability of predicted interval
%As a graph
%epsilon=0.001; % prospective (scenario) unreliability
%k=0; % dicarded samples
%d=2*NTerms; % optimization variables
beta=0;  % confidence parameter
for j=0:(k+d-1)
    beta=beta+nchoosek(N,j).*epsilon.^j.*(1-epsilon).^(N-j);
end
beta=beta*nchoosek(k+d-1,k);
  
end