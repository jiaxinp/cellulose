function [res, varargout]=VelGrad(t)
global PAR varyingflow
if varyingflow
    slask=interp1(PAR.StreamLineData(:,1),PAR.StreamLineData,t);
else
    slask=interp1(PAR.StreamLineData(:,1),PAR.StreamLineData,0);
end
res=[slask(5:7)', slask(8:10)', slask(11:13)'];
%res=[0 10 0; 0 0 0; 0 0 0];
varargout{1}='CFD';
end