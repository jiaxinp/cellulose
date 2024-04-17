function [res]=PsiAlongStreamline(tend)
global PAR tshow varyingflow testmin testmax

done = 0;
while done == 0

    %PAR.r, PAR.Dr, PAR.Np, PAR.Nt has to be set before the call to this
    %function

    %With r=6 and Dr=0.05, Np=30 and Nt=20 gives a small error in the norm of Psi;
    %increasing resolution further does not give a lot of improvement, reducing
    %resolution gives drastic increase of the error.

    % PAR.Np=60; % resolution in phi
    % PAR.Nt=30; % resolution in theta
    dPHI = pi/PAR.Np;
    dTHETA = pi/PAR.Nt;

    % Create phi and theta vectors
    phi = dPHI/2:dPHI:pi;
    theta = dTHETA/2:dTHETA:pi;

    % Build differentiation matrices Dphi and Dtheta (using
    % Psi(T,P+delta)=Psi(pi-T,delta);

    % % Second order
    % Dphi = diag(ones((PAR.Np-1)*PAR.Nt,1),PAR.Nt)-diag(ones((PAR.Np-1)*PAR.Nt,1),-PAR.Nt);
    % slask = diag(ones(PAR.Nt,1));
    % slask = slask(:,end:-1:1);
    %
    % Dphi(1:PAR.Nt, ((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt)) = -slask;
    % Dphi(((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt), 1:PAR.Nt) = slask;
    % % end of second order
    %
    % Fourth order
    Dphi = 8*diag(ones((PAR.Np-1)*PAR.Nt,1),PAR.Nt) - 8*diag(ones((PAR.Np-1)*PAR.Nt,1),-PAR.Nt);
    Dphi = Dphi - diag(ones((PAR.Np-2)*PAR.Nt,1),2*PAR.Nt) + diag(ones((PAR.Np-2)*PAR.Nt,1),-2*PAR.Nt);
    slask = diag(ones(PAR.Nt,1));
    slask = slask(:,end:-1:1);
    Dphi(1:PAR.Nt, ((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt)) = -8*slask;
    Dphi(((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt), 1:PAR.Nt) = 8*slask;
    Dphi(1:PAR.Nt, ((PAR.Np-2)*PAR.Nt + 1):((PAR.Np-1)*PAR.Nt)) = Dphi(1:PAR.Nt, ((PAR.Np-2)*PAR.Nt + 1):((PAR.Np-1)*PAR.Nt)) + slask;
    Dphi(((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt), (PAR.Nt+1):(2*PAR.Nt)) = Dphi(((PAR.Np-1)*PAR.Nt + 1):((PAR.Np)*PAR.Nt), (PAR.Nt+1):(2*PAR.Nt)) - slask;
    Dphi((PAR.Nt+1):2*PAR.Nt, ((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt)) = Dphi((PAR.Nt+1):2*PAR.Nt, ((PAR.Np-1)*PAR.Nt + 1):(PAR.Np*PAR.Nt)) + slask;
    Dphi(((PAR.Np-2)*PAR.Nt + 1):((PAR.Np-1)*PAR.Nt), 1:PAR.Nt) = Dphi(((PAR.Np-2)*PAR.Nt + 1):((PAR.Np-1)*PAR.Nt), 1:PAR.Nt) - slask;

    Dphi=Dphi/6;
    % % end of fourth order

    Dphi = Dphi/(2*dPHI);
    PAR.Dphi = sparse(Dphi);

    % % Second order
    % slask = diag(ones(PAR.Nt-1,1),1)-diag(ones(PAR.Nt-1,1),-1);
    % slask(1,PAR.Nt) = -1;
    % slask(PAR.Nt,1) = 1;
    % %end of second order

    % Fourth order
    slask = 8*diag(ones(PAR.Nt-1,1),1)-8*diag(ones(PAR.Nt-1,1),-1);
    slask = slask - diag(ones(PAR.Nt-2,1),2)+diag(ones(PAR.Nt-2,1),-2);
    slask(1,PAR.Nt) = -8;
    slask(PAR.Nt,1) = 8;
    slask(1,PAR.Nt-1) = 1;
    slask(PAR.Nt,2) = -1;
    slask(2,PAR.Nt) = 1;
    slask(PAR.Nt-1,1) = -1;

    slask = slask/6;
    % end of fourth order

    Dtheta=[];
    for k=1:PAR.Np
        Dtheta=blkdiag(sparse(Dtheta),sparse(slask));
    end

    Dtheta = Dtheta/(2*dTHETA);
    PAR.Dtheta = sparse(Dtheta);

    % Create PP and TT: column vectors where each row is one position on the
    % unit half-sphere
    [PPmat, TTmat] = meshgrid(phi, theta);
    PAR.PP = reshape(PPmat,PAR.Np*PAR.Nt,1);
    PAR.TT = reshape(TTmat,PAR.Np*PAR.Nt,1);


    PAR.sinPHI=sin(PAR.PP);
    PAR.cosPHI=cos(PAR.PP);
    PAR.sinTHETA=sin(PAR.TT);
    PAR.cosTHETA=cos(PAR.TT);
    PAR.sinTHETAinvers=1./PAR.sinTHETA;


    % Create p and the unit vectors e_theta and e_phi where each row is the outwards pointing normal to the unit
    % sphere a one position
    p = zeros(PAR.Np*PAR.Nt,3);
    p(:,1) = PAR.sinTHETA.*PAR.cosPHI;
    p(:,2) = PAR.cosTHETA;
    p(:,3) = PAR.sinTHETA.*PAR.sinPHI;
    PAR.p=p;

    ePHI = zeros(PAR.Np*PAR.Nt,3);
    ePHI(:,1) = -PAR.sinPHI;
    ePHI(:,3) = PAR.cosPHI;
    PAR.ePHI = ePHI;

    eTHETA = zeros(PAR.Np*PAR.Nt,3);
    eTHETA(:,1) = PAR.cosTHETA.*PAR.cosPHI;
    eTHETA(:,2) = -PAR.sinTHETA;
    eTHETA(:,3) = PAR.cosTHETA.*PAR.sinPHI;
    PAR.eTHETA = eTHETA;

    PAR.Gamma = (PAR.r^2-1)/(PAR.r^2+1);
    PAR.dS = dPHI .* dTHETA .* abs(PAR.sinTHETA);


    %% Build matrix with |u x u'| dS (one U per column)
    Drhatbase=[];
    for k = 1:PAR.Np*PAR.Nt
        slask = sqrt( (p(k,2)*p(:,3) - p(k,3)*p(:,2)).^2 + (p(k,3)*p(:,1) - p(k,1)*p(:,3)).^2 + (p(k,1)*p(:,2) - p(k,2)*p(:,1)).^2);
        slask = slask .* PAR.dS;
        Drhatbase = [Drhatbase slask];
    end
    PAR.Drhatbase = Drhatbase;


    tshow = 0;
    dt=0.001;

    if 1 %run until equilibrium
        tvec=0:dt:tend;
        tshow = 0;
        varyingflow = 0;
        testmin = 42;
        testmax = 0;
        options=odeset('NonNegative',1:(PAR.Np*PAR.Nt),'RelTol',1e-3,'AbsTol',1e-3);
        slask = ode15s(@Psidot,tvec,1/(2*pi)*ones(PAR.Np*PAR.Nt,1),options);

        tvec=0:dt:tend;
        tshow = 0;
        varyingflow = 1;
        testmin = 42;
        testmax = 0;
        options=odeset('NonNegative',1:(PAR.Np*PAR.Nt),'RelTol',1e-3,'AbsTol',1e-3);
        res = ode15s(@Psidot,tvec,deval(slask,(tend-dt)),options);

    else % start with isotropic distribution
        tvec=0:dt:tend;
        tshow = 0;
        varyingflow = 1;
        testmin = 42;
        testmax = 0;
        options=odeset('NonNegative',1:(PAR.Np*PAR.Nt),'RelTol',1e-3,'AbsTol',1e-3);
        res = ode15s(@Psidot,tvec,1/(2*pi)*ones(PAR.Np*PAR.Nt,1),options);
    end


    norm = sum(PAR.dS.*deval(res,0));
    normmin = norm;
    normmax = norm;

    for t=0:tend/200:(tend-tend/200)
        norm=sum(PAR.dS.*deval(res,t));
        if norm < normmin
            normmin=norm;
        end
        if norm > normmax
            normmax = norm;
        end
    end
    %[normmax normmin]
    if (normmax-normmin)/(0.5*(normmax+normmin)) > 0.02
        %warning(sprintf('Variation of norm is %.2f percent. Consider increasing resolution.', (normmax-normmin)/normmin*100))
        PAR.Np = round(PAR.Np*1.5);
        PAR.Nt = round(PAR.Nt*1.5);
        display(sprintf('Increasing resolution to (%d, %d); Error: %.2f', PAR.Np, PAR.Nt, (normmax-normmin)/(0.5*(normmax+normmin))*100))
    else
        done = 1;
    end

end
end



function [res] = Psidot(t,Psi)
global PAR tshow timespent k_mobility Drconstant FTswitch CiFT testmin testmax
intermediateplot = 0;

gradu = VelGrad(t);


norm = sum(PAR.dS.*Psi);
if norm > testmax
    testmax = norm;
elseif norm < testmin
    testmin = norm;
end


if isempty(k_mobility)
    k_mobility = 1
end


if (testmax-testmin)/(0.5*(testmax+testmin)) > 0.02
    res = zeros(size(Psi));
else
    if t-tshow>0.01 & intermediateplot == 1
        tshow=t;
        figure(2)
        subplot(2,1,1)
        cla
        index = find(PAR.TT==0+pi/PAR.Nt/2);
        plot(PAR.PP(index),Psi(index),'b')
        hold on
        title(sprintf('t=%f, Np=%d',t,PAR.Np))
        index = find(PAR.PP==pi/PAR.Np/2);
        plot(PAR.TT(index),Psi(index),'r')
        hold off

        subplot(2,1,2)
        dPHI = pi/PAR.Np;
        dTHETA = pi/PAR.Nt;
        dS = dPHI .* dTHETA .* abs(PAR.sinTHETA);
        plot(t,sum(dS.*Psi),'*');
        hold on
        drawnow
    elseif isempty(tshow)
        tshow=0;
    end

    gradu = VelGrad(t);

    %Calculate vorticity and rate of strain on good format for speedup
    WTET = 0.5*[(gradu' - gradu) (gradu' + gradu)];
    %E = 0.5*(gradu + gradu');

    % Calculate the terms in pdot
    % Wp = (W * p')';
    % Ep = (E * p')';

    %This is faster!

    WpEp = PAR.p * WTET;
    %Wp=slask(:,1:3);
    Ep= WpEp(:,4:6);

    % Wp = PAR.p * W';
    % Ep = PAR.p * E';

    % for k = 1:Np*Nt
    %     pEpp(k,:)= ((p(k,:) * Ep(k,:)') * p(k,:)')' ;
    % end

    %This is faster!
    pEp = sum(PAR.p .* Ep,2); % scalar products between each row

    %pEpp=[pEp pEp pEp].*PAR.p;

    % Calculate pdot
    %pdot = WpEp(:,1:3) + PAR.Gamma * (WpEp(:,4:6)-pEpp);
    pdot = (WpEp(:,1:3) + PAR.Gamma * (Ep - [pEp, pEp, pEp] .* PAR.p)) * k_mobility;

    % Project pdot onto ePHI and eTHETA (utilising the fact that we only need
    % sinTHETA*phidot and not PHIdot alone)
    phidotsinTHETA = sum(pdot .* PAR.ePHI,2); %projection of each row through row-wise
    thetadot = sum(pdot .* PAR.eTHETA,2);


    if isempty(Drconstant)
        Drconstant = 1;
    end
    if isempty(FTswitch)
        FTswitch = 0;
    end

    if FTswitch == 0
        if Drconstant == 1
            DrPsi = PAR.Dr*Psi;

            % %original
            % res = PAR.Dphi * (PAR.sinTHETAinvers.*(PAR.Dphi*DrPsi) - phidotsinTHETA.*Psi);
            % res = res + PAR.Dtheta*(PAR.sinTHETA.*(PAR.Dtheta*DrPsi) - thetadot.*PAR.sinTHETA.*Psi);
            % res = res .* PAR.sinTHETAinvers;

            %single line: slightly faster
            res = (PAR.Dphi * (PAR.sinTHETAinvers.*(PAR.Dphi*DrPsi) - phidotsinTHETA.*Psi) + PAR.Dtheta*(PAR.sinTHETA.*(PAR.Dtheta*DrPsi) - thetadot.*PAR.sinTHETA.*Psi)).* PAR.sinTHETAinvers;
        else
            % calculate Drhat
            % calculate dpsi....
            % is it THAT easy?
            Drhat = PAR.Dr*(Psi' * PAR.Drhatbase)';
            res = (PAR.Dphi * (Drhat.*PAR.sinTHETAinvers.*(PAR.Dphi*Psi) - phidotsinTHETA.*Psi) + PAR.Dtheta*(Drhat.*PAR.sinTHETA.*(PAR.Dtheta*Psi) - thetadot.*PAR.sinTHETA.*Psi)).* PAR.sinTHETAinvers;
            PlotDrhat = 0;
            if min(Drhat)<0.5*max(Drhat) & PlotDrhat == 1
                figure(1)
                subplot(2,1,1)
                PlotPsiOnSphere(Drhat);
                title('Dr')
                axis([-1.1 1.1 -1.1 1.1 -1.1 1.1])
                colorbar
                subplot(2,1,2)
                PlotPsiOnSphere(Psi);
                title('\Psi')
                colorbar
                axis([-1.1 1.1 -1.1 1.1 -1.1 1.1])
                drawnow
            end
        end
    else
        FTconst = sqrt(0.5*sum(sum(WTET(:,4:6).^2)));
        DrFT = CiFT * FTconst;
        DrPsi = (PAR.Dr + DrFT)*Psi;
        res = (PAR.Dphi * (PAR.sinTHETAinvers.*(PAR.Dphi*DrPsi) - phidotsinTHETA.*Psi) + PAR.Dtheta*(PAR.sinTHETA.*(PAR.Dtheta*DrPsi) - thetadot.*PAR.sinTHETA.*Psi)).* PAR.sinTHETAinvers;
    end

    IndexToZero = find(Psi < 1e-5*max(Psi) & res < 0);
    res(IndexToZero) = 0;

    [slask index1] = max(Psi);
    [slask index2] = max(slask);
    if res(index1,index2) < 0
        decrease = 1;
    else
        decrease = 0;
    end

    if OrderParam(Psi,[0 1 0]) < -0.45 & decrease | OrderParam(Psi,[0 1 0]) > 0.9 & not(decrease)
        res = zeros(size(Psi));
    end

end
end
