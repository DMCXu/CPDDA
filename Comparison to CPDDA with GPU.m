clear all
clc
Data=xlsread('D:\A_optic\code\MPDDA-1.0-master\absd.xlsx');
Wavelength=Data(:,1);
Re_n=Data(:,2);                 % Real part of the refractive index
Im_n=Data(:,3);                 % Imaginary part of the refractive index

eps=(Re_n+1i*Im_n).^2;          % Dielectric function of the bulk metal

CT=10^(-5);                     % Convergence threshold value in interative solver

d = 0.25:0.01:2;
GPU = 0;
nb = 1.33;
r_eff = 20;
Np_shape = "rec_block";
r_np=r_eff*(10^(-9));               % effective radius of the nanoparticle
W=2*pi*3*10^17./Wavelength;         % incident radiation frequency
k=2*pi./Wavelength*nb;              % wave vector of light in first layer
Np_name = "Au";
clc
if Np_name=="Au"
    Wp=8.9*1.5186*(10^15);              % plasma frequency of Au
    L0=0.07/6.58211951440*10^(-16);     % Collision freguency of Au in bulk medium
    Ap=0.5;                             % damping correction factor
    Vf=1.4*(10^6);                      % Fermi Velocity
elseif Np_name=="Ag"
    L0=3.22*(10^13);  % Collision freguency of Ag in bulk medium or bulk damping constant, Jonhson Paper 1972
    Vf=1.39*(10^6);    % Fermi velocity of Ag atoms in bulk medium
    Ap=0.25;
    Wp=1.393*(10^16); %Jonhson Paper 197
elseif Np_name=="Cu"
    L0=1.45*(10^14);  % Collision freguency of Cu in bulk medium or bulk damping constant, Jonhson Paper 1972
    Ap=0.5;
    Wp=1.344*(10^16); %Jonhson Paper 1972
    Vf=1.59*(10^6);    % Fermi velocity of Cu atoms in bulk medium
end
L=L0+Ap*Vf/r_np;                         % Modified damping frequency
eps_nps=eps+(Wp^2)./(W.^2+1i*L0.*W)-(Wp.^2)./(W.^2+1i*W.*L); % Modified dielectric function
[Lx,Ly,Lz]= Nps_parameters(r_eff,Np_shape);
%=========================================================================%
d_eff=2*r_eff;
volume=4*pi/3*(r_eff^3);
epsb=nb^2; 
E01 = [1 0 0];
K01 = [0 0 1];
Structure = "monomeric";
arrangement="monomeric";
if GPU==1 
    E0=gpuArray(E01);
    K0=gpuArray(K01);   
elseif GPU==0
    E0=E01;               % Incident electric field
    K0=K01;               % unit vector in direction of wave vector
end
Refractive=eps_nps.^(0.5);
Re_Refractive=real(Refractive);
Im_Refractive=imag(Refractive);

if GPU==1
    ep_nps_eb=gpuArray(eps_nps./epsb);  % Ratio of metal-to-medium dielectric function
elseif GPU==0
    ep_nps_eb=eps_nps./epsb;  % Ratio of metal-to-medium dielectric function
end

Meshing = 2;
for J =1:length(d)
    d_j =d(J);
    [Max_x,Max_y,Max_z,N,Nx,Ny,Nz,r_block,X,Y,Z,d_inter]=Coordinates1(Meshing,GPU,d_j,Lx,Ly,Lz,...
    d_eff,Structure,arrangement); 


    IB = "plane wave";
    if IB=="plane wave" 
        z0=0;              % Focus point of the Gaussian beam
        Waist_r=100;       % ratio of waist raduis of beam to wavelength
    elseif IB=="gaussian"
        fprintf('\nA Gaussian beam has been choosen.');
        z0=input('\nEnter focus point of beam, \nEx. could be -Max_z/2 (Max_z is the length of structure in z-direction, center is at 0):');
        clc
        fprintf('\nChoosing the waist raduis of beam:');
        fprintf('\nIf the waist raduis is much bigger than the NPs size, the results will be like plane wave results');
        Waist_r=input('\nEnter the RATIO of waist raduis of beam to wavelength, \nEx. 0.1:');
        clc
    end
    [INDEX_INSIDE]=INDEX_INSIDE_NP(GPU,X,Y,Z,N,Np_shape,Lx,Ly,Lz,Structure,d_inter,arrangement);

    INDEX_IN=reshape(INDEX_INSIDE,[Nx,Ny,Nz]); 

    [rjkrjk1_I,rjkrjk2_I,rjkrjk3_I,rjkrjk4_I,rjkrjk5_I,rjkrjk6_I,rjkrjk31_I,rjkrjk32_I,...
    rjkrjk33_I,rjkrjk34_I,rjkrjk35_I,rjkrjk36_I,RJK]=RijRij(r_block);
    tic;
    eps_NP_eb=ep_nps_eb;
    Lambda=Wavelength;
    
    if GPU==1
        kvec=gpuArray(k*K0);
    elseif GPU==0
        kvec=k*K0;
    end
    [E_x,E_y,E_z,E_vector]=Incident_Field(Lambda,IB,nb,r_block,kvec,K0,INDEX_INSIDE,Nx,Ny,Nz,E0,z0,Waist_r);
    %=====================================================================%
    
    
    %======Obtaining inverse of P of each dipole at different Lambda======%
    %=====================================================================%
    [Inverse_Alpha]=Polarizability(GPU,kvec,eps_NP_eb,INDEX_IN,d_j,E0);
    Exp_ikvec_rjk=exp(1i*norm(kvec)*RJK)./RJK;
    ikvec_rjk=(1i*norm(kvec)*RJK-1)./(RJK.^2); 
    
    [Axx,Axy,Axz,Ayy,Ayz,Azz]=Interaction_Matrix(kvec,Exp_ikvec_rjk,...
        ikvec_rjk, rjkrjk1_I,rjkrjk2_I,rjkrjk3_I,rjkrjk4_I,rjkrjk5_I,rjkrjk6_I,...
        rjkrjk31_I,rjkrjk32_I,rjkrjk33_I,rjkrjk34_I,rjkrjk35_I,rjkrjk36_I,Nx,Ny,Nz);
    [FFT_AXX,FFT_AXY,FFT_AXZ,FFT_AYY,FFT_AYZ,FFT_AZZ]=FFT_Interaction(GPU,Axx...
            ,Axy,Axz,Ayy,Ayz,Azz,Nx,Ny,Nz);
    %=====================================================================%
    
    
             % Iterative Method, Biconjugate gradient & inverse FFT%
    %=== Applying Biconjugate gradient & inverse FFT to obtainPx,Py,Pz ===% 
    %=====================================================================%
    [px,py,pz]=Biconjugate_Gradient(E_x,E_y,E_z,Nx,Ny,Nz,N,Inverse_Alpha,...
             INDEX_IN,E_vector,FFT_AXX,FFT_AXY,FFT_AXZ,FFT_AYY,FFT_AYZ,FFT_AZZ,CT);
    %=====================================================================%
    
    
                    % Deleting unnessesary data %
    %=====================================================================%                
    clear FFT_AXX FFT_AXY FFT_AXZ FFT_AYY FFT_AYZ FFT_AZZ 
    %=====================================================================%
    
    
             % Ignoring polarizibality of dipole outside NPs %
    %=====================================================================%        
    px=px.*INDEX_IN;
    py=py.*INDEX_IN;
    pz=pz.*INDEX_IN;
    %=====================================================================%   
    
    PX_vector=reshape(px,[N,1]);
    PY_vector=reshape(py,[N,1]);
    PZ_vector=reshape(pz,[N,1]);
    Inv_Alpha=reshape(Inverse_Alpha,[N,1]);
    Inv_Alpha_vec=[Inv_Alpha;Inv_Alpha;Inv_Alpha];
    
    
    %======== Calculating Cabs, Cscat & Cext for each wavelength =========%
    %=====================================================================%
    P_vector=[PX_vector;PY_vector;PZ_vector];
    Cabs(J)=4*pi*norm(kvec)/sum(abs(E0.^2))*((imag(dot((conj(P_vector)),conj(P_vector.*Inv_Alpha_vec)))-2/3*norm(kvec)^3*(norm(P_vector).^2)));
    Cext(J)=4*pi*norm(kvec)/sum(abs(E0.^2))*imag(dot((E_vector),P_vector));
    Cscat(J)=Cext(J)-Cabs(J);
    total_time = toc;
    clear PX_vector PY_vector PZ_vector P_vector px py pz a_CM_Nps anr_Nps ...
        aLDR_Nps  a_CM_Matrix anr_Matrix aLDR_Matrix Ex Ey Ez E_x E_y E_z ...
        E_vector Exp_ikvec_rjk ikvec_rjk Inverse_Alpha
            
    disp(total_time);
end


    
    
    
