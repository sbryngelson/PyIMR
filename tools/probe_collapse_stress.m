% Dump IMRv2's collapse initial stress at full precision (PLAN.md W8).
%
% imr-fast's precursor lands on S0 = -0.159945110; the trajectory that matches
% ref_collapse_zener.csv to 1.5e-05 implies S0 = -0.160046461. This prints what
% upstream actually computes, plus the intermediates that feed it, so the
% 1.0e-04 gap can be attributed rather than guessed at.
%
% Usage: matlab -batch "run('tools/probe_collapse_stress.m')"

imrv2 = getenv('IMRV2_ROOT');
if isempty(imrv2)
    imrv2 = fullfile(getenv('HOME'), 'research/docs/_ideas-boed/upstream/IMRv2');
end
addpath(fullfile(imrv2, 'src'));

R0 = 225e-6; Req = R0/6; tv = linspace(0, 120e-6, 300);
rho8 = 1064; P8 = 101325; t0 = R0/sqrt(P8/rho8);
G = 2500; mu = 0.1; kappa = 1.4;

argv = {'progdisplay',0,'method',23,'dimout',0,'tvector',tv, ...
        'r0',R0,'req',Req,'rho8',rho8,'p8',P8,'kappa',kappa, ...
        'bubtherm',1,'medtherm',1,'masstrans',1,'vapor',1,'nt',25,'mt',25, ...
        'collapse',1,'stress',3,'g',G,'mu',mu, ...
        'lambda1',2*t0,'lambda2',0.4*t0};

% f_call_params returns init_stress as its fourth output
[eqns_opts, solve_opts, init_opts, init_stress] = f_call_params(argv{:});

fprintf('\n--- IMRv2 collapse initial stress ---\n');
fprintf('Szero            = %.16g\n', init_stress);
fprintf('Req_zero         = %.16g\n', init_opts(7));
fprintf('Rzero            = %.16g\n', init_opts(1));
fprintf('Rdotzero         = %.16g\n', init_opts(2));
fprintf('Pb_star          = %.16g\n', init_opts(3));
fprintf('Pv_star          = %.16g\n', init_opts(6));
fprintf('\nimr-fast precursor   = -0.159945110\n');
fprintf('reference-implied    = -0.160046461\n');
