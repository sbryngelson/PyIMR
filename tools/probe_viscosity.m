% Diagnostic: isolate where imr-fast and IMRv2 disagree on nu_model=1 (Carreau).
% Runs reduction limits that must collapse onto known Newtonian trajectories.
% Usage: matlab -batch "run('tools/probe_viscosity.m')"

imrv2 = getenv('IMRV2_ROOT');
if isempty(imrv2)
    imrv2 = fullfile(getenv('HOME'), 'research/docs/_ideas-boed/upstream/IMRv2');
end
addpath(fullfile(imrv2, 'src'));
outdir = tempdir;  % diagnostic only; nothing here is a pinned reference

R0 = 225e-6; Req = R0/6; tv = linspace(0, 120e-6, 300);
rho8 = 1064; P8 = 101325; t0 = R0/sqrt(P8/rho8);
G = 2500; kappa = 1.4;
base = {'progdisplay',0,'method',23,'dimout',0,'tvector',tv, ...
        'r0',R0,'req',Req,'rho8',rho8,'p8',P8,'kappa',kappa,'stress',1,'g',G};

probes = {
  'newt_hi',   [base, {'mu',0.5}]                                                       % eta = 0.5
  'newt_lo',   [base, {'mu',0.1}]                                                       % eta = 0.1
  'cy_n1',     [base, {'mu',0.1,'mu0',0.5,'nu_model',1,'v_nc',1.0,'v_lambda',1e-5}]     % f == 1 -> 0.5
  'cy_lamsm',  [base, {'mu',0.1,'mu0',0.5,'nu_model',1,'v_nc',0.5,'v_lambda',1e-14}]    % f -> 1 -> 0.5
  'cy_lamlg',  [base, {'mu',0.1,'mu0',0.5,'nu_model',1,'v_nc',0.5,'v_lambda',1e2}]      % f -> 0 -> 0.1
};

R = struct();
for k = 1:size(probes,1)
    tag = probes{k,1};
    try
        [~, r] = f_imr_fd(probes{k,2}{:});
        R.(tag) = r(:);
        fprintf('%-10s OK    min R/R0=%.6f\n', tag, min(r));
        writematrix(r(:), fullfile(outdir, sprintf('probe_%s.csv', tag)), 'FileType','text');
    catch err
        fprintf('%-10s FAIL  %s\n', tag, err.message);
    end
end

fprintf('\nreduction checks (max|dR|):\n');
if isfield(R,'cy_n1')    && isfield(R,'newt_hi'), fprintf('  cy(n=1)      vs newt(0.5): %.3e\n', max(abs(R.cy_n1-R.newt_hi))); end
if isfield(R,'cy_lamsm') && isfield(R,'newt_hi'), fprintf('  cy(lambda~0) vs newt(0.5): %.3e\n', max(abs(R.cy_lamsm-R.newt_hi))); end
if isfield(R,'cy_lamlg') && isfield(R,'newt_lo'), fprintf('  cy(lambda>>) vs newt(0.1): %.3e\n', max(abs(R.cy_lamlg-R.newt_lo))); end
