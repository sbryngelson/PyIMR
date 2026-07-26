% Generate pinned IMRv2 reference trajectories for cases imr-fast currently
% validates only against its own reduction limits (PLAN.md W1).
%
% Usage:  matlab -batch "run('tools/gen_imrv2_cases.m')"
%
% Writes tests/ref_*.csv (300-point R/R0 columns, matching tests/ref_t.csv)
% and prints a per-case PASS/FAIL log. Cases that error upstream are reported
% rather than silently skipped.

imrv2 = getenv('IMRV2_ROOT');
if isempty(imrv2)
    imrv2 = fullfile(getenv('HOME'), 'research/docs/_ideas-boed/upstream/IMRv2');
end
addpath(fullfile(imrv2, 'src'));

here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'tests');

% match tests/ref_t.csv and the imr-fast defaults exactly
R0    = 225e-6;
Req   = R0/6;
tfin  = 120e-6;
npts  = 300;
tv    = linspace(0, tfin, npts);
rho8  = 1064;
P8    = 101325;
t0    = R0/sqrt(P8/rho8);
G     = 2500;
mu    = 0.1;
kappa = 1.4;          % imr-fast default; IMRv2 default_case ships 1.47

base = {'progdisplay',0,'method',23,'dimout',0,'tvector',tv, ...
        'r0',R0,'req',Req,'rho8',rho8,'p8',P8,'kappa',kappa};

cases = {};

% ---- distributed nonlinear memory -------------------------------------
% NOT GENERATED. f_call_params.m:376-385 dispatches stress 6 (PTT) and 7
% (Giesekus) and forces spectral=1 for both, but the input gate at
% f_call_params.m:206-208 rejects stress > 5. Giesekus and PTT are therefore
% unreachable in IMRv2 at dea31cd -- confirmed empirically, see PLAN.md W1.
% Retained here so the gate is retested if upstream is ever bumped.
for s = [6 7]
    cases{end+1} = {sprintf('ref_stress%d_probe.csv', s), ...
        [base, {'stress',s,'eps3',0.2,'mu',mu,'g',G, ...
                'lambda1',2*t0,'lambda2',0.4*t0,'nv',150}]};
end

% ---- non-Newtonian viscosity, nu_model 1..7 ---------------------------
% f_viscosity.m assigns intf/dintf/ddintf only for nu_model 1 and 2; models
% 3-7 set f alone. f_imr_fd.m:419 requests all four outputs whenever
% nu_model ~= 0, so 3-7 are expected to raise. Probe rather than assume.
for nm = 1:7
    cases{end+1} = {sprintf('ref_numodel%d.csv', nm), ...
        [base, {'stress',1,'g',G,'mu',mu,'nu_model',nm, ...
                'mu0',0.5,'v_a',2.0,'v_nc',0.5,'v_lambda',1e-5}]};
end

% ---- collapse initialization ------------------------------------------
% f_call_params.m:234-236 requires the full coupled model: bubtherm,
% medtherm, masstrans and vapor all 1.
%
% Only stress 3 and 4 get a real precursor. f_call_params.m:447-460 leaves
% Szero empty for stress < 3 and returns zeros under an explicit "TODO" for
% stress == 5, and the recomputed Req_zero comes back identical (0.16666667).
% So collapse=1 is a no-op for NHKV and Oldroyd-B: those two files pin the
% fully coupled model WITHOUT a precursor, which is why they are named
% ref_coupled_*. Verified -- imr-fast reproduces them without collapse to
% 6.2e-06 and 1.6e-05 respectively.
full = {'bubtherm',1,'medtherm',1,'masstrans',1,'vapor',1,'nt',25,'mt',25};
cases{end+1} = {'ref_collapse_zener.csv', ...
    [base, full, {'collapse',1,'stress',3,'g',G,'mu',mu, ...
                  'lambda1',2*t0,'lambda2',0.4*t0}]};
cases{end+1} = {'ref_coupled_oldb.csv', ...
    [base, full, {'collapse',1,'stress',5,'mu',mu, ...
                  'lambda1',2*t0,'lambda2',0.4*t0}]};
cases{end+1} = {'ref_coupled_nhkv.csv', ...
    [base, full, {'collapse',1,'stress',1,'g',G,'mu',mu}]};

% ---- radial 6 and 7 (PLAN W4) -----------------------------------------
% The gate at f_call_params.m:187 permits radial 1..7 despite its message
% naming only 1..4. imr-fast supports 1..5; probe what 6 and 7 actually do.
for rr = [6 7]
    cases{end+1} = {sprintf('ref_radial%d.csv', rr), ...
        [base, {'radial',rr,'stress',1,'g',G,'mu',mu}]};
end

% ---- run ---------------------------------------------------------------
fprintf('\n%-34s %-8s %s\n', 'case', 'status', 'detail');
fprintf('%s\n', repmat('-', 1, 78));
nok = 0; nbad = 0;
for k = 1:numel(cases)
    name = cases{k}{1};
    argv = cases{k}{2};
    try
        [t, R] = f_imr_fd(argv{:});
        if abs(R(1) - 1) > 1e-6
            error('R(1)=%.6g, expected normalised R/R0', R(1));
        end
        if numel(R) ~= npts
            R = interp1(t, R, tv, 'pchip')';
        end
        if any(~isfinite(R))
            error('non-finite radius at %d of %d points', sum(~isfinite(R)), numel(R));
        end
        % a complex trajectory is an upstream failure, not a usable reference
        if ~isreal(R)
            error('non-real radius: max|imag|=%.3e over %d of %d points', ...
                max(abs(imag(R))), sum(imag(R) ~= 0), numel(R));
        end
        writematrix(R(:), fullfile(outdir, name), 'FileType', 'text');
        fprintf('%-34s %-8s min R/R0=%.6f\n', name, 'OK', min(R));
        nok = nok + 1;
    catch err
        fprintf('%-34s %-8s %s\n', name, 'FAIL', err.message);
        nbad = nbad + 1;
    end
end
fprintf('%s\n%d generated, %d failed\n', repmat('-', 1, 78), nok, nbad);
