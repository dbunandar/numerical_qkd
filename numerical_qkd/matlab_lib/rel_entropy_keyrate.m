% rel_entropy_keyrate
%====================
% Solve the primal optimization problem of quantum relative entropy
% Author: Darius Bunandar (dariusb@mit.edu)
% Unauthorized use and/or duplication of this material without express and
% written permission from the author and/or owner is strictly prohibited.
%====================

function H = rel_entropy_keyrate(key_map_povm, ...
    Gamma_exact, gamma, ...
    Gamma_inexact, gamma_ub, gamma_lb, ...
    dims, m, k, solver, verbose)

mat_size = prod(dims);
sz = [mat_size, mat_size];

cvx_begin sdp
    % solver selection
    if strcmp(solver, 'SEDUMI')
        cvx_solver sedumi
    elseif strcmp(solver, 'MOSEK')
        cvx_solver mosek
    elseif strcmp(solver, 'SDPT3')
        cvx_solver sdpt3
    else
        error('Solver is unavailable');
    end
    
    % whether to print out intermediate stpes
    if verbose
        cvx_quiet false
    else
        cvx_quiet true
    end

    % define variables
    variable rho(sz) hermitian semidefinite
    cq_rho = zeros(sz);
    for i = 1:length(key_map_povm)
        Z = key_map_povm{i};
        cq_rho = cq_rho + Z*rho*Z;
    end
    
    % constraints
    for i = 1:length(Gamma_exact)
        trace(rho * Gamma_exact{i}) == gamma(i);
    end
    for i = 1:length(Gamma_inexact)
        trace(rho * Gamma_inexact{i}) <= gamma_ub(i);
        trace(rho * Gamma_inexact{i}) >= gamma_lb(i);
    end
    trace(rho) == 1;  % normalized
    
    % objective function
    minimize(quantum_rel_entr(rho, cq_rho, m, k) );
cvx_end

H = cvx_optval/log(2);  % change from nats to bits

end