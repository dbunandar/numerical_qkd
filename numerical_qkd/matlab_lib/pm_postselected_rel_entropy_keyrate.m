% pm_postselected_rel_entropy_keyrate
%====================
% Solve the P&M postselected quantum relative entropy primal problem
% Author: Darius Bunandar (dariusb@mit.edu)
% Unauthorized use and/or duplication of this material without express and
% written permission from the author and/or owner is strictly prohibited.
%====================

function H = pm_postselected_rel_entropy_keyrate(...
    key_map_povm, n_povms, indices, bob_prob, ...
    kraus_AB, ...
    Gamma_exact, gamma, ...
    Gamma_inexact, gamma_ub, gamma_lb, ...
    trace_val, ...
    dims, n_sift, m, k, solver, verbose)

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
        
    obj = 0;
    povm_count = 0;
    for i = 1:n_sift
        n_povm = n_povms(i);
        
        k_AB = kraus_AB{i};
        rho2 = k_AB * rho * k_AB';
        
        ind = indices{i};
        rho_curr = rho2(ind, ind);
        cq_rho = zeros(length(ind));
        for j = 1:n_povm
            povm_count = povm_count + 1;
            Z = key_map_povm{povm_count};
            cq_rho = cq_rho + Z*rho_curr*Z;
        end
        p = bob_prob(i);
        obj = obj + p*quantum_rel_entr(rho_curr, cq_rho, m, k);
    end
    
    % constraints
    for i = 1:length(Gamma_exact)
        trace(rho * Gamma_exact{i}) == gamma(i);
    end
    for i = 1:length(Gamma_inexact)
        trace(rho * Gamma_inexact{i}) <= gamma_ub(i);
        trace(rho * Gamma_inexact{i}) >= gamma_lb(i);
    end
    trace(rho) == trace_val;  % normalized
    
    % objective function
    minimize(obj);
cvx_end

H = cvx_optval/log(2);  % change from nats to bits

end