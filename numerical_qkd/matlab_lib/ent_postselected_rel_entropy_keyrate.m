% ent_postselected_rel_entropy_keyrate
%====================
% Solve the entangled postselected quantum relative entropy primal problem
% Author: Darius Bunandar (dariusb@mit.edu)
% Unauthorized use and/or duplication of this material without express and
% written permission from the author and/or owner is strictly prohibited.
%====================

function H = ent_postselected_rel_entropy_keyrate(...
    key_map_povm, n_povms, alice_prob, bob_prob, ...
    kraus_AB, ...
    Gamma_exact, gamma, ...
    Gamma_inexact, gamma_ub, gamma_lb, ...
    trace_val, ...
    dims, n_sift, p_pass, m, k, solver, verbose)

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
    
    prob = alice_prob .* bob_prob ./ p_pass;
    
    obj = 0;
    povm_count = 0;
    for i = 1:n_sift
        n_povm = n_povms(i);
        
        k_AB = kraus_AB{i};
        rho2 = k_AB * rho * k_AB';
        cq_rho2 = zeros(sz);
        for j = 1:n_povm
            povm_count = povm_count + 1;
            Z = key_map_povm{povm_count};
            cq_rho2 = cq_rho2 + Z*rho2*Z;
        end
        p = prob(i);
        obj = obj + p*quantum_rel_entr(rho2, cq_rho2, m, k);
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

H = p_pass * cvx_optval/log(2);  % change from nats to bits

end