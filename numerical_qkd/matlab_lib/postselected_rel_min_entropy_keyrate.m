% postselected_rel_min_entropy_keyrate
%====================
% Solve the dual problem for min entropy
% Author: Darius Bunandar (dariusb@mit.edu)
% Unauthorized use and/or duplication of this material without express and
% written permission from the author and/or owner is strictly prohibited.
%====================

function Hmin = postselected_rel_min_entropy_keyrate(...
    key_map_povm, n_basis, n_value_A, ...
    Gamma_exact, gamma, exact_const_exist, ...
    Gamma_inexact, gamma_ub, gamma_lb, inexact_const_exist, ...
    kraus_ops, ...
    dims, p_pass, trace_val, solver, verbose)

    mat_size_AB = prod(dims);
    sz_AB = [mat_size_AB, mat_size_AB];
    mat_size_ABAv = mat_size_AB * n_value_A;
    mat_size_ApBp = n_basis * mat_size_ABAv;
    sz_ApBp = [mat_size_ApBp, mat_size_ApBp];
    
    n_Gamma_ex = length(Gamma_exact);
    n_Gamma_inex = length(Gamma_inexact);
    
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
        variable Y11_list([mat_size_ABAv, mat_size_ABAv, n_basis]) complex
        variable Y22(sz_ApBp) hermitian semidefinite
        variable y_id
        variable y nonnegative
        variable zi_vec(n_Gamma_ex)
        variable yi_vec(n_Gamma_inex) nonnegative
        variable xi_vec(n_Gamma_inex) nonnegative
        
        % define objectives and other variables
        obj_exact = 0;
        zi_sum = zeros(sz_AB);
        if exact_const_exist
            for i = 1:n_Gamma_ex
                obj_exact = obj_exact + zi_vec(i) * gamma(i);
                zi_sum = zi_sum + zi_vec(i) * Gamma_exact{i};
            end
        end
        
        obj_inexact = 0;
        xi_sum = zeros(sz_AB);
        yi_sum = zeros(sz_AB);
        if inexact_const_exist
            for i = 1:n_Gamma_inex
                obj_inexact = obj_inexact + yi_vec(i) * gamma_ub(i) - xi_vec(i) * gamma_lb(i);
                yi_sum = yi_sum + yi_vec(i) * Gamma_inexact{i};
                xi_sum = xi_sum + xi_vec(i) * Gamma_inexact{i};
            end
        end
        
        sum_of_Gammas = zi_sum + yi_sum - xi_sum;
        
        for i = 1:n_basis
            Y11_cell{i} = Y11_list(:,:,i);
        end
        Y11 = blkdiag(Y11_cell{:});
        
        zero_mat = zeros(sz_ApBp);
        Y = [Y11, zero_mat; zero_mat, Y22];
        id_mat = eye(sz_ApBp);
        A = .5*[zero_mat, id_mat; id_mat, zero_mat];
        
        cq_Y22 = zeros(sz_ApBp);
        n_povm = length(key_map_povm);
        for i = 1:n_povm
            povm = key_map_povm{i};
            povm_ = kron(eye(n_basis * mat_size_AB), povm);
            cq_Y22 = cq_Y22 + povm_ * Y22 * povm_;
        end
        
        Y11_AB = zeros(sz_AB);
        for i = 1:n_basis
            mat = Y11_cell{i};
            k_AB = kraus_ops{i};
            Y11_AB = Y11_AB + k_AB' * mat * k_AB;
        end
        
        % constraints
        Y11 >= 0;
        Y11 == Y11';
        y * eye(sz_ApBp) >= cq_Y22;
        sum_of_Gammas + y_id * eye(sz_AB) >= Y11_AB;
        Y >= A;
        
        % objective function
        minimize(y + y_id*trace_val + obj_exact + obj_inexact)
        
    cvx_end
    Hmin = -2*log2(cvx_optval) + log2(p_pass);
end