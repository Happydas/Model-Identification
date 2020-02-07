import numpy as np
import torch

def pde_matrix_mul(u, du):
    """
    :param u: input data polynomials.
    :param du: input data spatial derivatives.
    :return:  matrix multiplication for the given data u and du.
    """
    matrix_mul = []
    for u_ele in u:
        for du_ele in du:
            if ((u_ele == '1') and ('du_el' == '1')):
                mul = '1'
            elif u_ele == '1':
                mul = du_ele
            elif du_ele == '1':
                mul = u_ele
            else:
                mul = u_ele + du_ele
            matrix_mul.append(mul)
    return matrix_mul

#xi = torch.randn((9, 1), device="cpu", dtype=torch.float32)
#print(xi)


def sparse_coeff(mask, xi):
    sparse_xi = np.zeros_like(mask, dtype=np.float)
    #numpy.where() iterates over the bool array and for every True it yields corresponding element array x and for every False it yields corresponding element from array y.
    # So, basically it returns an array of elements from x where condition is True, and elements from y elsewhere.
    sparse_xi[np.where(mask)[0]] = xi
    return sparse_xi


def pde_Recover(xi, library_coeffs, equation_form='u_t'):
    '''
    Prints PDE with non-zero components according to xi.
    Set equation_form  for PDE equations.
    '''
    # Return the indices of the elements that are non-zero.
    id = np.nonzero(xi)[0]
    Equation = equation_form + ' = '
    # Returns an element-wise indication of the sign of a number.
    for idx, form in enumerate(id):
        if idx != 0:
            if np.sign(xi[form]) == -1:
                Equation += ' - '
            else:
                Equation += ' + '
        Equation += '%.4f' % np.abs(xi[form]) + library_coeffs[form]
    print('Burger equation:')
    return Equation


def normalized_xi_threshold(xi, mode, tresld=0.0):
    if mode == 'auto':
        high_order, low_order = np.median(xi)+np.std(xi), np.median(xi) - np.std(xi)
        sparsity = (xi <= high_order) & (xi >= low_order)
    else:
        sparsity = np.abs(xi) < tresld
    return ~sparsity

