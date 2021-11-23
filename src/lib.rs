extern crate nalgebra as na;
extern crate conv;

extern crate ndarray as nd;
extern crate ndarray_linalg as nd_linalg;

extern crate blas_src;


type DMatrixF = na::DMatrix<f64>;
type DVectorF = na::DVector<f64>;


struct SigmaPoints {
    Chi: DMatrixF,
    Weights: DMatrixF
}



fn matrix_mult(left: &DMatrixF, right: &DMatrixF) -> DMatrixF {
    left * right
}


#[allow(non_snake_case)]
fn van_der_merwe_sigma_points(x_mean: &DVectorF, P_covarience: &DMatrixF, alpha: f64, beta: f64, kappa: f64) {
    // alpha = 1e-3
    // beta = 2
    // kappa = 0
    let (_, n) = x_mean.shape();
    let L = n as f64;
    let lambda = alpha.powi(2) * (L + kappa) - L;
    let mut Chi = DMatrixF::from_element(2 * n + 1, n, 0.0);

    let L_lambda = L + lambda;
    let scaled_root_cov = na::linalg::Cholesky::new(L_lambda * P_covarience).unwrap().l();

    Chi.set_row(0, &na::RowDVector::from_vec(x_mean.data.as_vec().clone()));
    Chi.row_mut(1).tr_copy_from(&x_mean);
    for i in 1..n {
        Chi.row_mut(i).tr_copy_from(&(x_mean + scaled_root_cov.row(i)))
    }
    for i in n..(2 * n) {
        Chi.row_mut(i).tr_copy_from(&(x_mean - scaled_root_cov.row(i)))
    }


    let weight_const = 1.0 / (2.0 * L_lambda);
    let mut W_m = DVectorF::from_element(2 * n + 1, weight_const);
    let mut W_c = DVectorF::from_element(2 * n + 1, weight_const);

    W_m[0] = lambda / L_lambda;
    W_c[0] = W_m[0] + (1.0 - alpha.powi(2) + beta);
}



#[cfg(test)]
mod tests {
    use super::*;

    extern crate ndarray as nd;

    // use approx::AbsDiffEq;

    use matrixcompare::{assert_matrix_eq};

    use na::{DMatrix, Matrix3};

    #[test]
    fn simple_matrix_test() {
        let iter = (0..9).map(|x| x as f64);
        // from_iterator initialize matrix in column-wise order (different from numpy)
        let a = DMatrix::from_iterator(3, 3, iter.clone()).transpose();
        let b = a.transpose();
        let res = Matrix3::new( 5.,  14.,  23.,
                                14.,  50.,  86.,
                                23.,  86., 149.);

        assert_matrix_eq!(&res, matrix_mult(&a, &b), comp = abs, tol=1e-12);
    }

    #[test]
    fn ndarray_matrix_test() {
        let a = nd::Array::range(0.0, 9.0, 1.0).into_shape((3, 3)).unwrap();
        let b = a.t();
        let res = nd::Array::from_shape_vec((3, 3), vec![5.,  14.,  23.,
                                                         14.,  50.,  86.,
                                                         23.,  86., 149.]).unwrap();
        assert!(res.abs_diff_eq(&a.dot(&b), 1e-12));



        // check cholesky
    }
}
