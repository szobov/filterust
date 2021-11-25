extern crate nalgebra as na;
extern crate blas_src;

type DMatrixF = na::DMatrix<f64>;
type DVectorF = na::DVector<f64>;


#[derive(Debug)]
#[allow(non_snake_case)]
struct VanDerMarweSigmaPoints {

    x_size: usize,
    L_lambda: f64,

    W_m: DVectorF,
    W_c: DVectorF,
}

#[allow(non_snake_case)]
impl VanDerMarweSigmaPoints {
    fn new(x_size: usize, alpha: f64, beta: f64, kappa: f64) -> VanDerMarweSigmaPoints {
        let L = x_size as f64;
        let lambda = alpha.powi(2) * (L + kappa) - L;
        println!("Lambda: {}, alpha**2: {}, kappa: {}", lambda, alpha.powi(2), kappa);
        let L_lambda = L + lambda;

        let weight_const = 1.0 / (2.0 * L_lambda);
        let mut W_m = DVectorF::from_element(2 * x_size + 1, weight_const);
        let mut W_c = DVectorF::from_element(2 * x_size + 1, weight_const);

        W_m[0] = lambda / L_lambda;
        W_c[0] = W_m[0] + (1.0 - alpha.powi(2) + beta);

        VanDerMarweSigmaPoints {
            x_size: x_size,
            L_lambda: L_lambda,
            W_m: W_m,
            W_c: W_c
        }
    }

    fn calculate(&self, x_mean: &DMatrixF, P_covarience: &DMatrixF) -> DMatrixF {
        let scaled_root_cov = na::linalg::Cholesky::new(self.L_lambda * P_covarience).unwrap().l().transpose();

        let mut Chi = DMatrixF::from_element(2 * self.x_size + 1, self.x_size, 0.0);

        println!("x_mean: {}, P_cov: {}", x_mean, self.L_lambda * P_covarience);
        println!("scaled_root_cov: {}", (scaled_root_cov));

        Chi.row_mut(0).tr_copy_from(&x_mean);
        for i in 0..self.x_size {
            Chi.row_mut(i + 1).copy_from(&(x_mean.transpose() + scaled_root_cov.row(i)))
        }
        for i in self.x_size..(2 * self.x_size) {
            Chi.row_mut(i + 1).copy_from(&(x_mean.transpose() - scaled_root_cov.row(i - self.x_size)))
        }
        Chi
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    extern crate ndarray as nd;

    use matrixcompare::{assert_matrix_eq};

    #[test]
    #[allow(non_snake_case)]
    fn van_der_marwe_sigma_points_test() {
        let x = na::DMatrix::from_vec(1, 2, vec![1., 2.]).transpose();
        let P = na::DMatrix::from_vec(2, 2, vec![1., 1.1,
                                                 1.1, 3.]);
        let x_size = x.len();
        let alpha = 1e-1;
        let beta = 2.;
        let kappa = -1.;
        let sigmas = VanDerMarweSigmaPoints::new(x_size, alpha, beta, kappa);




        let res = sigmas.calculate(&x, &P);
        println!("res: {}", res);

        let expected_sigmas =  na::DMatrix::from_vec(x_size, 2 * x_size + 1,
                               vec![1.,          2.,
                                    1.1,         2.11,
                                    1.,          2.13379088,
                                    0.9,         1.89,
                                    1.,          1.86620912]).transpose();
        let expected_W_c = na::DVector::from_vec(vec![-196.01,   50.  ,   50.  ,   50.  ,   50.  ]);
        let expected_W_m = na::DVector::from_vec(vec![-199.,   50.,   50.,   50.,   50.]);


        assert_matrix_eq!(&res, &expected_sigmas, comp = abs, tol=1e-8);
        assert_matrix_eq!(&sigmas.W_c, &expected_W_c, comp = abs, tol=1e-8);
        assert_matrix_eq!(&sigmas.W_m, &expected_W_m, comp = abs, tol=1e-8);

    }

    #[test]
    fn ndarray_matrix_test() {
        let a = nd::Array::range(0.0, 9.0, 1.0).into_shape((3, 3)).unwrap();
        let b = a.t();
        let res = nd::Array::from_shape_vec((3, 3), vec![5.,  14.,  23.,
                                                         14.,  50.,  86.,
                                                         23.,  86., 149.]).unwrap();
        assert!(res.abs_diff_eq(&a.dot(&b), 1e-12));
    }
}
