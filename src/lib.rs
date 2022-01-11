extern crate nalgebra as na;
extern crate blas_src;

use std::collections::HashMap;

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

type ProcessFunctT = dyn Fn(&DMatrixF, f64) -> DMatrixF;
type MeasurementFunctT = dyn Fn(&DMatrixF, &HashMap<String, DMatrixF>) -> DMatrixF;

struct UKF {

    x: DMatrixF,  // state
    P: DMatrixF,  // state uncertainty
    dt: f64,

    Q: DMatrixF,  // process noise

    sigma_points: VanDerMarweSigmaPoints,

    process_func: Box<ProcessFunctT>,
    measurement_func: Box<MeasurementFunctT>,
    additional_params: HashMap<String, DMatrixF>,
}

impl UKF {

    // fn new() -> UKF {
    // }

    #[allow(non_snake_case)]
    fn predict_and_update(&mut self, z: &DMatrixF, R: &DMatrixF) {
        let chi = self.sigma_points.calculate(&self.x, &self.P);
        let gamma = (self.process_func)(&chi, self.dt);
        let (x_prior, P_prior) = unscented_transform(&gamma, &self.sigma_points.W_m, &self.sigma_points.W_c, &self.Q);

        let Z = (self.measurement_func)(&gamma, &self.additional_params);
        let (x_z, P_z) = unscented_transform(&Z, &self.sigma_points.W_m, &self.sigma_points.W_c, &R);
        let y = z - x_z.clone();
        let P_xz = na::DMatrix::from_rows(&[(self.sigma_points.W_c.clone() * (gamma - x_prior.clone()) * (Z - x_z).transpose()).row_sum()]);
        let K = P_xz.clone() * P_z.try_inverse().unwrap();
        self.x = x_prior + K.clone() * y;
        self.P = P_prior - K.clone() * P_xz * K.transpose();
    }
}

#[allow(non_snake_case)]
fn unscented_transform(y: &DMatrixF, W_m: &DVectorF, W_c: &DVectorF, Q: &DMatrixF) -> (DMatrixF, DMatrixF){
    let x_prior = na::DMatrix::from_rows(&[(W_m * y).row_sum()]).clone_owned();
    let P_prior = na::DMatrix::from_rows(&[(W_c * (y - x_prior.clone()) * (y - x_prior.clone()).transpose()).row_sum()]) + Q;
    (x_prior, P_prior)
}



#[cfg(test)]
#[allow(non_snake_case)]
mod tests;
