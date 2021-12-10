use super::*;

use matrixcompare::assert_matrix_eq;

use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug)]
struct RadarStation {
    position: na::Matrix1x2<f64>,
    range_std: f64,
    elevation_angle_std: f64,
}

impl RadarStation {
    fn new(pose: &na::Matrix1x2<f64>, range_std: f64, elevation_angle_std: f64) -> RadarStation {
        RadarStation {
            position: pose.clone_owned(),
            range_std: range_std,
            elevation_angle_std: elevation_angle_std,
        }
    }

    fn read_aircraft_position(&self, aircraft_position: &na::Matrix1x2<f64>) -> (f64, f64) {
        let diff = aircraft_position - self.position;
        let range = diff.norm() as f64;
        let elevation = diff[(0, 1)].atan2(diff[(0, 0)]);
        (range, elevation)
    }

    fn read_with_noise(&self, aircraft_position: &na::Matrix1x2<f64>) -> (f64, f64) {
        let (range, bearing) = self.read_aircraft_position(aircraft_position);
        let range_noised: f64 =
            range + rand::thread_rng().sample::<f64, _>(StandardNormal) * self.range_std;
        let bearing_noised = bearing
            + rand::thread_rng().sample::<f64, _>(StandardNormal) * self.elevation_angle_std;
        (range_noised, bearing_noised)
    }
}

#[derive(Debug)]
struct AirCraftSim {
    position: na::Matrix1x2<f64>,
    velocity: na::Matrix1x2<f64>,
    velocity_std: f64,
}

impl AirCraftSim {
    fn new(
        position: &na::Matrix1x2<f64>,
        velocity: &na::Matrix1x2<f64>,
        velocity_std: f64,
    ) -> AirCraftSim {
        AirCraftSim {
            position: position.clone_owned(),
            velocity: velocity.clone_owned(),
            velocity_std: velocity_std,
        }
    }

    fn update(&mut self, dt: f64) -> na::Matrix1x2<f64> {
        let dx = (self.velocity * dt).add_scalar(
            (rand::thread_rng().sample::<f64, _>(StandardNormal) * self.velocity_std) * dt,
        );
        self.position += dx;
        self.position
    }
}

fn fx(x: &DMatrixF, dt: f64) -> DMatrixF {
    let mut F = na::DMatrix::identity(3, 3);
    F[(0, 1)] = dt;
    F * x
}

fn hx(x: &DMatrixF, init_params: &HashMap<&str, &na::Matrix1x2<f64>>) -> DMatrixF {
    let dx = x[(0, 0)] - init_params["radar_pose"][(0, 0)];
    let dy = x[(0, 2)] - init_params["radar_pose"][(0, 1)];
    let slant_range = (dx.powi(2) + dy.powi(2)).sqrt();
    let elevation_angle = dy.atan2(dx);
    na::DMatrix::from_vec(1, 2, vec![slant_range, elevation_angle])
}

#[test]
fn ukf_test() {
    let x = na::DMatrix::from_vec(1, 2, vec![1., 2.]).transpose();
    let P = na::DMatrix::from_vec(2, 2, vec![1., 1.1, 1.1, 3.]);
    let x_size = x.len();
    let alpha = 1e-1;
    let beta = 2.;
    let kappa = -1.;
    let sigmas = VanDerMarweSigmaPoints::new(x_size, alpha, beta, kappa);

    let dt = 3.;
    let range_std = 5.;
    let elevation_angle_std = 0.5_f64.to_radians();
    let aircraft_sim = AirCraftSim::new(&na::Matrix1x2::new(0., 1000.), &na::Matrix1x2::new(100., 0.), 0.02);

    let radar_pose = na::Matrix1x2::new(0., 0.);
    let init_params = HashMap::from([("radar_pose", &radar_pose)]);
    let radar_sim = RadarStation::new(&radar_pose, range_std, elevation_angle_std);



}

#[test]
#[allow(non_snake_case)]
fn van_der_marwe_sigma_points_test() {
    let x = na::DMatrix::from_vec(1, 2, vec![1., 2.]).transpose();
    let P = na::DMatrix::from_vec(2, 2, vec![1., 1.1, 1.1, 3.]);
    let x_size = x.len();
    let alpha = 1e-1;
    let beta = 2.;
    let kappa = -1.;
    let sigmas = VanDerMarweSigmaPoints::new(x_size, alpha, beta, kappa);

    let res = sigmas.calculate(&x, &P);
    println!("res: {}", res);

    let expected_sigmas = na::DMatrix::from_vec(
        x_size,
        2 * x_size + 1,
        vec![1., 2., 1.1, 2.11, 1., 2.13379088, 0.9, 1.89, 1., 1.86620912],
    )
    .transpose();
    let expected_W_c = na::DVector::from_vec(vec![-196.01, 50., 50., 50., 50.]);
    let expected_W_m = na::DVector::from_vec(vec![-199., 50., 50., 50., 50.]);

    assert_matrix_eq!(&res, &expected_sigmas, comp = abs, tol = 1e-8);
    assert_matrix_eq!(&sigmas.W_c, &expected_W_c, comp = abs, tol = 1e-8);
    assert_matrix_eq!(&sigmas.W_m, &expected_W_m, comp = abs, tol = 1e-8);
}
