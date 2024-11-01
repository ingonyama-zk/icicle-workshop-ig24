use icicle_babybear::field::ScalarField as babybearScalar;
use icicle_babybear::polynomials::DensePolynomial as PolynomialBabyBear;
use icicle_bn254::curve::ScalarField as bn254Scalar;
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;

use icicle_runtime::memory::{DeviceVec, HostSlice};

use icicle_core::{
    ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig},
    polynomials::UnivariatePolynomial,
    traits::{FieldImpl, GenerateRandom},
};

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

fn randomize_poly<P>(size: usize, from_coeffs: bool) -> P
where
    P: UnivariatePolynomial,
    P::Field: FieldImpl,
    P::FieldConfig: GenerateRandom<P::Field>,
{
    println!(
        "Randomizing polynomial of size {} (from_coeffs: {})",
        size, from_coeffs
    );
    let coeffs_or_evals = P::FieldConfig::generate_random(size);
    let p = if from_coeffs {
        P::from_coeffs(HostSlice::from_slice(&coeffs_or_evals), size)
    } else {
        P::from_rou_evals(HostSlice::from_slice(&coeffs_or_evals), size)
    };
    p
}
