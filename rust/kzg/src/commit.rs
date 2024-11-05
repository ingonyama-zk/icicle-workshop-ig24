use std::error::Error;

use icicle_babybear::field::ScalarField as babybearScalar;
use icicle_babybear::polynomials::DensePolynomial as PolynomialBabyBear;
use icicle_bn254::curve::{G1Affine, ScalarField as bn254Scalar};
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;

use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
    Device,
};

use icicle_core::{
    msm::MSMConfig,
    vec_ops::{
        accumulate_scalars, add_scalars, bit_reverse, bit_reverse_inplace, mul_scalars,
        sub_scalars, transpose_matrix, VecOps, VecOpsConfig,
    },
};

use icicle_core::{
    curve::Affine,
    field::Field,
    ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig},
    polynomials::UnivariatePolynomial,
    traits::{FieldImpl, GenerateRandom},
};

use icicle_bn254::curve::{
    CurveCfg as BN254CurveCfg, G1Projective, G2CurveCfg, G2Projective, ScalarCfg as BN254ScalarCfg,
    ScalarField as BN254ScalarField,
};
use icicle_core::{curve::Curve, msm};

/// Loads the specified backend from a custom path and sets the device for computation.
///
/// ### Arguments
/// * `device_type` - A string slice that specifies the type of device to use (e.g., "CPU" or "GPU").
/// * `backend_path` - A string slice specifying the custom path to the backend library.
/// * `device_id` - An integer specifying the device id (in case multiple devices are used).
///   An error will be thrown if the backend fails to load from this path.
///
/// ### Returns
/// * `Result<(), Box<dyn Error>>` - Returns `Ok(())` on success, or an error message wrapped in a `Box<dyn Error>` if any operation fails.
///
/// ### Behavior
/// - If `device_type` is not "CPU", the function attempts to load the backend from the specified path.
/// - If backend loading fails, a detailed error is returned.
/// - The device is created and set using the provided `device_type`.
///
pub fn try_load_and_set_backend_device(
    device_type: &str,
    backend_path: &str,
    device_id: i32,
) -> Result<(), Box<dyn Error>> {
    if device_type != "CPU" {
        icicle_runtime::load_backend(backend_path)
            .map_err(|e| format!("Failed to load backend from {}: {:?}", backend_path, e))?;
    }

    println!("Setting device {}", device_type);

    let device = Device::new(device_type, device_id);
    icicle_runtime::set_device(&device).map_err(|e| format!("Failed to set device: {:?}", e))?;

    Ok(())
}

/// Computes the sequence [1, x, x^2, ..., x^(n-1)] for a given scalar `x` and size `n`
/// using repeated Hadamard products.
///
/// # Arguments
/// * `x` - The scalar value.
/// * `n` - The size of the sequence (number of elements).
///
/// # Returns
/// * A `Result` containing a vector of `F` elements representing the sequence,
///   or an error if the computation fails.
pub fn compute_powers_of_scalar<F: FieldImpl>(x: F, n: usize) -> Result<Vec<F>, Box<dyn Error>>
where
    <F as FieldImpl>::Config: VecOps<F>,
{
    // Initialize a vector with the first element as 1 and the rest as zero
    let mut result = vec![F::one(); n];

    // Set up an Icicle stream and configuration
    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = VecOpsConfig::default();
    cfg.stream_handle = *stream;

    // Use `mul_scalars` iteratively to compute each power of `x`
    for i in 1..n {
        let a = [result[i - 1].clone()]; // The previous power
        let b = [x.clone()]; // The scalar x
        let mut res = [F::zero()]; // Placeholder for the result

        // Perform scalar multiplication using `mul_scalars`
        mul_scalars(
            HostSlice::from_slice(&a),
            HostSlice::from_slice(&b),
            HostSlice::from_mut_slice(&mut res),
            &cfg,
        )
        .unwrap();

        result[i] = res[0]; // Store the result in the sequence
    }

    // Destroy the Icicle stream
    stream.destroy().unwrap();

    Ok(result)
}

pub fn generate_srs(log_max_size: u8) -> Vec<Affine<BN254CurveCfg>> {
    let max_size = 1 << log_max_size;
    let secret_tau = BN254ScalarCfg::generate_random(1)[0];
    let powers_of_secret_tau: Vec<Field<8, BN254ScalarCfg>> =
        compute_powers_of_scalar(secret_tau, max_size).unwrap();

    let generator = BN254CurveCfg::generate_random_affine_points(1)[0];
    let generator_vector = vec![generator; max_size];
    let trial = powers_of_secret_tau[0] * generator;

    // Wrap the scalars and points in HostSlices
    let scalars_slice = HostSlice::from_slice(&powers_of_secret_tau);
    let points_slice = HostSlice::from_slice(&generator_vector);

    // Allocate space for the MSM result on the device
    let mut srs_points = DeviceVec::<G1Projective>::device_malloc(1).unwrap();

    // Create an Icicle stream and configure MSM
    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = *stream;
    cfg.is_async = true;

    println!("Executing bn254 MSM on device...");
    msm::msm(scalars_slice, points_slice, &cfg, &mut srs_points[..]).unwrap();

    println!("Moving results to host...");
    let mut srs_points_host = vec![G1Projective::zero(); 1];

    stream.synchronize().unwrap();
    srs_points
        .copy_to_host(HostSlice::from_mut_slice(&mut srs_points_host[..]))
        .unwrap();

    // Destroy the stream to free resources
    stream.destroy().unwrap();

    // Convert the results from projective to affine form
    srs_points_host
        .into_iter()
        .map(|point| {
            let mut point_proj = G1Affine::zero();
            Curve::to_affine(&point, &mut point_proj);
            point_proj
        })
        .collect()
}

pub fn commit(
    witness_scalars: &Vec<Field<8, BN254ScalarCfg>>,
    srs_points: &Vec<Affine<BN254CurveCfg>>,
) {
}

/// Generates a random polynomial of the specified size, either from coefficients or
/// from evaluations at the roots of unity, based on the `from_coeffs` flag.
///
/// ### Arguments
/// * `size` - The size of the polynomial (number of coefficients or evaluations).
/// * `from_coeffs` - A boolean flag indicating how to construct the polynomial:
///   - `true`: Construct from coefficients.
///   - `false`: Construct from evaluations at the roots of unity.
///
/// ### Returns
/// * A randomized polynomial of type `P`.
///
pub fn randomize_poly<P>(size: usize, from_coeffs: bool) -> P
where
    P: UnivariatePolynomial,
    P::Field: FieldImpl,
    P::FieldConfig: GenerateRandom<P::Field>,
{
    println!(
        "Randomizing polynomial of size {} (from_coeffs: {})",
        size, from_coeffs
    );
    let random_values = P::FieldConfig::generate_random(size);

    if from_coeffs {
        P::from_coeffs(HostSlice::from_slice(&random_values), size)
    } else {
        P::from_rou_evals(HostSlice::from_slice(&random_values), size)
    }
}
