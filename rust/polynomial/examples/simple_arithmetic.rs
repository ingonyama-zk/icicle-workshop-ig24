use rand::random;

use icicle_bn254::{curve::ScalarField as Fr, polynomials::DensePolynomial as polybn254};

use icicle_core::{
    ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig},
    polynomials::UnivariatePolynomial,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::memory::HostSlice;
use icicle_runtime::{self, Device};

fn init_ntt_domain(max_ntt_size: u64) {
    // Initialize NTT domain for all fields. Polynomial operations rely on NTT.
    println!(
        "Initializing NTT domain for max size 2^{}",
        max_ntt_size.trailing_zeros()
    );
    let rou_bn254: Fr = get_root_of_unity(max_ntt_size);
    initialize_domain(rou_bn254, &NTTInitDomainConfig::default()).unwrap();
}

/// Function to load the CUDA backend and initialize the GPU or fallback to CPU
fn setup_device() -> Device {
    // Attempt to load the CUDA backend with better error handling
    match icicle_runtime::load_backend("../../cuda_backend") {
        Ok(_) => println!("Successfully loaded the CUDA backend."),
        Err(error) => {
            eprintln!("Failed to load the CUDA backend: {:?}", error);
            std::process::exit(1); // Exit the program with an error code
        }
    };

    // Initialize CPU and GPU devices
    let cpu_device = Device::new("CPU", 0);
    let mut gpu_device = Device::new("CUDA", 0);

    // Check if the CUDA device (GPU) is available
    let is_gpu_available = icicle_runtime::is_device_available(&gpu_device);

    if is_gpu_available {
        println!("GPU detected and will be used for computations.");
    } else {
        // If the GPU is not available, fall back to the CPU
        println!("GPU not available. Using CPU for computations.");
        gpu_device = cpu_device.clone();
    }
    gpu_device
}

// Run with: cargo run --package polynomial-icicle --example simple_arithmetic
fn main() {
    let gpu_device = setup_device();

    // Set up a constant size used for polynomial operations
    let polynomial_size: usize = 1024;
    println!("Polynomial size: {}", polynomial_size);

    // Display polynomial identities that will be verified
    println!("Checking the following polynomial identities:");
    println!("1. (p1 + p2)^2 + (p1 - p2)^2 = 2 * (p1^2 + p2^2)");
    println!("2. (p1 + p2)^2 - (p1 - p2)^2 = 4 * p1 * p2");

    // Define the log size for the domain used in number-theoretic transformations
    // This has to be greater than polynomial_log_size + 2
    let log_domain_size = 12;
    println!("Domain size (log): {}", log_domain_size);

    // Set the active device for computation
    icicle_runtime::set_device(&gpu_device).unwrap();

    // Initialize the Number-Theoretic Transform (NTT) domain
    init_ntt_domain(1 << log_domain_size);

    // Set constants for polynomial operations
    let constant_two = Fr::from_u32(2);
    let constant_four = Fr::from_u32(4);

    // Generate a random value `random_point` for testing the Schwarz-Zippel lemma
    let random_point = Fr::from_u32(random::<u32>());

    // Generate two random vectors `vector1` and `vector2` of the given polynomial size
    let vector1 = <Fr as FieldImpl>::Config::generate_random(polynomial_size);
    let vector2 = <Fr as FieldImpl>::Config::generate_random(polynomial_size);

    // Create polynomials `poly1` and `poly2` from the generated vectors
    let poly1 = polybn254::from_rou_evals(HostSlice::from_slice(&vector1), polynomial_size);
    let poly2 = polybn254::from_rou_evals(HostSlice::from_slice(&vector2), polynomial_size);

    // Compute polynomial expressions
    let poly_sum_squared = &(&poly1 + &poly2) * &(&poly1 + &poly2); // (p1 + p2)^2
    let poly_diff_squared = &(&poly1 - &poly2) * &(&poly1 - &poly2); // (p1 - p2)^2
    let identity1_left = &poly_sum_squared + &poly_diff_squared; // (p1 + p2)^2 + (p1 - p2)^2
    let identity2_left = &poly_sum_squared - &poly_diff_squared; // (p1 + p2)^2 - (p1 - p2)^2
    let identity1_right = &(&(&poly1 * &poly1) + &(&poly2 * &poly2)) * &constant_two; // 2 * (p1^2 + p2^2)
    let identity2_right = &(&poly1 * &poly2) * &constant_four; // 4 * (p1 * p2)

    // Check the identities at a random point `random_point` using the Schwarz-Zippel lemma
    println!("Checking identities at random x = {:?}", random_point);

    let identity1_left_eval = identity1_left.eval(&random_point);
    let identity1_right_eval = identity1_right.eval(&random_point);
    let identity2_left_eval = identity2_left.eval(&random_point);
    let identity2_right_eval = identity2_right.eval(&random_point);

    // Verify the first identity
    assert_eq!(identity1_left_eval, identity1_right_eval);
    println!(
        "Identity 1 verified: (p1(x) + p2(x))^2 + (p1(x) - p2(x))^2 = 2 * (p1(x)^2 + p2(x)^2)"
    );

    // Verify the second identity
    assert_eq!(identity2_left_eval, identity2_right_eval);
    println!("Identity 2 verified: (p1(x) + p2(x))^2 - (p1(x) - p2(x))^2 = 4 * p1(x) * p2(x)");
}
