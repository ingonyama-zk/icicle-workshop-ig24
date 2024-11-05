use icicle_babybear::field::{ScalarCfg, ScalarField};
use icicle_core::{
    ntt::{self, get_root_of_unity, initialize_domain, ntt, NTTConfig},
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::{self, Device};









