#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]

use anyhow::Result;
use clap::Parser;
use plonky2::timed;
use core::num::ParseIntError;
use log::{info, Level};
use plonky2::gates::noop::NoopGate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use plonky2::plonk::prover::prove;
use plonky2::util::timing::TimingTree;
use plonky2_field::extension::Extendable;
use plonky2_field::goldilocks_field::GoldilocksField;
use std::alloc::{AllocError, Allocator, Layout};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use plonky2::fri::oracle::CUDAAllocator;
use plonky2_field::fft::fft_root_table;
use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2_util::{log2_ceil, log2_strict};

#[cfg(feature = "cuda")]
use plonky2::fri::oracle::CudaInnerContext;

#[cfg(feature = "cuda")]
use rustacuda::memory::DeviceBuffer;
#[cfg(feature = "cuda")]
use rustacuda::memory::{cuda_malloc, DeviceBox};
#[cfg(feature = "cuda")]
use rustacuda::prelude::*;

// #[macro_use]
// extern crate rustacuda;
// extern crate rustacuda_core;

// extern crate cuda;
// use cuda::runtime::{CudaError, cudaMalloc, cudaMemcpy, cudaFree};
// use cuda::runtime::raw::{cudaError_t, cudaError_enum};

#[cfg(feature = "cuda")]
fn prove_sum_cuda<
    F: RichField + Extendable<D> + rustacuda::memory::DeviceCopy,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    a: &[u64],
    b: &[u64],
    c: &[u64],
) -> Result<()>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());

    let a_target = builder.add_virtual_target();
    let b_target = builder.add_virtual_target();
    let c_target = builder.add_virtual_public_input();

    let sum = builder.add(a_target, b_target);

    builder.connect(sum, c_target);

    while builder.num_gates() < 1 << 15 {
        // pad the circuit to 2^16 gates
        builder.add_gate(plonky2::gates::noop::NoopGate, vec![]);
    }

    println!("Constructing proof with {} gates", builder.num_gates());

    let data = builder.build::<C>();

    println!(
        "Circuit built with {} gates, {} gate instances, {} wires, {} public inputs",
        data.common.num_gates(),
        data.common.num_gate_constraints,
        data.common.config.num_wires,
        data.common.num_public_inputs
    );
    println!(
        "num_gate_constraints: {}, num_constants: {}, selectors_info: {:?}",
        data.common.num_gate_constraints, data.common.num_constants, data.common.selectors_info,
    );

    let mut ctx;
    {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device_index = 0;
        let device = rustacuda::prelude::Device::get_device(device_index).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let poly_num: usize = data.common.config.num_wires;
        let values_num_per_poly = 1 << 16; // Degree of the circuit
        let blinding = false; // Because zero knowledge is false
        const SALT_SIZE: usize = 4;
        let rate_bits = data.common.config.fri_config.rate_bits;
        let cap_height = data.common.config.fri_config.cap_height;

        let lg_n = log2_strict(values_num_per_poly);
        let n_inv = F::inverse_2exp(lg_n);
        let _n_inv_ptr: *const F = &n_inv;

        let fft_root_table_max = fft_root_table(1 << (lg_n + rate_bits)).concat();
        let fft_root_table_deg = fft_root_table(1 << lg_n).concat();

        let salt_size = if blinding { SALT_SIZE } else { 0 };
        let values_flatten_len = poly_num * values_num_per_poly;
        let ext_values_flatten_len =
            (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
        let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
        unsafe {
            ext_values_flatten.set_len(ext_values_flatten_len);
        }

        let mut values_flatten: Vec<F, CUDAAllocator> =
            Vec::with_capacity_in(values_flatten_len, CUDAAllocator {});
        unsafe {
            values_flatten.set_len(values_flatten_len);
        }

        let (values_flatten2, ext_values_flatten2) = {
            let poly_num = 20;
            let values_flatten_len = poly_num * values_num_per_poly;
            let ext_values_flatten_len =
                (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
            let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten: Vec<F, CUDAAllocator> =
                Vec::with_capacity_in(values_flatten_len, CUDAAllocator {});
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let (values_flatten3, ext_values_flatten3) = {
            let poly_num = data.common.config.num_challenges * (1 << rate_bits);
            let values_flatten_len = poly_num * values_num_per_poly;
            let ext_values_flatten_len =
                (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
            let mut ext_values_flatten: Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten: Vec<F, CUDAAllocator> =
                Vec::with_capacity_in(values_flatten_len, CUDAAllocator {});
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let len_cap = 1 << cap_height;
        let num_digests = 2 * (values_num_per_poly * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;
        let mut digests_and_caps_buf: Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash> =
            Vec::with_capacity(num_digests_and_caps);
        unsafe {
            digests_and_caps_buf.set_len(num_digests_and_caps);
        }

        let digests_and_caps_buf2 = digests_and_caps_buf.clone();
        let digests_and_caps_buf3 = digests_and_caps_buf.clone();
        let pad_extvalues_len = ext_values_flatten.len();

        let cache_mem_device = {
            let cache_mem_device = unsafe {
                DeviceBuffer::<F>::uninitialized(
                    pad_extvalues_len + ext_values_flatten_len + digests_and_caps_buf.len() * 4,
                )
            }
            .unwrap();

            cache_mem_device
        };

        let root_table_device = {
            let root_table_device = DeviceBuffer::from_slice(&fft_root_table_deg).unwrap();
            root_table_device
        };

        let root_table_device2 = {
            let root_table_device = DeviceBuffer::from_slice(&fft_root_table_max).unwrap();
            root_table_device
        };

        let constants_sigmas_commitment_leaves_device = DeviceBuffer::from_slice(
            &data
                .prover_only
                .constants_sigmas_commitment
                .merkle_tree
                .leaves
                .concat(),
        )
        .unwrap();

        let shift_powers = F::coset_shift()
            .powers()
            .take(1 << (lg_n))
            .collect::<Vec<F>>();
        let shift_powers_device = {
            let shift_powers_device = DeviceBuffer::from_slice(&shift_powers).unwrap();
            shift_powers_device
        };

        let shift_inv_powers = F::coset_shift()
            .powers()
            .take(1 << (lg_n + rate_bits))
            .map(|f| f.inverse())
            .collect::<Vec<F>>();
        let shift_inv_powers_device = {
            let shift_inv_powers_device = DeviceBuffer::from_slice(&shift_inv_powers).unwrap();
            shift_inv_powers_device
        };
        let quotient_degree_bits = log2_ceil(data.common.quotient_degree_factor);
        let points = F::two_adic_subgroup(data.common.degree_bits() + quotient_degree_bits);

        let z_h_on_coset = ZeroPolyOnCoset::new(data.common.degree_bits(), quotient_degree_bits);

        let points_device = DeviceBuffer::from_slice(&points).unwrap();
        let z_h_on_coset_evals_device = DeviceBuffer::from_slice(&z_h_on_coset.evals).unwrap();
        let z_h_on_coset_inverses_device =
            DeviceBuffer::from_slice(&z_h_on_coset.inverses).unwrap();
        let k_is_device = DeviceBuffer::from_slice(&data.common.k_is).unwrap();

        ctx = plonky2::fri::oracle::CudaInvContext {
            inner: CudaInnerContext { stream, stream2 },
            ext_values_flatten: Arc::new(ext_values_flatten),
            values_flatten: Arc::new(values_flatten),
            digests_and_caps_buf: Arc::new(digests_and_caps_buf),

            ext_values_flatten2: Arc::new(ext_values_flatten2),
            values_flatten2: Arc::new(values_flatten2),
            digests_and_caps_buf2: Arc::new(digests_and_caps_buf2),

            ext_values_flatten3: Arc::new(ext_values_flatten3),
            values_flatten3: Arc::new(values_flatten3),
            digests_and_caps_buf3: Arc::new(digests_and_caps_buf3),

            cache_mem_device,
            second_stage_offset: ext_values_flatten_len,
            root_table_device,
            root_table_device2,
            constants_sigmas_commitment_leaves_device,
            shift_powers_device,
            shift_inv_powers_device,

            points_device,
            z_h_on_coset_evals_device,
            z_h_on_coset_inverses_device,
            k_is_device,

            ctx: _ctx,
        };
    }

    for i in 0..a.len() {
        let mut pw = PartialWitness::new();
        pw.set_target(a_target, F::from_canonical_u64(a[i]))?;
        pw.set_target(b_target, F::from_canonical_u64(b[i]))?;
        pw.set_target(c_target, F::from_canonical_u64(c[i]))?;

        let mut timing = TimingTree::new("prove cpu", Level::Debug);
        let proof = prove(
            &data.prover_only,
            &data.common,
            pw,
            &mut timing,
            Some(&mut ctx),
        )?;
        timed!(
            timing,
            "verify",
            data.verify(proof.clone()).expect("verify error")
        );

        timing.print();
    }

    Ok(())
}

fn prove_sum<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    a: &[u64],
    b: &[u64],
    c: &[u64],
) -> Result<()>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());

    let a_target = builder.add_virtual_target();
    let b_target = builder.add_virtual_target();
    let c_target = builder.add_virtual_public_input();

    let sum = builder.add(a_target, b_target);

    builder.connect(sum, c_target);

    while builder.num_gates() < 1 << 15 {
        // pad the circuit to 2^16 gates
        builder.add_gate(plonky2::gates::noop::NoopGate, vec![]);
    }

    println!("Constructing proof with {} gates", builder.num_gates());

    println!("Number of LUTs: {}", builder.num_luts());

    builder.print_gate_counts(0);
    let data = builder.build::<C>();

    println!("Gates: {:?}", data.common.gates);
    println!(
        "Circuit built with {} gates, {} gate instances, {} wires, {} public inputs",
        data.common.num_gates(),
        data.common.num_gate_constraints,
        data.common.config.num_wires,
        data.common.num_public_inputs
    );
    println!(
        "num_gate_constraints: {}, num_constants: {}, selectors_info: {:?}",
        data.common.num_gate_constraints, data.common.num_constants, data.common.selectors_info,
    );

    for i in 0..a.len() {
        let mut pw = PartialWitness::new();
        pw.set_target(a_target, F::from_canonical_u64(a[i]))?;
        pw.set_target(b_target, F::from_canonical_u64(b[i]))?;
        pw.set_target(c_target, F::from_canonical_u64(c[i]))?;

        let mut timing = TimingTree::new("prove cpu", Level::Debug);

        let proof = prove(
            &data.prover_only,
            &data.common,
            pw,
            &mut timing,
            #[cfg(feature = "cuda")]
            None,
        )?;
        timed!(
            timing,
            "verify",
            data.verify(proof.clone()).expect("verify error")
        );

        timing.print();
    }

    Ok(())
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    r: Option<u64>,
}

fn main() -> Result<()> {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.try_init()?;

    let args = Cli::parse();

    // if args.sig.is_none() || args.pk.is_none() || args.msg.is_none() {
    //     println!("The required arguments were not provided: --msg MSG_IN_HEX  --pk PUBLIC_KEY_IN_HEX  --sig SIGNATURE_IN_HEX");
    //     return Ok(());
    // }

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let l = args.r.unwrap_or(1);
    let a = (0..l).map(|i| i as u64).collect::<Vec<u64>>();
    let b = (0..l).map(|i| (i + 1) as u64).collect::<Vec<u64>>();
    let c = (0..l).map(|i| (i + 1 + i) as u64).collect::<Vec<u64>>();

    #[cfg(feature = "cuda")]
    {
        prove_sum_cuda::<F, C, D>(&a, &b, &c)?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        prove_sum::<F, C, D>(&a, &b, &c)?;
    }
    return Ok(());
}
