//! Cost of the type-erased micropool `ParJoin::for_each` paths.
//!
//! `for_each` does not hand the join type and callback to micropool/paralight
//! (which would monomorphize their whole pipeline per join, callback and split
//! strategy); it erases the callback instead:
//!
//! * `chunked`  — `micropool_join_with_cache(..).for_each(f)`: one indirect
//!   call per work unit, the per-entity loop stays monomorphic.
//! * `per_item` — `.with_thread_pool(P).for_each(f)`: `P`'s partitioning is
//!   opaque, so the fallback erases one call per entity.
//! * `serial`   — single-threaded `join().for_each`, for scale.
//!
//! Before the erasure, `chunked` was measured against the old generic path
//! and was not slower (faster for trivial per-entity work).

extern crate criterion;
extern crate specs;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use specs::micropool;
use specs::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Pos([f32; 3]);

impl Component for Pos {
    type Storage = VecStorage<Self>;
}

#[derive(Clone, Copy, Debug)]
struct Vel([f32; 3]);

impl Component for Vel {
    type Storage = VecStorage<Self>;
}

const DT: f32 = 1.0 / 60.0;

#[inline(always)]
fn trivial(p: &mut Pos, v: &Vel) {
    for k in 0..3 {
        p.0[k] += v.0[k] * DT;
    }
}

#[inline(always)]
fn spin(p: &mut Pos, v: &Vel, rounds: u32) {
    let mut x = p.0;
    for _ in 0..rounds {
        for k in 0..3 {
            x[k] = x[k].mul_add(0.999, v.0[k] * DT);
        }
    }
    p.0 = x;
}

#[derive(Clone, Copy)]
enum Work {
    Trivial,
    Spin(u32),
}

impl Work {
    fn name(self) -> String {
        match self {
            Work::Trivial => "trivial".to_owned(),
            Work::Spin(rounds) => format!("spin_{rounds}"),
        }
    }
}

/// Binds `$f` to the per-entity workload closure and evaluates `$run`, so each
/// workload is monomorphized like a real call site.
macro_rules! dispatch_work {
    ($work:expr, |$f:ident| $run:expr) => {
        match $work {
            Work::Trivial => {
                let $f = |(p, v): (&mut Pos, &Vel)| trivial(p, v);
                $run
            }
            Work::Spin(rounds) => {
                let $f = |(p, v): (&mut Pos, &Vel)| spin(p, v, black_box(rounds));
                $run
            }
        }
    };
}

fn make_world(n: usize) -> World {
    let mut world = World::new();
    world.register::<Pos>();
    world.register::<Vel>();
    for i in 0..n {
        world
            .create_entity()
            .with(Pos([i as f32, 0.0, 1.0]))
            .with(Vel([1.0, 0.5, 0.25]))
            .build();
    }
    world
}

fn bench(c: &mut Criterion) {
    for work in [Work::Trivial, Work::Spin(32), Work::Spin(256)] {
        let mut group = c.benchmark_group(format!("micropool_dyn/{}", work.name()));
        for n in [1_000usize, 10_000, 100_000] {
            let world = make_world(n);
            let mut pos = world.write_storage::<Pos>();
            let vel = world.read_storage::<Vel>();
            let mut cache = ParJoinCache::new();
            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("serial", n), |b| {
                b.iter(|| dispatch_work!(work, |f| (&mut pos, &vel).join().for_each(f)))
            });
            group.bench_function(BenchmarkId::new("chunked", n), |b| {
                b.iter(|| {
                    dispatch_work!(work, |f| (&mut pos, &vel)
                        .micropool_join_with_cache(&mut cache)
                        .for_each(f))
                })
            });
            group.bench_function(BenchmarkId::new("per_item", n), |b| {
                b.iter(|| {
                    dispatch_work!(work, |f| (&mut pos, &vel)
                        .micropool_join_with_cache(&mut cache)
                        .with_thread_pool(micropool::split_by_threads())
                        .for_each(f))
                })
            });
        }
        group.finish();
    }
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(40)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
}

criterion::criterion_group! {
    name = benches;
    config = config();
    targets = bench
}
criterion::criterion_main!(benches);
