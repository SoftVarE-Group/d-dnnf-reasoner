use criterion::{Criterion, criterion_group, criterion_main};
use ddnnife::Ddnnf;
use std::hint::black_box;
use std::path::Path;

fn small_c2d(c: &mut Criterion) {
    let path = Path::new("../example_input/small_example_c2d.nnf");
    c.bench_function("load small (c2d)", |b| {
        b.iter(|| Ddnnf::from_file(black_box(path), None))
    });
}

fn busybox_c2d(c: &mut Criterion) {
    let path = Path::new("../example_input/busybox-1.18.0_c2d.nnf");
    c.bench_function("load busybox (c2d)", |b| {
        b.iter(|| Ddnnf::from_file(black_box(path), None))
    });
}

fn auto1_d4(c: &mut Criterion) {
    let path = Path::new("../example_input/auto1_d4_2513.nnf");
    c.bench_function("load auto1 (d4)", |b| {
        b.iter(|| Ddnnf::from_file(black_box(path), None))
    });
}

fn auto2_c2d(c: &mut Criterion) {
    let path = Path::new("../example_input/auto2_4_c2d.nnf");
    c.bench_function("load auto2 (c2d)", |b| {
        b.iter(|| Ddnnf::from_file(black_box(path), None))
    });
}

criterion_group!(benches, small_c2d, busybox_c2d, auto1_d4, auto2_c2d);
criterion_main!(benches);
