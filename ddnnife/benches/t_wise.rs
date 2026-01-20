use criterion::{Criterion, criterion_group, criterion_main};
use ddnnife::Ddnnf;
use std::hint::black_box;
use std::path::Path;

fn bench_t_wise(c: &mut Criterion, name: &str, path: &Path, t: usize) {
    let ddnnf = Ddnnf::from_file(path, None);
    let id = format!("t-wise {name} t={t}");
    c.bench_function(&id, |bencher| {
        bencher.iter(|| ddnnf.sample_t_wise(black_box(t)))
    });
}

fn small_c2d_2(c: &mut Criterion) {
    let path = Path::new("../example_input/small_example_c2d.nnf");
    bench_t_wise(c, "small (c2d)", path, 2);
}

fn small_c2d_3(c: &mut Criterion) {
    let path = Path::new("../example_input/small_example_c2d.nnf");
    bench_t_wise(c, "small (c2d)", path, 3);
}

fn busybox_c2d_2(c: &mut Criterion) {
    let path = Path::new("../example_input/busybox-1.18.0_c2d.nnf");
    bench_t_wise(c, "busybox (c2d)", path, 2);
}

fn busybox_c2d_3(c: &mut Criterion) {
    let path = Path::new("../example_input/busybox-1.18.0_c2d.nnf");
    bench_t_wise(c, "busybox (c2d)", path, 3);
}

criterion_group!(benches, small_c2d_2, small_c2d_3, busybox_c2d_2,);
criterion_main!(benches);
