use criterion::{Criterion, criterion_group, criterion_main};
use ddnnife::Ddnnf;
use std::{hint::black_box, path::Path};

static BENCHMARKS: [(&str, &str); 6] = [
    ("auto1_d4.nnf", "auto1 (d4)"),
    ("auto2_c2d.nnf", "auto2 (c2d)"),
    ("axTLS_d4.nnf", "axTLS (d4)"),
    ("busybox_c2d.nnf", "BusyBox (c2d)"),
    ("VP9_d4.nnf", "VP9 (d4)"),
    ("X264_c2d.nnf", "X264 (c2d)"),
];

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat");
    let data_dir = Path::new("tests/data");

    for (path, name) in BENCHMARKS {
        let ddnnf = Ddnnf::from_file(&data_dir.join(path), None);
        group.bench_function(name, |bencher| {
            bencher.iter(|| ddnnf.sat_immutable(black_box(&[])))
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
