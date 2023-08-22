pub fn format_vec<T: ToString>(vals: impl Iterator<Item = T>) -> String {
    vals.map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(" ")
}

pub fn format_vec_vec<T>(vals: impl Iterator<Item = T>) -> String
where
    T: IntoIterator,
    T::Item: ToString,
{
    vals.map(|res| format_vec(res.into_iter()))
        .collect::<Vec<String>>()
        .join(";")
}
