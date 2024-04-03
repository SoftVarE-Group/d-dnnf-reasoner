//! Traits for using sequential or parallel iterators based on the `parallel` feature using a unified API.
//!
//! This makes it possible to call a single method on an iterable type and use `std`'s or `rayon`'s iterator implementation.

// This is basically a copy of how `rayon` defines and implements the corresponding traits.
// The only difference is that the trait fields and implementations are feature-guarded to use/call
// the sequential/parallel methods respectively.
// For each trait, there exists one parallel and one sequential implementation.

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Conversion into an iterator.
pub trait IntoMaybeParallelIterator {
    #[cfg(feature = "parallel")]
    type Iter: ParallelIterator<Item = Self::Item>;

    #[cfg(not(feature = "parallel"))]
    type Iter: Iterator<Item = Self::Item>;

    #[cfg(feature = "parallel")]
    type Item: Send;

    #[cfg(not(feature = "parallel"))]
    type Item;

    /// Depending on the `parallel` feature, either translates to `into_iter` or `into_par_iter`.
    fn into_maybe_par_iter(self) -> Self::Iter;
}

#[cfg(feature = "parallel")]
impl<I> IntoMaybeParallelIterator for I
where
    I: IntoParallelIterator,
{
    type Iter = <I as IntoParallelIterator>::Iter;
    type Item = <I as IntoParallelIterator>::Item;

    fn into_maybe_par_iter(self) -> Self::Iter {
        self.into_par_iter()
    }
}

#[cfg(not(feature = "parallel"))]
impl<I> IntoMaybeParallelIterator for I
where
    I: IntoIterator,
{
    type Iter = <I as IntoIterator>::IntoIter;
    type Item = <I as IntoIterator>::Item;

    fn into_maybe_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

/// A possibly parallel iterator from a reference.
pub trait IntoMaybeParallelRefIterator<'data> {
    #[cfg(feature = "parallel")]
    type Iter: ParallelIterator<Item = Self::Item>;

    #[cfg(not(feature = "parallel"))]
    type Iter: Iterator<Item = Self::Item>;

    #[cfg(feature = "parallel")]
    type Item: Send + 'data;

    #[cfg(not(feature = "parallel"))]
    type Item: 'data;

    /// Depending on the `parallel` feature, either translates to `into_iter` or `into_par_iter`.
    fn maybe_par_iter(&'data self) -> Self::Iter;
}

#[cfg(feature = "parallel")]
impl<'data, I: 'data + ?Sized> IntoMaybeParallelRefIterator<'data> for I
where
    &'data I: IntoParallelIterator,
{
    type Iter = <&'data I as IntoParallelIterator>::Iter;
    type Item = <&'data I as IntoParallelIterator>::Item;

    fn maybe_par_iter(&'data self) -> Self::Iter {
        self.into_par_iter()
    }
}

#[cfg(not(feature = "parallel"))]
impl<'data, I: 'data> IntoMaybeParallelRefIterator<'data> for I
where
    &'data I: IntoIterator,
{
    type Iter = <&'data I as IntoIterator>::IntoIter;
    type Item = <&'data I as IntoIterator>::Item;

    fn maybe_par_iter(&'data self) -> Self::Iter {
        self.into_iter()
    }
}

/// A possibly parallel iterator from a reference.
pub trait IntoMaybeParallelRefMutIterator<'data> {
    #[cfg(feature = "parallel")]
    type Iter: ParallelIterator<Item = Self::Item>;

    #[cfg(not(feature = "parallel"))]
    type Iter: Iterator<Item = Self::Item>;

    #[cfg(feature = "parallel")]
    type Item: Send + 'data;

    #[cfg(not(feature = "parallel"))]
    type Item: 'data;

    /// Depending on the `parallel` feature, either translates to `into_iter` or `into_par_iter`.
    fn maybe_par_iter_mut(&'data mut self) -> Self::Iter;
}

#[cfg(feature = "parallel")]
impl<'data, I: 'data + ?Sized> IntoMaybeParallelRefMutIterator<'data> for I
where
    &'data mut I: IntoParallelIterator,
{
    type Iter = <&'data mut I as IntoParallelIterator>::Iter;
    type Item = <&'data mut I as IntoParallelIterator>::Item;

    fn maybe_par_iter_mut(&'data mut self) -> Self::Iter {
        self.into_par_iter()
    }
}

#[cfg(not(feature = "parallel"))]
impl<'data, I: 'data> IntoMaybeParallelRefMutIterator<'data> for I
where
    &'data mut I: IntoIterator,
{
    type Iter = <&'data mut I as IntoIterator>::IntoIter;
    type Item = <&'data mut I as IntoIterator>::Item;

    fn maybe_par_iter_mut(&'data mut self) -> Self::Iter {
        self.into_iter()
    }
}

/// An extension on `Iterator` for functions which are different on the parallel variant.
pub trait PolyfillIterator {
    type Item;

    /// Translates to `Iterator::find_map`.
    fn find_map_first<P, R>(&mut self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R> + Sync + Send,
        R: Send;
}

impl<I, T> PolyfillIterator for I
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn find_map_first<P, R>(&mut self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R> + Sync + Send,
        R: Send,
    {
        self.find_map(predicate)
    }
}
