#[cfg(feature = "parallel")]
use hibitset::BitProducer;
use hibitset::BitSetLike;
#[cfg(feature = "micropool")]
use micropool::iter::{
    Accumulator, ExactParallelSourceExt, ExactSizeAccumulator, GenericThreadPool,
    IntoExactParallelRefSource, IntoExactParallelSource,
    ParallelIterator as MicropoolParallelIterator,
    ParallelIteratorExt as MicropoolParallelIteratorExt, ParallelSourceExt,
};
#[cfg(feature = "parallel")]
use rayon::iter::plumbing::UnindexedProducer;
#[cfg(feature = "parallel")]
use rayon::iter::{
    ParallelIterator,
    plumbing::{Folder, UnindexedConsumer, bridge_unindexed},
};
#[cfg(feature = "micropool")]
use smallvec::SmallVec;
#[cfg(feature = "micropool")]
use std::num::NonZeroUsize;
#[cfg(feature = "micropool")]
use std::ops::ControlFlow;
#[cfg(feature = "micropool")]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::world::Index;

#[cfg(feature = "micropool")]
const DEFAULT_MIN_ITEMS_PER_WORK_UNIT: NonZeroUsize =
    NonZeroUsize::new(1).expect("default micropool work-unit size must be non-zero");

// `hibitset` has four hierarchy levels with one machine word per level.
// Its maximum index count is therefore `usize::BITS.pow(4)`.
#[cfg(feature = "micropool")]
const HIBITSET_INDEX_COUNT: usize = hibitset::BitSet::BITS_PER_USIZE.pow(4);

#[cfg(feature = "micropool")]
static WARNED_UNCONSTRAINED_MICROPOOL_JOIN: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "micropool")]
#[derive(Clone, Copy, Debug)]
enum MicropoolSplit {
    Adaptive {
        max_parallelism: Option<NonZeroUsize>,
        min_items_per_work_unit: NonZeroUsize,
    },
    PerItem,
    Per(usize),
    By(usize),
    ByThreads,
}

#[cfg(feature = "micropool")]
impl Default for MicropoolSplit {
    fn default() -> Self {
        Self::Adaptive {
            max_parallelism: None,
            min_items_per_work_unit: DEFAULT_MIN_ITEMS_PER_WORK_UNIT,
        }
    }
}

/// Reusable entity-index storage for [`ParJoin::micropool_join_with_cache`].
///
/// The first 32 indices are stored inline. Larger joins retain their heap
/// allocation for subsequent runs, making this suitable for storage directly
/// on an ECS `System`. Unconstrained joins (for example, a tuple containing
/// only `MaybeJoin`s) bypass this cache and use a range source without
/// materializing the full index array.
#[cfg(feature = "micropool")]
#[derive(Debug, Default)]
pub struct ParJoinCache {
    indices: SmallVec<[Index; 32]>,
}

#[cfg(feature = "micropool")]
impl ParJoinCache {
    /// Creates an empty reusable join cache without allocating.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn refill<M: BitSetLike>(&mut self, mask: &M) {
        self.indices.clear();
        self.indices.extend(mask.iter());
    }
}

/// The purpose of the `ParJoin` trait is to provide a way
/// to access multiple storages in parallel at the same time with
/// the merged bit set.
///
/// # Safety
///
/// `ParJoin::get` must be callable from multiple threads, simultaneously.
///
/// The `Self::Mask` value returned with the `Self::Value` must correspond such
/// that it is safe to retrieve items from `Self::Value` whose presence is
/// indicated in the mask. As part of this, `BitSetLike::iter` must not produce
/// an iterator that repeats an `Index` value.
pub unsafe trait ParJoin {
    /// Type of joined components.
    type Type;
    /// Type of joined storages.
    type Value;
    /// Type of joined bit mask.
    type Mask: BitSetLike;

    /// Create a Rayon parallel iterator over the contents.
    #[cfg(feature = "parallel")]
    fn par_join(self) -> JoinParIter<Self>
    where
        Self: Sized,
    {
        if Self::is_unconstrained() {
            log::warn!(
                "`ParJoin` possibly iterating through all indices, \
                you might've made a join with all `MaybeJoin`s, \
                which is unbounded in length."
            );
        }

        // Open on the calling thread: guards stay there, while only the
        // mask and value views satisfying Send/Sync reach Rayon workers.
        // SAFETY: the paired views remain private to JoinParIter.
        let (keys, values) = unsafe { self.open() };
        JoinParIter { keys, values }
    }

    /// Create a micropool parallel iterator over the contents.
    ///
    /// The iterator uses micropool's current pool, or its global pool when no
    /// pool was installed on the calling thread. Use
    /// [`micropool_join_with`](Self::micropool_join_with) when invoking this
    /// from a task running on an explicitly created micropool.
    #[cfg(feature = "micropool")]
    fn micropool_join(self) -> JoinMicropoolIter<'static, Self>
    where
        Self: Sized,
    {
        JoinMicropoolIter {
            cache: None,
            index_bound: None,
            join: self,
            pool: None,
            split: MicropoolSplit::default(),
        }
    }

    /// Create a micropool parallel iterator using an explicit thread pool.
    ///
    /// Passing the active pool explicitly also keeps nested joins on the same
    /// pool.
    #[cfg(feature = "micropool")]
    fn micropool_join_with(self, pool: &micropool::ThreadPool) -> JoinMicropoolIter<'_, Self>
    where
        Self: Sized,
    {
        JoinMicropoolIter {
            cache: None,
            index_bound: None,
            join: self,
            pool: Some(pool),
            split: MicropoolSplit::default(),
        }
    }

    /// Create a micropool parallel iterator backed by reusable index storage.
    ///
    /// Store the cache on the owning ECS `System` to keep index capacity across
    /// frames. The mutable borrow prevents the same cache from being reused by
    /// overlapping joins.
    #[cfg(feature = "micropool")]
    fn micropool_join_with_cache(self, cache: &mut ParJoinCache) -> JoinMicropoolIter<'_, Self>
    where
        Self: Sized,
    {
        JoinMicropoolIter {
            cache: Some(cache),
            index_bound: None,
            join: self,
            pool: None,
            split: MicropoolSplit::default(),
        }
    }

    /// Open this join by returning the mask and the storages.
    ///
    /// # Safety
    ///
    /// This is unsafe because implementations of this trait can permit the
    /// `Value` to be mutated independently of the `Mask`. If the `Mask` does
    /// not correctly report the status of the `Value` then illegal memory
    /// access can occur.
    unsafe fn open(self) -> (Self::Mask, Self::Value);

    /// Get a joined component value by a given index.
    ///
    /// # Safety
    ///
    /// * A call to `get` must be preceded by a check if `id` is part of
    ///   `Self::Mask`.
    /// * The value returned from this method must no longer be alive before
    ///   subsequent calls with the same `id`.
    unsafe fn get(value: &Self::Value, id: Index) -> Self::Type;

    /// If this `LendJoin` typically returns all indices in the mask, then
    /// iterating over only it or combined with other joins that are also
    /// dangerous will cause the `JoinLendIter` to go through all indices which
    /// is usually not what is wanted and will kill performance.
    #[inline]
    fn is_unconstrained() -> bool {
        false
    }
}

/// `JoinParIter` is a `ParallelIterator` over a group of storages.
#[cfg(feature = "parallel")]
#[must_use]
pub struct JoinParIter<J: ParJoin> {
    keys: J::Mask,
    values: J::Value,
}

#[cfg(feature = "parallel")]
impl<J> ParallelIterator for JoinParIter<J>
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Type: Send,
    J::Value: Send + Sync,
{
    type Item = J::Type;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        // SAFETY: `keys` and `values` are not exposed outside this module and
        // we only use `values` for calling `ParJoin::get`.
        let Self { keys, values } = self;
        // Create a bit producer which splits on up to three levels
        let producer = BitProducer((&keys).iter(), 3);

        bridge_unindexed(JoinProducer::<J>::new(producer, &values), consumer)
    }
}

#[cfg(feature = "parallel")]
struct JoinProducer<'a, J>
where
    J: ParJoin,
    J::Mask: Send + Sync + 'a,
    J::Type: Send,
    J::Value: Send + Sync + 'a,
{
    keys: BitProducer<'a, J::Mask>,
    values: &'a J::Value,
}

#[cfg(feature = "parallel")]
impl<'a, J> JoinProducer<'a, J>
where
    J: ParJoin,
    J::Type: Send,
    J::Value: 'a + Send + Sync,
    J::Mask: 'a + Send + Sync,
{
    fn new(keys: BitProducer<'a, J::Mask>, values: &'a J::Value) -> Self {
        JoinProducer { keys, values }
    }
}

#[cfg(feature = "parallel")]
impl<'a, J> UnindexedProducer for JoinProducer<'a, J>
where
    J: ParJoin,
    J::Type: Send,
    J::Value: 'a + Send + Sync,
    J::Mask: 'a + Send + Sync,
{
    type Item = J::Type;

    fn split(self) -> (Self, Option<Self>) {
        let (cur, other) = self.keys.split();
        let values = self.values;
        let first = JoinProducer::new(cur, values);
        let second = other.map(|o| JoinProducer::new(o, values));

        (first, second)
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let JoinProducer { values, keys, .. } = self;
        // SAFETY: `idx` is obtained from the `Mask` returned by
        // `ParJoin::open`. The indices here are guaranteed to be distinct
        // because of the fact that the bit set is split and because `ParJoin`
        // requires that the bit set iterator doesn't repeat indices.
        let iter = keys.0.map(|idx| unsafe { J::get(values, idx) });

        folder.consume_iter(iter)
    }
}

#[cfg(feature = "micropool")]
fn warn_if_unconstrained<J: ParJoin>(index_bound: usize) {
    if !WARNED_UNCONSTRAINED_MICROPOOL_JOIN.swap(true, Ordering::Relaxed) {
        let join_type = std::any::type_name::<J>();
        log::warn!(
            "`MicropoolJoin<{join_type}>` has no bounded join member, possibly \
            because every member is a `MaybeJoin`. Micropool will scan \
            {index_bound} candidate indices without materializing them; use \
            `with_index_bound` with `EntitiesRes::index_bound`, or add a \
            bounded join member."
        );
    }
}

#[cfg(feature = "micropool")]
#[inline]
fn unconstrained_index_bound<J: ParJoin>(index_bound: Option<usize>) -> usize {
    let index_bound = index_bound
        .unwrap_or(HIBITSET_INDEX_COUNT)
        .min(HIBITSET_INDEX_COUNT);
    warn_if_unconstrained::<J>(index_bound);
    index_bound
}

/// A micropool parallel iterator over a group of storages.
///
/// Bounded entity indices are materialized before dispatch because micropool's
/// parallel-iterator backend operates on indexed sources. Joined component
/// values are still fetched lazily on worker threads. Unconstrained masks are
/// streamed over hibitset's finite index range instead, avoiding a 64 MiB
/// index allocation on 64-bit targets.
#[cfg(feature = "micropool")]
#[must_use = "iterator adaptors are lazy"]
pub struct JoinMicropoolIter<'pool, J> {
    cache: Option<&'pool mut ParJoinCache>,
    index_bound: Option<usize>,
    join: J,
    pool: Option<&'pool micropool::ThreadPool>,
    split: MicropoolSplit,
}

#[cfg(feature = "micropool")]
impl<'pool, J> JoinMicropoolIter<'pool, J>
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
{
    /// Limits this join to at most `max_parallelism` execution lanes.
    ///
    /// The calling thread counts as one lane. This limits the number of work
    /// units exposed by this join; it does not reserve threads, so the actual
    /// parallelism may be lower when the pool is busy or adaptive batching
    /// produces fewer work units. Without this modifier, the current pool's
    /// full capacity is used as the upper bound. Calling this after a native
    /// `split_*` method switches the join back to adaptive scheduling.
    ///
    /// # Panics
    ///
    /// Panics when `max_parallelism` is zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// (&storage)
    ///     .micropool_join()
    ///     .with_max_parallelism(3)
    ///     .for_each(process);
    /// ```
    #[must_use]
    pub fn with_max_parallelism(mut self, max_parallelism: usize) -> Self {
        let max_parallelism = Some(
            NonZeroUsize::new(max_parallelism)
                .expect("micropool join max parallelism must be greater than zero"),
        );
        let min_items_per_work_unit = match self.split {
            MicropoolSplit::Adaptive {
                min_items_per_work_unit,
                ..
            } => min_items_per_work_unit,
            _ => DEFAULT_MIN_ITEMS_PER_WORK_UNIT,
        };
        self.split = MicropoolSplit::Adaptive {
            max_parallelism,
            min_items_per_work_unit,
        };
        self
    }

    /// Sets the minimum number of joined items placed in each work unit.
    ///
    /// The default is one, allowing every item to be stolen when a small join
    /// contains expensive work. Higher values batch cheap items to reduce
    /// scheduling overhead. This setting composes with
    /// [`Self::with_max_parallelism`]. Calling it after a native `split_*`
    /// method switches the join back to adaptive scheduling.
    ///
    /// # Panics
    ///
    /// Panics when `min_items` is zero.
    #[must_use]
    pub fn with_min_items_per_work_unit(mut self, min_items: usize) -> Self {
        let min_items_per_work_unit = NonZeroUsize::new(min_items)
            .expect("micropool join minimum items per work unit must be greater than zero");
        let max_parallelism = match self.split {
            MicropoolSplit::Adaptive {
                max_parallelism, ..
            } => max_parallelism,
            _ => None,
        };
        self.split = MicropoolSplit::Adaptive {
            max_parallelism,
            min_items_per_work_unit,
        };
        self
    }

    /// Bounds an unconstrained join to candidate indices in
    /// `0..index_bound`.
    ///
    /// Use [`EntitiesRes::index_bound`](crate::world::EntitiesRes::index_bound)
    /// to replace hibitset's full 32/64-bit-dependent index space with the
    /// current world's entity-allocation high-water mark. The value is a
    /// snapshot and is clamped to hibitset's maximum supported index count.
    ///
    /// This setting has no effect on a join with at least one bounded member;
    /// those joins continue to iterate their merged mask through
    /// [`ParJoinCache`].
    #[must_use]
    pub fn with_index_bound(mut self, index_bound: usize) -> Self {
        self.index_bound = Some(index_bound.min(HIBITSET_INDEX_COUNT));
        self
    }

    /// Uses micropool's native `split_per_item` strategy.
    ///
    /// Every joined entity becomes a separate work unit. This maximizes steal
    /// opportunities for a small number of expensive or uneven tasks, but it
    /// also maximizes scheduling overhead and per-work-unit terminal storage.
    /// Bound an unconstrained join before selecting this strategy; otherwise
    /// the work-unit count is hibitset's entire index space. Selecting any
    /// `split_*` strategy replaces the adaptive `with_*` settings.
    #[must_use]
    pub fn split_per_item(mut self) -> Self {
        self.split = MicropoolSplit::PerItem;
        self
    }

    /// Uses micropool's native `split_per` strategy.
    ///
    /// Each work unit contains up to `chunk_size` joined entities. Like
    /// micropool itself, a value of zero is treated as one.
    #[must_use]
    pub fn split_per(mut self, chunk_size: usize) -> Self {
        self.split = MicropoolSplit::Per(chunk_size);
        self
    }

    /// Uses micropool's native `split_by` strategy.
    ///
    /// The joined entities are divided into `chunks` work units. Work units
    /// may outnumber worker threads, improving load balancing for uneven
    /// workloads. Like micropool itself, zero is treated as one.
    #[must_use]
    pub fn split_by(mut self, chunks: usize) -> Self {
        self.split = MicropoolSplit::By(chunks);
        self
    }

    /// Uses micropool's native `split_by_threads` strategy.
    ///
    /// This creates one work unit per worker thread, plus one for an external
    /// calling thread.
    #[must_use]
    pub fn split_by_threads(mut self) -> Self {
        self.split = MicropoolSplit::ByThreads;
        self
    }

    /// Executes this join with any micropool-compatible thread-pool strategy.
    ///
    /// This provides direct access to micropool's strategy objects:
    ///
    /// ```ignore
    /// (&storage)
    ///     .micropool_join_with_cache(&mut cache)
    ///     .with_thread_pool(micropool::split_per_item())
    ///     .for_each(process);
    /// ```
    ///
    /// For an explicit pool, pass a strategy created from that pool, such as
    /// `pool.split_by(64)`. This method replaces both the pool and split
    /// strategy previously selected on this join.
    #[must_use]
    pub fn with_thread_pool<P>(self, thread_pool: P) -> JoinMicropoolIterWithPool<'pool, J, P>
    where
        P: GenericThreadPool,
    {
        JoinMicropoolIterWithPool {
            cache: self.cache,
            index_bound: self.index_bound,
            join: self.join,
            thread_pool,
        }
    }

    /// Runs `f` through micropool's native indexed pipeline.
    ///
    /// This terminal path fills the reusable cache directly and avoids routing
    /// the common `.micropool_join().for_each(...)` form through the generic
    /// adaptor pipeline.
    pub fn for_each<F>(self, f: F)
    where
        F: Fn(J::Type) + Sync,
    {
        run_micropool_for_each(
            self.join,
            self.pool,
            self.cache,
            self.index_bound,
            self.split,
            f,
        );
    }
}

/// A micropool storage join configured with a concrete native pool strategy.
#[cfg(feature = "micropool")]
#[must_use = "iterator adaptors are lazy"]
pub struct JoinMicropoolIterWithPool<'cache, J, P> {
    cache: Option<&'cache mut ParJoinCache>,
    index_bound: Option<usize>,
    join: J,
    thread_pool: P,
}

#[cfg(feature = "micropool")]
impl<J, P> JoinMicropoolIterWithPool<'_, J, P>
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
{
    /// Bounds an unconstrained join to candidate indices in
    /// `0..index_bound`.
    ///
    /// This is equivalent to calling
    /// [`JoinMicropoolIter::with_index_bound`] before
    /// [`JoinMicropoolIter::with_thread_pool`].
    #[must_use]
    pub fn with_index_bound(mut self, index_bound: usize) -> Self {
        self.index_bound = Some(index_bound.min(HIBITSET_INDEX_COUNT));
        self
    }

    /// Runs `f` using the concrete micropool strategy supplied by
    /// [`JoinMicropoolIter::with_thread_pool`].
    pub fn for_each<F>(self, f: F)
    where
        F: Fn(J::Type) + Sync,
    {
        run_micropool_for_each_with_pool(
            self.join,
            self.cache,
            self.index_bound,
            self.thread_pool,
            f,
        );
    }
}

#[cfg(feature = "micropool")]
impl<J, P> MicropoolParallelIterator for JoinMicropoolIterWithPool<'_, J, P>
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
{
    type Item = J::Type;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        run_micropool_upper_bounded_with_pool(
            self.join,
            self.cache,
            self.index_bound,
            self.thread_pool,
            init,
            process_item,
            finalize,
            reduce,
        )
    }

    fn iter_pipeline<Output, Accum: Send>(
        self,
        accum: impl Accumulator<Self::Item, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
    ) -> Output {
        run_micropool_pipeline_with_pool(
            self.join,
            self.cache,
            self.index_bound,
            self.thread_pool,
            accum,
            reduce,
        )
    }
}

#[cfg(feature = "micropool")]
impl<J> MicropoolParallelIterator for JoinMicropoolIter<'_, J>
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
{
    type Item = J::Type;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        run_micropool_upper_bounded(
            self.join,
            self.pool,
            self.cache,
            self.index_bound,
            self.split,
            init,
            process_item,
            finalize,
            reduce,
        )
    }

    fn iter_pipeline<Output, Accum: Send>(
        self,
        accum: impl Accumulator<Self::Item, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
    ) -> Output {
        run_micropool_pipeline(
            self.join,
            self.pool,
            self.cache,
            self.index_bound,
            self.split,
            accum,
            reduce,
        )
    }
}

#[cfg(feature = "micropool")]
#[inline]
fn effective_parallelism(max_parallelism: Option<NonZeroUsize>, available: usize) -> usize {
    max_parallelism
        .map(NonZeroUsize::get)
        .unwrap_or(available)
        .min(available)
        .max(1)
}

#[cfg(feature = "micropool")]
#[inline]
fn adaptive_work_units(
    input_len: usize,
    max_parallelism: usize,
    min_items_per_work_unit: NonZeroUsize,
) -> usize {
    input_len
        .div_ceil(min_items_per_work_unit.get())
        .max(1)
        .min(max_parallelism.max(1))
}

#[cfg(feature = "micropool")]
macro_rules! with_micropool_split {
    ($pool:expr, $split:expr, $input_len:expr, |$thread_pool:ident| $body:block) => {{
        let input_len = $input_len;
        match ($pool, $split) {
            (
                Some(pool),
                MicropoolSplit::Adaptive {
                    max_parallelism,
                    min_items_per_work_unit,
                },
            ) => {
                let available = pool.num_threads().saturating_add(1);
                let max_parallelism = effective_parallelism(max_parallelism, available);
                let work_units =
                    adaptive_work_units(input_len, max_parallelism, min_items_per_work_unit);
                let $thread_pool = pool.split_by(work_units);
                $body
            }
            (
                None,
                MicropoolSplit::Adaptive {
                    max_parallelism,
                    min_items_per_work_unit,
                },
            ) => {
                let available = micropool::num_threads().saturating_add(1);
                let max_parallelism = effective_parallelism(max_parallelism, available);
                let work_units =
                    adaptive_work_units(input_len, max_parallelism, min_items_per_work_unit);
                let $thread_pool = micropool::split_by(work_units);
                $body
            }
            (Some(pool), MicropoolSplit::PerItem) => {
                let $thread_pool = pool.split_per_item();
                $body
            }
            (None, MicropoolSplit::PerItem) => {
                let $thread_pool = micropool::split_per_item();
                $body
            }
            (Some(pool), MicropoolSplit::Per(chunk_size)) => {
                let $thread_pool = pool.split_per(chunk_size);
                $body
            }
            (None, MicropoolSplit::Per(chunk_size)) => {
                let $thread_pool = micropool::split_per(chunk_size);
                $body
            }
            (Some(pool), MicropoolSplit::By(chunks)) => {
                let $thread_pool = pool.split_by(chunks);
                $body
            }
            (None, MicropoolSplit::By(chunks)) => {
                let $thread_pool = micropool::split_by(chunks);
                $body
            }
            (Some(pool), MicropoolSplit::ByThreads) => {
                let $thread_pool = pool.split_by_threads();
                $body
            }
            (None, MicropoolSplit::ByThreads) => {
                let $thread_pool = micropool::split_by_threads();
                $body
            }
        }
    }};
}

#[cfg(feature = "micropool")]
#[inline]
fn unconstrained_item<J>(keys: &J::Mask, values: &J::Value, raw_index: usize) -> Option<J::Type>
where
    J: ParJoin,
{
    debug_assert!(raw_index < HIBITSET_INDEX_COUNT);
    // `HIBITSET_INDEX_COUNT` is at most 2^24 on supported targets, so this
    // narrowing conversion always fits in hibitset's `u32` index.
    let index = raw_index as Index;
    if keys.contains(index) {
        // SAFETY: The mask was checked for `index`, and Micropool's range
        // source visits each raw index at most once. Therefore concurrent
        // calls use distinct indices as required by `ParJoin`.
        Some(unsafe { J::get(values, index) })
    } else {
        None
    }
}

#[cfg(feature = "micropool")]
fn run_unconstrained_for_each<J, P, F>(
    keys: &J::Mask,
    values: &J::Value,
    index_bound: usize,
    pool: P,
    f: F,
) where
    J: ParJoin,
    J::Mask: Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    F: Fn(J::Type) + Sync,
{
    (0..index_bound)
        .into_par_iter()
        .filter_map(|index| unconstrained_item::<J>(keys, values, index))
        .with_thread_pool(pool)
        .for_each(f);
}

#[cfg(feature = "micropool")]
fn run_unconstrained_upper_bounded<J, P, Output, Accum>(
    keys: &J::Mask,
    values: &J::Value,
    index_bound: usize,
    pool: P,
    init: impl Fn() -> Accum + Sync,
    process_item: impl Fn(Accum, usize, J::Type) -> ControlFlow<Accum, Accum> + Sync,
    finalize: impl Fn(Accum) -> Output + Sync,
    reduce: impl Fn(Output, Output) -> Output,
) -> Output
where
    J: ParJoin,
    J::Mask: Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Output: Send,
{
    (0..index_bound)
        .into_par_iter()
        .filter_map(|index| unconstrained_item::<J>(keys, values, index))
        .with_thread_pool(pool)
        .upper_bounded_pipeline(init, process_item, finalize, reduce)
}

#[cfg(feature = "micropool")]
fn run_unconstrained_pipeline<J, P, Output, Accum>(
    keys: &J::Mask,
    values: &J::Value,
    index_bound: usize,
    pool: P,
    accum: impl Accumulator<J::Type, Accum> + Sync,
    reduce: impl ExactSizeAccumulator<Accum, Output>,
) -> Output
where
    J: ParJoin,
    J::Mask: Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Accum: Send,
{
    (0..index_bound)
        .into_par_iter()
        .filter_map(|index| unconstrained_item::<J>(keys, values, index))
        .with_thread_pool(pool)
        .iter_pipeline(accum, reduce)
}

#[cfg(feature = "micropool")]
fn run_micropool_for_each<J, F>(
    join: J,
    pool: Option<&micropool::ThreadPool>,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    split: MicropoolSplit,
    f: F,
) where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    F: Fn(J::Type) + Sync,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return with_micropool_split!(pool, split, index_bound, |thread_pool| {
            run_unconstrained_for_each::<J, _, _>(&keys, &values, index_bound, thread_pool, f)
        });
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    let indices = cache.indices.as_slice();
    with_micropool_split!(pool, split, indices.len(), |thread_pool| {
        run_indexed_for_each::<J, _, _>(indices, &values, thread_pool, f)
    })
}

#[cfg(feature = "micropool")]
fn run_micropool_for_each_with_pool<J, P, F>(
    join: J,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    thread_pool: P,
    f: F,
) where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    F: Fn(J::Type) + Sync,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return run_unconstrained_for_each::<J, _, _>(&keys, &values, index_bound, thread_pool, f);
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    run_indexed_for_each::<J, _, _>(cache.indices.as_slice(), &values, thread_pool, f);
}

#[cfg(feature = "micropool")]
fn run_indexed_for_each<J, P, F>(indices: &[Index], values: &J::Value, pool: P, f: F)
where
    J: ParJoin,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    F: Fn(J::Type) + Sync,
{
    indices
        .par_iter()
        .with_thread_pool(pool)
        .map(|&index| {
            // SAFETY: Every index came from the mask returned alongside
            // `values`, and every source position is processed at most once.
            unsafe { J::get(values, index) }
        })
        .for_each(f);
}

#[cfg(feature = "micropool")]
fn run_micropool_upper_bounded<J, Output, Accum>(
    join: J,
    pool: Option<&micropool::ThreadPool>,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    split: MicropoolSplit,
    init: impl Fn() -> Accum + Sync,
    process_item: impl Fn(Accum, usize, J::Type) -> ControlFlow<Accum, Accum> + Sync,
    finalize: impl Fn(Accum) -> Output + Sync,
    reduce: impl Fn(Output, Output) -> Output,
) -> Output
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    Output: Send,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return with_micropool_split!(pool, split, index_bound, |thread_pool| {
            run_unconstrained_upper_bounded::<J, _, _, _>(
                &keys,
                &values,
                index_bound,
                thread_pool,
                init,
                process_item,
                finalize,
                reduce,
            )
        });
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    let indices = cache.indices.as_slice();
    with_micropool_split!(pool, split, indices.len(), |thread_pool| {
        run_indexed_upper_bounded::<J, _, _, _>(
            indices,
            &values,
            thread_pool,
            init,
            process_item,
            finalize,
            reduce,
        )
    })
}

#[cfg(feature = "micropool")]
fn run_micropool_upper_bounded_with_pool<J, P, Output, Accum>(
    join: J,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    thread_pool: P,
    init: impl Fn() -> Accum + Sync,
    process_item: impl Fn(Accum, usize, J::Type) -> ControlFlow<Accum, Accum> + Sync,
    finalize: impl Fn(Accum) -> Output + Sync,
    reduce: impl Fn(Output, Output) -> Output,
) -> Output
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Output: Send,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return run_unconstrained_upper_bounded::<J, _, _, _>(
            &keys,
            &values,
            index_bound,
            thread_pool,
            init,
            process_item,
            finalize,
            reduce,
        );
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    run_indexed_upper_bounded::<J, _, _, _>(
        cache.indices.as_slice(),
        &values,
        thread_pool,
        init,
        process_item,
        finalize,
        reduce,
    )
}

#[cfg(feature = "micropool")]
fn run_indexed_upper_bounded<J, P, Output, Accum>(
    indices: &[Index],
    values: &J::Value,
    pool: P,
    init: impl Fn() -> Accum + Sync,
    process_item: impl Fn(Accum, usize, J::Type) -> ControlFlow<Accum, Accum> + Sync,
    finalize: impl Fn(Accum) -> Output + Sync,
    reduce: impl Fn(Output, Output) -> Output,
) -> Output
where
    J: ParJoin,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Output: Send,
{
    indices
        .par_iter()
        .with_thread_pool(pool)
        .map(|&index| {
            // SAFETY: Every index came from the mask returned alongside
            // `values`, and every source position is processed at most once.
            unsafe { J::get(values, index) }
        })
        .upper_bounded_pipeline(init, process_item, finalize, reduce)
}

#[cfg(feature = "micropool")]
fn run_micropool_pipeline<J, Output, Accum>(
    join: J,
    pool: Option<&micropool::ThreadPool>,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    split: MicropoolSplit,
    accum: impl Accumulator<J::Type, Accum> + Sync,
    reduce: impl ExactSizeAccumulator<Accum, Output>,
) -> Output
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    Accum: Send,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return with_micropool_split!(pool, split, index_bound, |thread_pool| {
            run_unconstrained_pipeline::<J, _, _, _>(
                &keys,
                &values,
                index_bound,
                thread_pool,
                accum,
                reduce,
            )
        });
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    let indices = cache.indices.as_slice();
    with_micropool_split!(pool, split, indices.len(), |thread_pool| {
        run_indexed_pipeline::<J, _, _, _>(indices, &values, thread_pool, accum, reduce)
    })
}

#[cfg(feature = "micropool")]
fn run_micropool_pipeline_with_pool<J, P, Output, Accum>(
    join: J,
    cache: Option<&mut ParJoinCache>,
    index_bound: Option<usize>,
    thread_pool: P,
    accum: impl Accumulator<J::Type, Accum> + Sync,
    reduce: impl ExactSizeAccumulator<Accum, Output>,
) -> Output
where
    J: ParJoin,
    J::Mask: Send + Sync,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Accum: Send,
{
    // SAFETY: `keys` and `values` remain paired and are not exposed. Both
    // execution paths call `J::get` only after checking mask membership and
    // at most once for each distinct index.
    let (keys, values) = unsafe { join.open() };
    if J::is_unconstrained() {
        let index_bound = unconstrained_index_bound::<J>(index_bound);
        return run_unconstrained_pipeline::<J, _, _, _>(
            &keys,
            &values,
            index_bound,
            thread_pool,
            accum,
            reduce,
        );
    }

    let mut local_cache = ParJoinCache::new();
    let cache = match cache {
        Some(cache) => cache,
        None => &mut local_cache,
    };
    cache.refill(&keys);

    run_indexed_pipeline::<J, _, _, _>(
        cache.indices.as_slice(),
        &values,
        thread_pool,
        accum,
        reduce,
    )
}

#[cfg(feature = "micropool")]
fn run_indexed_pipeline<J, P, Output, Accum>(
    indices: &[Index],
    values: &J::Value,
    pool: P,
    accum: impl Accumulator<J::Type, Accum> + Sync,
    reduce: impl ExactSizeAccumulator<Accum, Output>,
) -> Output
where
    J: ParJoin,
    J::Value: Send + Sync,
    P: GenericThreadPool,
    Accum: Send,
{
    indices
        .par_iter()
        .with_thread_pool(pool)
        .map(|&index| {
            // SAFETY: Every index came from the mask returned alongside
            // `values`, and every source position is processed at most once.
            unsafe { J::get(values, index) }
        })
        .iter_pipeline(accum, reduce)
}

#[cfg(all(test, feature = "micropool"))]
mod tests {
    use super::*;
    use crate::join::LendJoin;
    use std::sync::atomic::AtomicUsize;

    struct EmptyUnconstrainedJoin;

    // SAFETY: The returned mask is empty, so `get` is never callable after the
    // required mask check.
    unsafe impl ParJoin for EmptyUnconstrainedJoin {
        type Mask = hibitset::BitSet;
        type Type = ();
        type Value = ();

        unsafe fn open(self) -> (Self::Mask, Self::Value) {
            (hibitset::BitSet::new(), ())
        }

        unsafe fn get(_: &Self::Value, _: Index) -> Self::Type {
            unreachable!("an empty mask must never yield an item")
        }

        fn is_unconstrained() -> bool {
            true
        }
    }

    struct SparseUnconstrainedJoin {
        mask: hibitset::BitSet,
    }

    // SAFETY: `get` accepts every index present in `mask`, and the mask
    // iterator does not repeat indices.
    unsafe impl ParJoin for SparseUnconstrainedJoin {
        type Mask = hibitset::BitSet;
        type Type = Index;
        type Value = ();

        unsafe fn open(self) -> (Self::Mask, Self::Value) {
            (self.mask, ())
        }

        unsafe fn get(_: &Self::Value, index: Index) -> Self::Type {
            index
        }

        fn is_unconstrained() -> bool {
            true
        }
    }

    #[test]
    fn default_work_units_keep_every_item_stealable() {
        let default_grain = DEFAULT_MIN_ITEMS_PER_WORK_UNIT;

        assert_eq!(adaptive_work_units(10, 32, default_grain), 10);
        assert_eq!(adaptive_work_units(32, 32, default_grain), 32);
        assert_eq!(adaptive_work_units(33, 32, default_grain), 32);
        assert_eq!(adaptive_work_units(100, 32, default_grain), 32);
        assert_eq!(adaptive_work_units(100, 3, default_grain), 3);

        let grain = NonZeroUsize::new(32).unwrap();
        assert_eq!(adaptive_work_units(10, 32, grain), 1);
        assert_eq!(adaptive_work_units(100, 32, grain), 4);
    }

    #[test]
    fn cache_keeps_inline_and_spilled_storage() {
        let mut mask = hibitset::BitSet::new();
        let mut cache = ParJoinCache::new();

        for index in 0..10 {
            mask.add(index);
        }
        cache.refill(&mask);
        assert!(!cache.indices.spilled());

        for index in 10..100 {
            mask.add(index);
        }
        cache.refill(&mask);
        assert!(cache.indices.spilled());
        let allocation = cache.indices.as_ptr();
        let capacity = cache.indices.capacity();

        cache.refill(&mask);
        assert_eq!(cache.indices.as_ptr(), allocation);
        assert_eq!(cache.indices.capacity(), capacity);
    }

    #[test]
    fn unconstrained_join_bypasses_index_cache_for_adaptor_pipelines() {
        let mut seeded_mask = hibitset::BitSet::new();
        let mut cache = ParJoinCache::new();
        for index in 0..100 {
            seeded_mask.add(index);
        }
        cache.refill(&seeded_mask);

        let allocation = cache.indices.as_ptr();
        let capacity = cache.indices.capacity();
        let len = cache.indices.len();
        let empty_mask = hibitset::BitSet::new();

        let first_missing = (&empty_mask)
            .maybe()
            .micropool_join_with_cache(&mut cache)
            .with_index_bound(8)
            .find_first(Option::is_none);
        assert_eq!(first_missing, Some(None));

        let any_missing = (&empty_mask)
            .maybe()
            .micropool_join_with_cache(&mut cache)
            .with_index_bound(8)
            .find_any(Option::is_none);
        assert_eq!(any_missing, Some(None));

        assert_eq!(cache.indices.as_ptr(), allocation);
        assert_eq!(cache.indices.capacity(), capacity);
        assert_eq!(cache.indices.len(), len);
    }

    #[test]
    fn unconstrained_for_each_bypasses_index_cache() {
        let mut cache = ParJoinCache::new();

        EmptyUnconstrainedJoin
            .micropool_join_with_cache(&mut cache)
            .with_index_bound(8)
            .for_each(|()| unreachable!("an empty mask must not execute the callback"));

        assert!(cache.indices.is_empty());
        assert!(!cache.indices.spilled());
    }

    #[test]
    fn unconstrained_index_bound_is_exclusive_for_adaptor_pipelines() {
        let mut mask = hibitset::BitSet::new();
        mask.add(4);
        mask.add(5);

        let first = SparseUnconstrainedJoin { mask }
            .micropool_join()
            .with_index_bound(5)
            .find_first(|_| true);

        assert_eq!(first, Some(4));

        let mut mask = hibitset::BitSet::new();
        mask.add(5);
        let outside_bound = SparseUnconstrainedJoin { mask }
            .micropool_join()
            .with_index_bound(5)
            .find_any(|_| true);

        assert_eq!(outside_bound, None);
    }

    #[test]
    fn unconstrained_index_bound_reaches_for_each_and_concrete_pool() {
        let mut mask = hibitset::BitSet::new();
        mask.add(1);
        mask.add(3);
        mask.add(5);

        let count = AtomicUsize::new(0);
        SparseUnconstrainedJoin { mask }
            .micropool_join()
            .with_index_bound(4)
            .with_thread_pool(micropool::split_by_threads())
            .for_each(|_| {
                count.fetch_add(1, Ordering::Relaxed);
            });

        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn unconstrained_index_bound_is_clamped_to_hibitset_capacity() {
        assert_eq!(
            unconstrained_index_bound::<EmptyUnconstrainedJoin>(Some(usize::MAX)),
            HIBITSET_INDEX_COUNT
        );
    }
}
