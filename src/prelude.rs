//! Prelude module
//!
//! Contains all of the most common traits, structures,

pub use crate::join::Join;
#[nougat::gat(Type)]
pub use crate::join::LendJoin;
#[cfg(any(feature = "parallel", feature = "micropool"))]
pub use crate::join::ParJoin;
pub use hibitset::BitSet;
pub use shred::{
    Accessor, Dispatcher, DispatcherBuilder, Read, ReadExpect, Resource, ResourceId, RunNow,
    StaticAccessor, System, SystemData, World, Write, WriteExpect,
};
pub use shrev::ReaderId;

#[cfg(feature = "micropool")]
pub use micropool::iter::ParallelIteratorExt as MicropoolParallelIterator;
#[cfg(feature = "micropool")]
pub use crate::join::ParJoinCache;
#[cfg(feature = "parallel")]
pub use rayon::iter::ParallelIterator;
#[cfg(any(feature = "parallel", feature = "micropool"))]
pub use shred::AsyncDispatcher;

pub use crate::{
    changeset::ChangeSet,
    storage::{
        ComponentEvent, DefaultVecStorage, DenseVecStorage, FlaggedStorage, HashMapStorage,
        NullStorage, ReadStorage, Storage, Tracked, VecStorage, WriteStorage,
    },
    world::{Builder, Component, Entities, Entity, EntityBuilder, LazyUpdate, WorldExt},
};
