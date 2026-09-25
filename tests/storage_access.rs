use specs::{prelude::*, world::EntitiesRes};

#[derive(Debug, PartialEq)]
struct Position(i32);
impl Component for Position { type Storage = VecStorage<Self>; }

#[derive(Debug, PartialEq)]
struct Velocity(i32);
impl Component for Velocity { type Storage = VecStorage<Self>; }

#[test]
fn storage_queries_and_try_access_preserve_exclusive_writes() {
    let mut world = World::new();
    assert_eq!(world.is_storage_locked::<Position>(), None);
    assert!(matches!(world.try_write_component::<Position>(), Err(AccessError::Missing)));
    assert_eq!(world.is_resource_locked::<EntitiesRes>(), Some(false));
    world.register::<Position>();
    let entity = world.create_entity().with(Position(1)).build();
    assert_eq!(world.is_component_locked::<Position>(), Some(false));
    let read = world.read_storage::<Position>();
    let second = world.try_read_storage::<Position>().unwrap();
    assert_eq!(world.is_storage_locked::<Position>(), Some(true));
    assert_eq!(world.is_storage_write_locked::<Position>(), Some(false));
    assert_eq!(world.is_storage_owned_by_current_thread::<Position>(), Some(true));
    assert!(matches!(world.try_write_storage::<Position>(), Err(AccessError::ReentrantConflict)));
    drop(read);
    drop(second);
    let mut write = world.try_write_storage::<Position>().unwrap();
    assert_eq!(world.is_component_write_locked::<Position>(), Some(true));
    assert!(matches!(world.try_read_storage::<Position>(), Err(AccessError::ReentrantConflict)));
    write.get_mut(entity).unwrap().0 = 2;
    drop(write);
    assert_eq!(world.read_storage::<Position>().get(entity), Some(&Position(2)));
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
    assert_eq!(world.is_storage_owned_by_current_thread::<Position>(), Some(false));
}

#[test]
fn failure_releases_intermediate_entity_borrows() {
    let mut world = World::new();
    world.register::<Position>();
    // Hold only the component resource, so a leaked intermediate entity borrow
    // would remain observable after a failed storage acquisition.
    let write = world.fetch_mut::<specs::storage::MaskedStorage<Position>>();
    assert!(matches!(world.try_read_component::<Position>(), Err(AccessError::ReentrantConflict)));
    assert!(matches!(world.try_write_component::<Position>(), Err(AccessError::ReentrantConflict)));
    assert_eq!(world.is_resource_locked::<EntitiesRes>(), Some(false));
    assert!(world.try_fetch_mut_result::<EntitiesRes>().is_ok());
    drop(write);
    assert!(world.try_write_storage::<Position>().is_ok());
}

#[test]
fn entity_contention_blocks_the_whole_storage() {
    let mut world = World::new();
    world.register::<Position>();
    let entities = world.entities_mut();
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
    assert!(matches!(world.try_read_component::<Position>(), Err(AccessError::ReentrantConflict)));
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
    drop(entities);
    assert!(world.try_read_storage::<Position>().is_ok());
}

#[test]
fn another_thread_sees_busy_instead_of_a_panic() {
    let mut world = World::new();
    world.register::<Position>();
    let read = world.read_component::<Position>();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            assert!(matches!(world.try_write_component::<Position>(), Err(AccessError::Busy)));
            // Concurrent reads remain fine.
            assert!(world.try_read_component::<Position>().is_ok());
        }).join().unwrap();
    });
    drop(read);
}

#[test]
fn typed_storage_bundle_shares_entities_and_rejects_aliases() {
    let mut world = World::new();
    world.register::<Position>();
    world.register::<Velocity>();
    let entity = world.create_entity().with(Position(1)).with(Velocity(2)).build();
    let (mut positions, velocities) = world.try_storages::<(WriteStorage<Position>, ReadStorage<Velocity>)>().unwrap();
    positions.get_mut(entity).unwrap().0 += velocities.get(entity).unwrap().0;
    assert!(matches!(world.try_write_component::<Position>(), Err(AccessError::ReentrantConflict)));
    drop((positions, velocities));
    assert_eq!(world.is_resource_locked::<EntitiesRes>(), Some(false));
    assert_eq!(world.read_component::<Position>().get(entity), Some(&Position(3)));
    assert!(matches!(world.try_storages::<(ReadStorage<Position>, WriteStorage<Position>)>(), Err(AccessError::ConflictingRequest)));
}

#[test]
fn storage_tuple_failure_releases_every_lock() {
    let mut world = World::new();
    world.register::<Position>();
    assert!(matches!(world.try_storages::<(WriteStorage<Position>, ReadStorage<Velocity>)>(), Err(AccessError::Missing)));
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
    assert_eq!(world.is_resource_locked::<EntitiesRes>(), Some(false));
    world.register::<Velocity>();
    let first = world.try_storages::<(WriteStorage<Position>, ReadStorage<Velocity>)>().unwrap();
    assert!(matches!(world.try_storages::<(ReadStorage<Position>, WriteStorage<Velocity>)>(), Err(AccessError::ReentrantConflict)));
    drop(first);
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
    assert_eq!(world.is_storage_locked::<Velocity>(), Some(false));
    assert_eq!(world.is_resource_locked::<EntitiesRes>(), Some(false));
}

#[cfg(any(feature = "parallel", feature = "micropool"))]
#[test]
fn parallel_joins_keep_guard_ownership_on_the_caller() {
    let mut world = World::new();
    world.register::<Position>();
    for _ in 0..1000 { world.create_entity().with(Position(0)).build(); }
    let mut positions = world.write_component::<Position>();
    let mut expected = 0;
    #[cfg(feature = "parallel")]
    {
        use rayon::iter::ParallelIterator;
        (&mut positions).par_join().for_each(|position| position.0 += 1);
        expected += 1;
    }
    #[cfg(feature = "micropool")]
    {
        use micropool::iter::ParallelIteratorExt;
        let sum = (&mut positions).micropool_join().map(|position| { position.0 += 1; 1usize }).sum::<usize>();
        assert_eq!(sum, 1000);
        expected += 1;
    }
    assert!((&positions).join().all(|position| position.0 == expected));
    assert_eq!(world.is_storage_owned_by_current_thread::<Position>(), Some(true));
    drop(positions);
    assert_eq!(world.is_storage_locked::<Position>(), Some(false));
}
