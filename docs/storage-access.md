# Storage 借用：fail-fast、非阻塞获取与访问审计

shred 的 `ResourceCell` 是一个原子借用字（与 `atomic_refcell` 同构：最高位=写者，其余=读者计数）。
**借用永不等待**：冲突时普通接口 panic，`try_*` 接口返回 `AccessError`。

为什么不等待：本引擎的 ECS 靠 dispatcher 静态调度保证系统间不冲突，冲突只来自
（a）同线程重入——等待即死锁；（b）调度器看不见的访问（如 `AnimationDirectAccessWorld`）——
等待会把必现 panic 变成随机的写入顺序。两种情况都应当尽早暴露，而不是隐藏。
需要「稍后再做」时，投递到 `SCOPE_ANY_IDLE_ECS` 之类的确定时点，而不是阻塞。

## 错误分类

| `AccessError` | 含义 | 建议处理 |
| --- | --- | --- |
| `Missing` | 资源/组件未注册 | — |
| `Busy` | **其他线程**持有不兼容借用 | 跳过或延后到空闲时点 |
| `ReentrantConflict` | **本线程**已持有不兼容借用（精确） | 这是调用结构的 bug：把已有 Storage 传下去 |
| `ConflictingRequest` | 整组请求里同一资源出现写+其他 | 修请求 |
| `AccessMismatch` | 从整组提取的模式与请求不符 | 修 `FetchBundle` 实现 |

`Busy` 与 `ReentrantConflict` 的区分靠线程局部的「本线程持有集」，每次借用一次 TLS 查找，
无共享状态。这也是活动 guard 为 `!Send` 的原因（`T: Sync` 时仍为 `Sync`）：
guard 在别的线程释放会弄乱持有集。把 `&guard` 借给 worker，不要按值移动。

## 接口

```rust,ignore
use specs::prelude::*;

match world.try_write_storage::<Position>() {
    Ok(mut positions) => { /* ... */ }
    Err(AccessError::Busy) => { /* 其他线程占用：跳过或延后 */ }
    Err(AccessError::ReentrantConflict) => { /* 本线程已持有：调用结构有误 */ }
    Err(error) => { /* Missing 等 */ }
}

// 整组、全有或全无；失败时释放本次拿到的全部 guard，EntitiesRes 读合并。
let (mut positions, velocities) =
    world.try_storages::<(WriteStorage<'_, Position>, ReadStorage<'_, Velocity>)>()?;
```

- Storage：`try_read_storage` / `try_write_storage`（及 `*_component` 别名）、`try_storages::<B>()`。
- 资源：`try_fetch_result` / `try_fetch_mut_result`（及 `_by_id`）、`try_fetch_bundle::<B>()`、
  动态 `try_lock_resources(&[AccessRequest])` + `take_read_by_id` / `take_write_by_id`。
- 整组支持 `Fetch`/`FetchMut`、`Read`/`Write`（含 Expect）、`ReadStorage`/`WriteStorage`、
  1～16 元组及嵌套，或自定义 `FetchBundle`。整组里任一资源的重入冲突优先于 `Busy` 报告。
- 旧接口 `read_storage`/`write_storage`/`fetch`/`fetch_mut`/`SystemData::fetch` 语义不变：冲突即 panic。
  panic 信息形如 ``cannot borrow `T` mutably: already borrowed by this thread (reentrant conflict); `T` is read-borrowed 1x``，
  开启 `access-trace` 时附带每个持有者的线程、调用点和持有时长。
- 查询：`is_storage_locked` / `is_storage_write_locked`（瞬时观测）、
  `is_storage_owned_by_current_thread`（精确）、`storage_access_snapshot`；资源层同名 `is_resource_*`。
  组件查询只针对 `MaskedStorage<T>`，不含隐式读取的 `EntitiesRes`。

## `access-trace` feature

shred `access-trace`（specs 同名转发，引擎 `ecs-access-trace`，随 `editor`/`dev` 开启）：

1. **持有者记录**：每个借用登记线程、`#[track_caller]` 调用点、获取时刻；进入冲突 panic 与
   `AccessSnapshot.holders`，并计数 `reads/writes/busy/reentrant`。代价：每次借用一次互斥锁 + 一次时钟读取。
2. **未声明访问审计**（`shred::access_audit`）：`DispatcherBuilder` 在构建时把并行阶段里的每个系统
   （含 batch）包一层，缓存其排序后的 reads/writes；运行时在本线程压入这组声明，期间本线程上的任何借用
   若不在声明内（写必须声明为写），按 `(系统, 资源, 模式)` 去重后报告一次。引擎把报告接到
   `log::warn!(target: "ecs_access_audit")`。报告是确定性的：第一次越权借用就报，不依赖线程碰巧相撞。
   thread-local 系统与手动 `run_now` 不审计。
   - 已知不会并发的越权访问可用 `let _s = shred::access_audit::suspend();` 豁免。
   - 系统把工作分发到其他 worker 时，先 `let scope = access_audit::current_scope();`，在 worker 闭包里
     `let _a = scope.enter();`，worker 上的借用就按该系统的声明审计（`AnimationApplySystem` 即如此）。
   - 局限：未 `enter` 的 worker 任务不归属该系统；work-stealing 线程执行别的系统的并行 Join 闭包时，
     会归属到该线程上被阻塞的系统。

未开启时上述全部为空操作，`AccessSnapshot.traced == false`，只有借用状态（写者/读者数）。

## 并行 Join

Rayon `par_join` 与 micropool 在调用线程上打开数据视图，只把满足 `Send/Sync` 的视图交给 worker，
活动 guard 留在原线程。直接把 guard 按值捕获进线程闭包无法编译；改为借引用。

## 兼容性

- `shred::cell` 仍导出 `atomic_refcell` 类型；`try_fetch_internal` 返回 `ResourceCell`，
  `MetaIter` 元素为 `ResourceRef` / `ResourceRefMut`。
- 已移除：限时等待接口（`*_for` / `*_until` / `lock_resources_until`）、`AccessError::Timeout`、
  `WaitInfo`、`Transfer`/`Transferable`，以及 shred 对 `parking_lot`（`send_guard`）的依赖。

## 引擎侧接入约定（ClayEngine）

按**入口**决定借不到时怎么办，而不是逐个调用点决定：

| 入口 | 借用方式 | 借不到时 |
| --- | --- | --- |
| 系统 | SystemData 声明 | 不会发生；发生即由审计报告 |
| JNI 入口、被 Java 直接调用的工具函数 | `framework::ecs::access::WorldAccessExt` 的 `*_or_log` / `storages_or_log` | 按 `(调用点, 类型, 错误)` 去重报一次 `error!`（附持有者快照），返回 `None` 由调用方降级 |
| 编辑器、加载 | 普通接口 | panic，暴露调用结构 bug |

- `WorldAccessExt` 另有 `try_fetch_or_log`（可选资源：未注册静默 `None`，冲突报错）与
  `system_data_or_log::<T>()`（`#[derive(SystemData)]` 没有 try 版本：按 `reads()`/`writes()` 逐个预检，
  本线程重入判定精确；预检与 `fetch` 之间被别的线程抢占仍会 panic）。
- **回调出去前先释放借用。** Java 构造函数、`onCreated`、信号槽位都可能再借同一个 Storage。
  `ecs_component.rs` 的 `get_component` / `upsert_component` 只在借用释放后才构造 Java 对象。
- **`EcsBorrowScope`**（`scene/guard.rs`）：持有借用期间把 ECS 标为 busy，离开时回到空闲就地排空
  `SCOPE_ANY_IDLE_ECS`。必须先于 Storage guard 声明（逆序析构 = 先释放借用再排空）。
- **立即版 Java 信号**（`java_signal` / `java_signal_custom`）：ECS 空闲且本线程不持有任何借用
  （`shred::current_thread_holds_borrows()`）才就地派发，否则把对象升为全局引用延后到 `SCOPE_ANY_IDLE_ECS`。
- **排空 `SCOPE_ANY_IDLE_ECS`** 统一走 `guard::drain_idle_tasks()`：本线程仍持有借用时不排空。
- `ECSJavaFields` / `ecs_java_methods` 生成的访问器：组件不可用时报一次错并返回默认值
  （原语默认值 / `None` / null），`&self` 方法只借读；setter 参数在借用前转换，失败报错返回 `false`。
- `JavaComponentContainerSystem` 的并行实例被前后屏障隔离成独占阶段，Java 代码的访问无法静态声明，
  在其中用 `access_audit::suspend()` 豁免审计。
- **两阶段回调**：`UIRenderer::extract_submitables` 在持有 `UIRenderSystemData` 时只收集
  `(GameObject, 视图快照)`，调用方释放后再 `submit_collected` 回调 Java `onSubmit`。
- saveload 的 `after_clone` 持有 `EntitySaveSystemData` 调 Java `newFromData` 是允许的：它只做反序列化，
  访问组件的逻辑在 `onCreated`，由 `JavaPendingCallOnCreate` 延后到借用释放后。
- 仍刻意保留普通接口（冲突 panic）的：编辑器绘制代码、`UIRenderExternal` 渲染回放（持有整组 UI 数据录制命令）。
- 新建 GameObject 挂 `GameObjectReference`、发 `ParentChangedEvent` 借不到时**延后补做**（`SCOPE_ANY_IDLE_ECS`）而非丢弃：丢了会留下无引用实体 / 不重算的 UI 变换。
- `UIInput` 公共入口在 ECS 忙**或本线程持有借用**时整次延后；内部逐步提交代码仍用普通接口，由入口保证安全。
- `Prefab::save_*` 保留 panic：借不到时返回空 Prefab 会被调用方写盘覆盖原文件。
- 编辑器面板不逐处改 `*_or_log`：`TreeBehavior::pane_ui` 以面板为单位 `catch_unwind`，出错的面板暂停并显示错误与「重试」，其余面板照常；借用守卫在展开中正常释放。
- 系统级 panic 守卫 `shred::panic_guard`：宿主装 handler 后每个系统单独 `catch_unwind`（Resume/Skip），`SystemPanic.run` 与 `current_run()` 让 panic hook 把记录对上这次运行。引擎按 `editor` feature 安装（当前所有 DLL 构建都带），策略见 `src/framework/ecs/system_panic.rs`：一律 Skip，只有「本线程在 ez_jni 帧内且 panic 就发生在本线程这次运行里」的 Java 异常 payload 才 Resume 交还 JNI 入口；日志 + Sentry（`ecs-system-panic`）按系统限频（首次立即、之后 60s 至多一次，附压下/累计次数），守卫运行内的 panic hook 只记位置、不调默认/Sentry hook；其余 panic（含系统扇出到并行 worker 的）按 panic 位置限频后才交给默认/Sentry hook。代价：守卫内被系统自己 `catch_unwind` 吞掉的 panic 也被静音，吞的地方用 `take_caught_panic_location()` 补位置。`*_or_log` 的首次跳过报 `ecs-access-skipped`。
- 局部引用帧：jni 0.21 的 `with_local_frame*` 在闭包 panic 时不弹帧，panic 一旦被捕获（系统守卫、`catch_unwind`）就失衡。一律用 `crate::bridge::java::ext::local_frame::LocalFrameExt::with_guarded_local_frame*`（panic 时弹帧后原样 `resume_unwind`），`clippy.toml` 禁用原版。
