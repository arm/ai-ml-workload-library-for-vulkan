.. _workload-library-architecture:

Workload Library architecture
=============================

The Workload Library separates workload descriptions, compiled pipelines, and
binding-dependent execution state. The application supplies its data and controls
allocation and submission around that state. The library accepts VGF data or
standalone workload descriptions; translating ML models into those inputs is
outside its scope.

.. uml:: puml/execution-ownership.puml
   :align: center
   :alt: Two sessions retain shared compiled state while each owns separate session state. A prepared execution borrows its session and external resources and owns descriptors and intermediate resources. The optional CompiledExecution wrapper also retains the shared state.
   :caption: Shared compiled objects and separately owned execution state.

Terms and responsibilities
--------------------------

``Workload``
   Resource, module, constant, and ordered executable metadata. A module provides
   code; an executable describes one compute or data-graph dispatch using it.
``Context``
   The owned or borrowed Vulkan® instance, device, and queue used by the library.
``CompiledExecution``
   An optional wrapper retaining shared compiled pipelines and layouts for an
   exact workload and context. It is separate from a Vulkan® pipeline cache.
``Session``
   A Workload Library session combines compiled state with its own Vulkan® data
   graph pipeline sessions, their memory, and synchronous command state.
``BindingSet``
   Application resource bindings and push data for one WL session; it is not a
   Vulkan® descriptor set.
``PreparedExecution``
   A binding snapshot with descriptors, intermediates, and a recording plan,
   created by ``Session::prepare()``. It borrows its session and external objects.
Metadata views
   Lightweight, non-owning views of workload resources, executables, and modules.
   The ``Workload`` must outlive them. ``ResourceView`` identifies a public
   resource; its ``requirements()`` view describes properties such as tensor
   shape or buffer size. A ``BindingSet`` later associates it with concrete objects.

Design choices
--------------

* **Common metadata, distinct pipeline paths:** VGF, standalone compute, and
  standalone data-graph descriptions produce a ``Workload`` and use the same
  ``Session``, ``BindingSet``, and ``PreparedExecution`` lifecycle. Pipeline creation
  depends on the executable type and its compilation input.
* **Separate compilation from preparation:** configuration establishes pipelines
  and layouts; preparation resolves concrete bindings, descriptors, and resources.
* **Share compiled state, keep session state separate:** sessions can reuse
  pipelines while retaining their own graph pipeline sessions, memory, and command
  state. A pipeline cache assists creation; shared state avoids repeating it.
* **Keep caller ownership explicit:** supplied resources and memory remain borrowed,
  with lifetime guarantees supplied by the caller. Workloads own copied metadata
  and file mappings; prepared executions own internal resources and captured push
  data. Allocation helpers return owning wrappers to the caller rather than
  retaining their allocations; see :ref:`wl-architecture-lifetimes`.
* **Leave allocation and submission choices with the application:** callers can
  allocate public resources themselves or use context helpers, and either run
  synchronously or record for their own submission. Borrowing does not remove
  external synchronization or GPU-completion obligations.

.. raw:: html

   <style>
   .wy-table-responsive table.workload-architecture-table {
       width: 100%;
   }
   .wy-table-responsive table.workload-architecture-table td,
   .wy-table-responsive table.workload-architecture-table th {
       white-space: normal;
   }
   </style>

.. contents:: On this page
   :local:
   :depth: 2

Workloads and execution stages
------------------------------

A VGF sequence segment becomes an executable; several executables can refer to
one module. Standalone compute and data-graph descriptions each construct one
executable. Resource metadata is separate from descriptor locations, and constant
payloads are separate from their references in graph executables. The public
binding identities are explained in :ref:`wl-architecture-interfaces`.

The resulting execution flow is:

#. Construct a ``Workload`` and create or wrap a ``Context``.
#. Create a ``Session``, supply any required compilation inputs and resource
   tiling choices, then call ``configure()``.
#. Create a ``BindingSet`` and bind the public resources and required push data.
#. Call ``prepare()`` to capture the bindings and create a ``PreparedExecution``.
#. Run synchronously or record into an application command buffer.

.. uml:: puml/execution-lifecycle.puml
   :align: center
   :alt: Workload and context feed session configuration, followed by resource binding and preparation. Prepared execution either submits and waits through run, or records for application submission.
   :caption: The common lifecycle, with separate submission responsibilities.

``Session::configure()`` builds or reuses pipelines and layouts, creates a
Vulkan® graph pipeline session and its memory for each graph executable, and
creates command-buffer and fence state for the synchronous path. A Workload
Library ``Session`` therefore can contain several Vulkan® graph pipeline
sessions.

:ref:`wl-architecture-preparation` explains snapshot creation;
:ref:`wl-architecture-lifetimes` gives the ownership and lifetime rules.

Workload construction and interfaces
------------------------------------

All three construction paths use an internal builder to create the same
``Workload`` representation. Construction does not create Vulkan®
pipelines or allocate execution resources.

.. uml:: puml/workload-construction.puml
   :align: center
   :alt: Alternative VGF and standalone factory calls populate an internal builder. The builder finalizes public resource and placeholder indices and returns the same Workload representation for session configuration.
   :caption: Alternative construction paths converge on one workload model.

VGF import and backing storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Workload::fromVGF()`` decodes the header and module, model-resource,
model-sequence, and constant tables through VGF Library. Module records supply
code and entry points; resource records supply kinds, requirements, roles, and
alias relationships. Sequence segments supply executable order, module
references, descriptor bindings, dispatch shapes, push-constant ranges, and
constant references. The constant table connects payloads to constant resources.
These relationships become workload metadata rather than exposing decoder
handles through the normal execution API.

File input keeps a memory mapping owned by the workload. The memory overload
borrows the caller's bytes, which must stay valid and unchanged for the workload
lifetime. Decoding copies metadata and embedded module code into library
containers, while constant payload views can still refer to the source bytes.
The import therefore combines owned metadata with borrowed payload storage.
Disabling VGF support removes the decoder dependency; calling ``fromVGF()`` in
that build reports that support is unavailable.

The builder retains resources in their internal order and constructs a separate
public index list for input/output resources. Constants and intermediates do not
become public bindings. Executables can still reference those internal resources
and constants during configuration and preparation.

Standalone descriptions
^^^^^^^^^^^^^^^^^^^^^^^

``Workload::fromComputeShader()`` takes a module implementation, entry point,
resource descriptions with descriptor locations and access modes, dispatch shape,
specialization data, push-constant size, and implicit-barrier policy. It creates
one compute executable. Public resource indices follow the description's resource
order, with one distinct resource per entry; descriptor numbers do not determine
that order, and compatible entries are not deduplicated.

``Workload::fromDataGraph()`` also creates one executable, with graph resource
metadata and constants instead of compute dispatch geometry. Its pipeline
metadata includes flags and specialization data. Graph constants
retain payload views alongside tensor metadata; standalone callers keep those
payloads valid for the workload lifetime. Sparse 2:4 constants additionally carry
the sparsity dimension used when constructing Vulkan® constant tensor descriptions.
The library passes that metadata to graph pipeline creation; it does not perform
model weight compression as part of workload construction.

Both paths accept the same ``ModuleImplementation`` representation. SPIR-V™ code
is consumed directly during configuration. GLSL and HLSL source is compiled to
SPIR-V™ by optional backends, using supplied build options and include paths;
source compilation supports compute executables only. These build capabilities
are distinct from the Vulkan® capabilities of the selected device. ``supports()`` reports
available source backends, not a complete device-capability preflight. Leaving a
module implementation unspecified defers supplying code to the session.

.. _wl-architecture-interfaces:

Public bindings and descriptor locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Public resource indices identify input/output resources; executable descriptor
locations identify how a segment uses them. VGF import assigns public indices
in model-resource-table order, skipping constants and intermediates. An input
at table index 1 after an intermediate is therefore ``workload.resource(0)``,
even if an executable uses it at set 2, binding 7. VGF model-interface IDs
are another identity: ``vgfutils::parseModelInterface()`` returns their
resource-table indices and input/output-only public indices. The application
uses the latter to choose the objects to bind.

Context and session configuration
---------------------------------

``Context::create()`` owns the Vulkan® instance, device, and queue it creates.
``Context::wrap()`` borrows application-owned Vulkan-Hpp RAII objects, including
the referenced queue wrapper. The application must keep those wrappers and their
underlying Vulkan® objects alive and stable while they are used. Wrapping does
not validate enabled device extensions or features; the application enables
those required by its workload and requested options.

Both context paths use the same session and preparation machinery. A wrapped
context can still create library-owned pipelines, descriptors, intermediate
resources, and allocations. The ownership choice for the base Vulkan® context
is independent of who supplies each public resource.

By default, a session has its own compiled state. Applications can instead pass
a ``CompiledExecution`` to several sessions for the same context and workload.
The first compilation builds the shared pipelines and layouts; subsequent
sessions reuse them while retaining separate graph sessions, graph-session
memory, and command state. This differs from a Vulkan® pipeline cache, which is
an optional caller-owned input to pipeline creation rather than an owner of the
library's shared compiled objects.


Sessions retain the shared compiled state independently of the lifetime of the
``CompiledExecution`` wrapper. Destroying or moving that wrapper does not destroy
state still retained by a session. The context and workload remain borrowed and
must outlive every user of that state.

See :ref:`wl-architecture-compilation` for shared-input compatibility and
configuration serialization, and :ref:`wl-architecture-submission` for execution
reuse and synchronization. Sharing pipelines does not make session state
independent.

Compilation inputs and executable code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The session supplies compilation inputs without modifying the original
``Workload``. Executable-level overrides include:

* **Module code:** binding an ``ExecutableView`` replaces code for that executable
  alone. On the module-backed path, resolution prefers this replacement, then
  embedded module code, then a supplied ``PlaceholderModuleView`` implementation
  for a module without code. An unresolved missing module prevents compilation.
  Two executables sharing a module can therefore use different implementations.
  Replacement shader code must match the executable's entry point, descriptor
  bindings, and effective push-constant layout. The caller guarantees compatibility;
  the library does not rewrite that interface to match arbitrary code.
* **Specialization:** an override replaces the entire specialization map; an empty
  map selects shader defaults.
* **Push-constant layout:** a compute executable can select a size before compilation.
  The override creates a range starting at zero, or removes the ranges for size
  zero. Without an override the original ranges remain. The bytes later supplied
  through ``BindingSet::bindPushConstants()`` are execution data captured by
  preparation, rather than pipeline layout inputs.

The original ``ExecutableView`` continues to describe the workload's original
metadata; session queries expose pending or compiled effective values.
See :ref:`wl-usage-push-constants` for range, size, and partial-update contracts.

.. _wl-architecture-compilation:

Compiling and reusing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration validates resource metadata and tiling compatibility before
building executable pipelines. For module-backed execution it resolves code,
compiles supported source modules if needed, creates shader modules and descriptor
set/pipeline layouts, and creates compute or graph pipelines. Graph pipeline
creation includes tensor resource descriptions and referenced graph constants.
The resulting pipelines and layouts form the compiled state shared by sessions.
Descriptor sets and binding-dependent intermediates are still created by
preparation, after public resources have been bound.

A ``CompiledExecution`` is associated with the exact ``Context`` and ``Workload``
objects supplied at construction; another wrapper around equivalent Vulkan®
handles is not interchangeable. The compiling session supplies the inputs for
all shared pipelines. Successful pipeline compilation freezes module code,
specialization, and push layout. A later session must have no pending compilation
inputs when reusing them, even if its pending values match those already compiled.

Graph tensor tiling, tiling of related alias peers, and graph alias tensor strides
must match the compiled requirements. Sessions created after compilation inherit
the fixed tiling choices; earlier sessions must select compatible choices before
reuse. Other tiling choices remain session-local. These compatibility constraints
are separate from host-side serialization: creating or configuring sessions and
changing their tiling selections while sharing compiled state must be serialized.

Compiled state is installed only after every executable pipeline has been built.
If pipeline compilation fails first, pending inputs (module code, specialization
values, push-constant layout overrides, graph binaries, and executable pipeline-cache
options) are retained so a missing implementation can be supplied or an override
corrected and compilation retried.
Pending inputs belong to that session; a new session must supply its own.
Configuration then continues beyond compilation by creating per-session graph
sessions and their memory, command pool, command buffer, and fence. Failure in
those later steps does not undo successful shared compilation. This retry
behavior therefore does not imply that every configuration failure restores an
unconfigured shared state.

.. _wl-architecture-binaries:

Graph binaries and pipeline cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a graph executable, ``Session::bindDataGraphBinary()`` selects a binary-backed
pipeline instead of shader-module compilation. A graph executable whose module is marked
``ExternalDataGraphBinary`` requires this binding; an ordinary graph executable
can also use it. The supplied executable index identifies the workload graph,
while the optional pipeline identifier selects a pipeline within the binary.
Several graph executables can reference one binary with different identifiers.
This selection does not change the workload's full ordered execution sequence.
Executables without a binary binding continue to use their module path.

A binary binding can select the binary path even when the workload contains
embedded module code. It conflicts with an explicit session module-code binding
for the same executable. A session placeholder-module binding also conflicts with
a binary targeting any executable that references that module. A binary-backed
graph cannot declare graph constants or a nonempty effective specialization map; tensor stride
overrides are also rejected on that pipeline path. Other executables retain
their independent inputs. These checks prevent mixing shader-module compilation
inputs with an already compiled graph representation.

The binary bytes are borrowed and must meet the device's binary-data alignment.
The binding copies the optional identifier, but does not copy the binary payload.
With ``ImplementationManaged`` storage, keep bytes alive through successful
configuration and through any pipeline-compilation retry, or supply a replacement
before retrying. With ``ApplicationBacked`` storage, keep them accessible and
unchanged until every pipeline created from them has been destroyed, including
pipelines retained by shared compiled state. This lifetime can exceed the
session that originally supplied the binary.

Binary-backed pipeline creation depends on device features and implementation
support. A source path in the library does not establish support on every device.

The selected Vulkan® pipeline cache is forwarded for binary-backed creation, where
Vulkan® ignores it.

For module-backed pipelines, ``SessionOptions::pipelineCache`` supplies the default; before shared
compilation, ``Session::setExecutableOptions()`` can select or explicitly disable
the cache for an executable. Keep borrowed cache wrappers at stable addresses
and their caches valid until configuration finishes using them. Successful
compilation fixes the shared pipelines; later setters are rejected, and pending
overrides prevent reuse of pipelines compiled by another session. See
:doc:`usage` for cache override and retry contracts.

Loading, validating, and saving cache files remains application
policy. Reusing that cache and retaining a ``CompiledExecution`` address different
costs: the cache assists pipeline creation, while shared compiled state avoids
creating those pipelines again for compatible sessions.

Resource preparation and ownership
----------------------------------

Resource requirements
^^^^^^^^^^^^^^^^^^^^^

A public resource describes the Vulkan® object an application must supply:

* Buffer requirements include requested size and usage.
* Tensor requirements include format, shape, byte strides, and tiling-dependent usage.
* Image requirements include format, type, extent, subresource range, and usage.

These declarations guide object creation, while the resulting Vulkan® memory
requirements determine allocation size, alignment, and compatible memory types.

``ResourceRequirementsView::byteSize()`` reports a logical span, not a Vulkan®
allocation size or a general transfer size. For buffers it is the requested
buffer size. For images it describes tightly packed texels for one mip level
and array layer, excluding row/depth pitch, tiling, padding, and subresource
offsets; unsupported logical-size calculations return zero. Tensor strides can
also make a logical span differ from a packed element count. Allocate from the
created object's memory requirements rather than this logical size.

Caller allocation
^^^^^^^^^^^^^^^^^

Applications can create and allocate public resources directly through Vulkan®,
or use ``Context::createTensor()``, ``createBuffer()``, and ``createImage()``.
These helpers return move-only allocation wrappers owning the resource and its
memory; an image allocation also owns its image view. Bindings borrow those
handles, so transferring a wrapper transfers its ownership without transferring
ownership to a binding set. A wrapped context uses the same allocation helpers.
Each helper allocates for one resource, not a complete alias group.

Select tensor/image tiling before ``configure()`` and create public objects with
the session's selected tiling and corresponding usage. Pass that tiling explicitly
to allocation helpers when their default differs. ``BindingSet`` cannot inspect
the creation tiling or all other creation properties of a caller-owned handle;
the caller guarantees they match the workload and session requirements.

.. _wl-architecture-preparation:

Binding snapshots and preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A binding set belongs to one configured session. Binding checks that resource
views belong to its workload and have the expected kind. Image binding also
checks the supplied view and subresource range. For sampled images, it additionally
checks the sampler policy: ``requiresSamplerBinding()`` requires a caller-supplied
sampler; otherwise preparation creates one from the workload sampler configuration.
Binding rejects a caller-supplied sampler when the resource does not require one.
Caller-provided image views and samplers remain borrowed.

``Session::prepare()`` requires a valid binding set from that same session and
builds a separate prepared execution:

#. Capture resource handles and binding metadata, check required backing-memory
   information, and copy the push-constant update payloads. Tensor views and
   any needed library-owned image views or samplers accompany the bound records.
#. Inspect executable bindings, allocate unaliased intermediates, and resolve
   alias groups against caller backing or a new runtime-owned allocation.
#. Allocate descriptor pools and sets for each executable and write the resolved
   tensors, buffers, images, and samplers into them. Missing resources required
   by those descriptors prevent preparation.
#. Build the internal dependency and image-barrier plan used during recording.
   Preparation does not submit GPU work.

The ``PreparedExecution`` returned by ``Session::prepare()`` is the snapshot:
it owns the captured descriptor state and runtime-created resources.
Changing or destroying the binding set leaves that snapshot unchanged; prepare
again to capture different resource handles, alias placements, or push data.
Changing the contents of an already bound GPU resource is different: preparation
does not copy those contents, so later execution accesses that same resource's
current data. Reusing a snapshot avoids repeating allocation and descriptor
setup, subject to the session serialization and resource/layout contracts.
See :ref:`wl-usage-image-layouts` for replay conditions and
:cpp:class:`mlsdk::workloadlib::BindingSet` for binding validation details.

Alias groups and backing regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A workload-declared alias group lets distinct resource objects use common backing
memory. Public resources can expose internal ``AliasPeerView`` entries whose
objects are created during preparation. Declaring a shared allocation in the
application does not by itself introduce a workload alias relationship.

For external backing, bound group members supply the same ``BoundMemoryInfo``:
a memory handle, allocation-relative base offset, and finite region size. This
region must describe actual accessible storage. It is not a buffer descriptor
range, and ``vk::WholeSize`` is not a finite region size. A peer's placement from
``BindingSet::setAliasPeerOffset()`` is relative to that base and defaults to
zero. Its effective Vulkan® binding offset is base plus peer offset, applied
once. That placement is separate from tensor byte strides and from the logical
byte size of the resource.

.. uml:: puml/alias-memory.puml
   :align: center
   :alt: Application owns a public resource and its allocation; prepared execution owns an internal peer. Both use a finite allocation region. The public resource starts at base 256 and the peer at base plus relative offset 256, or 512. Ownership and shared-backing arrows are distinct.
   :caption: Owning a peer object does not imply owning its external backing memory.

The diagram uses illustrative requirements and offsets, not device-independent
allocation sizes. Both effective binding offsets satisfy the illustrated
alignment, and each resource's full memory requirement fits inside the region.
For real allocations, query each resource's own requirements. After configuration,
``Session::aliasPeerMemoryRequirements()`` creates a temporary unbound peer using
the selected tiling and strides and reports size, alignment, memory-type bits,
and preferred/required dedicated-allocation hints. The query does not allocate
backing storage for the execution.

When preparation creates internal peers against external backing, it queries
the bound public objects and the new peers, checks effective alignment and
remaining range, rejects dedicated-allocation requirements, and intersects their
memory-type masks. It validates all new peers in a group before binding any of
them. Bound members must report the same memory handle, base, and region size.
These checks do not recover the allocation's actual memory type or verify every
property of external resources: the caller guarantees accurate metadata,
compatible creation flags/usages, and an allocation type supported by every
alias. Groups without new internal peers do not receive the same requirements
checks merely because several external resources were bound.

If an alias group has only internal resources and no caller backing, preparation
creates the peer objects, intersects their memory-type masks, and allocates
device-local memory large enough for the largest requirement. Those peers bind
at offset zero and share that allocation. Explicit peer offsets require
external backing; a required dedicated allocation cannot serve this shared path.
With external backing, the library owns the newly created peer objects and
borrows their memory. Missing public alias bindings can reuse an already bound object
of the same kind; preparation does not manufacture a missing public object of
a different kind.

Image/tensor aliases also require compatible layouts and formats. Storage-buffer
aliases and explicit tensor strides require linear tiling; image-aliasing tensor
usage can require optimal tiling. Conflicting requirements are rejected rather
than converted. Image/tensor component widths and channel counts must match,
tensors use scalar formats with the required rank and singleton outer dimensions,
and compressed, multiplane, depth/stencil image formats are not accepted for
this pairing. Library-created intermediate images and image allocation helpers
also require supported simple formats and concrete extents.

Linear image/tensor aliasing requires a caller-owned image. Query its actual
subresource layout first, then set byte strides on the linear intermediate tensor
peer with ``Session::setTensorStrides()`` before configuration. Use row/depth
pitches for strides and the subresource offset for peer placement; they are
different inputs. Configure, query the resulting peer requirements, and size
the shared region for both resources. Graph-related strides and tilings must
also satisfy :ref:`wl-architecture-compilation`.
The :ref:`wl-usage-external-aliases` example gives the complete
allocation sequence, including alignment and memory-type selection; its image
:ref:`wl-usage-image-layouts` section covers the entry/return-layout contract.

.. _wl-architecture-lifetimes:

Ownership through GPU completion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each Workload Library ``Session`` owns its Vulkan® data graph pipeline sessions
and their allocated memory, separately from prepared resources. Each prepared execution owns its descriptor pools/sets,
created views/samplers, intermediate objects, and any runtime-owned intermediate
or alias memory. Public allocations and externally supplied alias regions retain
their original owners. Keep every owner needed by recorded commands alive until
submitted GPU work completes, including the prepared execution itself.

.. list-table:: Principal lifetime relationships
   :header-rows: 1
   :class: workload-architecture-table
   :widths: 30 70

   * - Object or data
     - Ownership and required lifetime
   * - ``Workload`` and backing data
     - Workload owns normalized metadata and file mappings. Borrowed VGF bytes
       and graph constant payloads remain valid for its lifetime. Workload
       outlives derived views, binding sets, sessions, and shared compiled state.
   * - ``Context`` and wrapped Vulkan® objects
     - Owned context objects or borrowed RAII wrappers/handles outlive dependent
       sessions, allocations, compiled state, and GPU work using them.
   * - Shared pipelines and layouts
     - Shared state owns them. Sessions retain that state independently of the
       optional ``CompiledExecution`` wrapper; context/workload remain borrowed.
   * - ``Session`` and graph-session memory
     - Session owns graph pipeline sessions, their allocations, and synchronous
       command state. It outlives binding sets, prepared executions, and GPU use.
   * - ``BindingSet``
     - Owns copied binding records and push bytes, borrows resource handles;
       need not survive preparation once its snapshot has been captured.
   * - ``PreparedExecution``
     - Owns descriptors, created views/samplers, intermediates, and runtime-owned
       memory. It borrows its session and external handles; retain it through
       completion of submitted work using that state.
   * - Public allocations, views, and samplers
     - Caller owners or Context allocation wrappers retain ownership. Keep them
       valid while borrowed by bindings/prepared executions and through GPU use.
   * - External alias backing
     - Caller owns the finite region and public objects. Prepared execution owns
       created peers, which borrow that region through GPU completion.
   * - Runtime-owned intermediate/alias backing
     - Prepared execution owns objects and allocations, including shared memory
       for entirely internal alias groups; release after their GPU use completes.
   * - Graph binary bytes
     - Borrowed: ``ImplementationManaged`` until configuration succeeds; retain
       or replace for retries. ``ApplicationBacked`` remains unchanged until all
       derived pipelines are destroyed, including shared compiled pipelines.

Recording and synchronization
-----------------------------

Commands and recording hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PreparedExecution::record()`` writes the full ordered executable sequence into
a caller-owned command buffer that is already recording and compatible with the
context's queue family. It consumes the descriptor state, resources, copied push
updates, and barrier plan built during preparation. It leaves command-buffer
completion and submission to the application. Recording checks the command buffer
and required command entry points, but the caller establishes its recording state
and queue-family compatibility. An error, including a hook exception, can leave a
partial recording; the library does not guarantee rollback.

.. uml:: puml/execution-recording.puml
   :align: center
   :alt: Application calls record. For each executable the library emits planned barriers, binds descriptors and pipeline, updates push ranges, calls the optional Before hook, records a graph or compute dispatch, and calls After. Final image barriers precede return; the application ends and submits the commands.
   :caption: Recording hooks surround dispatch after barriers, bindings, and push updates.

For each executable, recording emits its planned barriers, binds descriptor sets
and the pipeline, updates the supplied portions of compiled push-constant ranges,
and then dispatches compute work or a graph pipeline session. An optional
``RecordOptions::executableHook`` receives the command buffer, executable view,
and ``Before``/``After`` phase immediately around that dispatch. The hook runs
on the host while commands are recorded.
It can record timestamps or debug labels using caller-owned Vulkan® objects.
Hook commands and their synchronization remain the caller's responsibility.
Preserve or restore the dispatch state they need: the library does not repeat
its bindings or push updates after ``Before``.

``After`` means the dispatch command has been recorded, not that the GPU has
finished executing it. It runs before the next executable's setup or the final
image barriers.
The hooks therefore surround the dispatch rather than the whole barrier/setup
region. Their callable state is used during ``record()``; any GPU objects they
reference must survive submitted use. ``run()`` uses the recording path with
default options and does not accept recording hooks.

.. _wl-architecture-dependencies:

Resource dependencies and barriers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dependency scope is the set of resources whose accesses the library tracks
together when deciding whether to insert a barrier. Preparation tracks accesses
using stage and read/write masks for the executable kind. By default, each resource has its own dependency scope; every member of an
alias group shares one scope, regardless of byte placement.

Distinct public resources bound to the same external image handle and identical
subresource range also share image state and a dependency scope, without requiring
a memory alias group. Their descriptor-required layouts and supplied entry layouts
must be compatible; conflicting layouts or alias-group assignments fail preparation.
This identity follows the image and full range, rather than shared allocation memory.

Uses within one executable are combined for that scope. Within a memory alias
group, accesses to multiple tensor/buffer peers can coexist when they require the
same image state. Distinct image peers, or an image and an optimal tensor peer
requiring different layouts, cannot be accessed in the same executable;
preparation rejects those combinations.

An executable's ``implicitBarrier`` setting controls which of its accesses are
retained for later memory dependencies. A retained earlier access causes a
barrier before a later use when either side writes. Read-only chains need no
ordinary memory dependency. Pending writer scope survives intervening readers
until a new write or image barrier updates that scope. The producer's setting
therefore matters even when the consumer's setting differs; disabling a setting
does not cancel all previously retained scopes or image-layout requirements.

Image entry barriers can also establish a shared-memory dependency for external
aliases. At the end, the plan adds barriers for used images whose current layout
must change to the required exit state, including pending alias writes where
needed. External consumers still need application-provided dependencies; this
exit processing does not establish a general buffer-release or cross-queue
synchronization contract.

.. _wl-architecture-image-replay:

Image entry state and replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image bindings distinguish the declared entry layout from the descriptor-required
layout. The application supplies the actual entry state. If a layout is supplied,
preparation plans the necessary entry barrier, even when the first required layout
is the same. Without a supplied layout, the caller establishes the workload's
required layout before execution. Conflicting declarations for the same bound
image/subresource range fail preparation.

If the caller declares an entry layout other than ``eUndefined`` or
``ePreinitialized``, execution restores that layout after the workload (or leaves
it unchanged when no transition is needed).
The same prepared recording can be reused when the application preserves this
entry contract between submissions. An ``eUndefined`` entry discards prior image
contents on each execution and exits in the workload-required layout. Replay
these commands only when each execution can discard the image's previous contents. An ``ePreinitialized`` entry preserves
initial contents but permits one execution of the prepared commands, because it
cannot be restored to that entry layout.

Optimal tensor peers can require ``eTensorAliasingARM`` transitions for images
used only as external backing. Linear tensor peers do not require those tensor-
alias layout transitions. With no explicit entry layout, internal transitions
return to the descriptor-required layout. Keep ``implicitBarrier`` enabled for
accesses followed by a required layout change, including exit restoration;
preparation rejects a layout change following an opted-out access. See :ref:`wl-usage-image-layouts` and
:ref:`wl-usage-external-aliases` for the full binding contract.

.. _wl-architecture-submission:

Submission, reuse, and caller synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PreparedExecution::run()`` waits for the session fence, resets that fence and
its command buffer, begins recording, records the workload, ends the buffer,
submits to the context queue, and waits for completion. It provides a synchronous
execution path using reusable session command/fence state. Calling ``record()``
instead lets the application combine commands and choose submission/completion
mechanisms.

.. list-table:: Execution responsibilities
   :header-rows: 1
   :class: workload-architecture-table
   :widths: 36 32 32

   * - Responsibility
     - ``run()``
     - ``record()``
   * - Workload dispatches and internal dependencies
     - Library
     - Library
   * - Command-buffer reset, begin, and end
     - Library, using session state
     - Application
   * - Queue submission and completion wait
     - Library
     - Application
   * - Synchronization with surrounding work and other queues
     - Application
     - Application
   * - External resource and allocation lifetimes
     - Application, through completion
     - Application, through completion

Re-recording reuses the snapshot described in :ref:`wl-architecture-preparation`.
Image entry-state rules in :ref:`wl-architecture-image-replay` also apply when
resubmitting an already recorded command buffer.

By default, finish one execution from a Workload Library ``Session`` before
starting another. Separate prepared executions do not remove shared graph pipeline
session state or resource hazards. Callers that overlap them must establish that
their Vulkan® implementation and resource usage support that overlap and satisfy
the concurrency and synchronization rules themselves. The library
provides no execution scheduler or locking; the application obeys command-buffer,
command-pool, queue, and memory synchronization rules. Shared pipeline ownership
is separate from safe overlapping use of session state or external resources.
The configuration serialization rules in :ref:`wl-architecture-compilation`
still apply.

The application manages uploads, output consumption, queue-family ownership
transfers, and synchronization with work outside the workload recording.
``run()`` waiting for its own submission does not supply those external
producer/consumer dependencies. Host access additionally requires the appropriate
memory visibility operations, including flush/invalidate for noncoherent memory
when required; a fence wait alone is not a host-coherency policy. Keep the prepared
execution and all borrowed owners alive through completion of submitted use.

Diagnostics and validation boundaries
-------------------------------------

Graph properties and session memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After configuration, a session can query the properties advertised by a graph
pipeline and retrieve raw property data through ``dataGraphPipelineProperty()``.
The returned ``DataGraphPipelinePropertyData`` owns its byte vector and reports
whether it contains text. Interpretation of the device's property format belongs
to the application. These queries require a configured graph executable;
unsupported properties and invalid or compute executable indices are rejected.

Graph-session memory queries report handles, allocation sizes, bind points,
object indices, memory properties, and whether an allocation serves neural
statistics. ``DataGraphSessionMemoryInfo`` is a value containing a borrowed
handle to session-owned memory; it does not transfer that allocation. Keep the
session alive and coordinate mapping/reading with GPU completion and the reported
memory properties. Raw property bytes and memory views have different ownership.

Options associated with compiled state select compute-pipeline robustness and
optional neural-statistics configuration. Neural statistics affect graph pipeline
and graph-session creation. ``requiredSessionMemoryProperties`` selects required
properties for graph-session allocations; by default ordinary session allocations
use device-local memory and statistics allocations use host-visible, coherent
memory. ``requireHostVisibleSessionMemory()`` requests host-visible, coherent
properties for all of them. The application enables the corresponding Vulkan®
extensions/features before requesting these options. These APIs expose data and
configuration, rather than providing a complete profiling/reporting workflow.
See :cpp:class:`mlsdk::workloadlib::Session` and
:cpp:struct:`mlsdk::workloadlib::SessionOptions` for query and option contracts.

The library expects trusted inputs and is not a sandbox or process-security
boundary. Its checks do not establish that arbitrary data is safe or that every
device feature, external allocation, or Vulkan® usage requirement is satisfied.
It relies on the trusted application, operating system, filesystem, Vulkan®
loader, and driver. See :doc:`security` for the trust model, :doc:`usage` for
executable examples and integration contracts, and :doc:`api` for public C++
interfaces.

Implementation map
------------------

The following paths are relative to the Workload Library repository. They map
the public execution stages to the implementation without exposing helper types
as application APIs.

* **Construction metadata:** ``src/workload_formats/vgf.cpp`` and
  ``src/workload_formats/standalone/`` populate workload descriptions.
* **Normalized workload:** ``src/internal/workload_builder.*`` and
  ``src/internal/workload_impl.hpp`` organize common metadata and public indices.
* **Compiled and session state:** ``src/session.cpp`` resolves code and configures
  execution, using ``src/internal/module_compiler.*`` and the GLSL/HLSL adapters
  in that internal directory.
* **Prepared execution:** ``src/binding_set.cpp`` captures bindings and
  ``src/prepared_execution.cpp`` prepares resources, plans dependencies, and
  records commands. ``src/context.cpp`` supplies public allocation helpers;
  ``src/internal/runtime_resource.*``, ``alias_memory.hpp``, and
  ``session_tiling.*`` support resource creation, backing checks, and tiling.
