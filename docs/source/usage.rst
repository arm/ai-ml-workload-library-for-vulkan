Using the ML Workload Library for Vulkan®
==========================================

Execution model
---------------

The library can create a runtime-owned Vulkan® context, or it can wrap Vulkan®
objects supplied by the application without taking ownership. When wrapping
objects, the application must enable the Vulkan® extensions and features required
by the workload. Similarly, the application can supply resources, or the context
can allocate them based on the workload resource requirements.

A workload is executed through the following sequence:

1. Construct a ``Workload``.
2. Create a runtime-owned ``Context``, or wrap the application's Vulkan® instance,
   physical device, device, queue family, and queue.

3. Create a ``Session``, bind any needed executable or placeholder modules or
   data-graph binaries, then call ``Session::configure()``.

   - To replace embedded code or supply missing code for one executable, call
     ``Session::bindModule(workload.executable(index), implementation)``.
   - For a placeholder module, call
     ``Session::bindModule(workload.placeholderModule(index), implementation)``.
   - For an external data-graph binary, call
     ``Session::bindDataGraphBinary(binary)``. Both standalone and VGF graphs
     can specify a storage mode and pipeline selector. The final executable
     index defaults to 0; use it to select another graph in a VGF workload.
     Multiple executables can share one binary with different selectors or use
     separate binaries.

   A data-graph binary cannot target an executable that has an
   executable-specific module binding or uses a bound placeholder module.
   Unselected graph and compute executables continue to use their workload
   modules. A VGF graph selected for binary execution may omit its shader module.

4. Create a ``BindingSet`` and bind each public resource using caller-owned or
   runtime-owned allocations.
5. Prepare a binding snapshot and either run it or record it into an
   application command buffer.

By default, each ``Session`` owns its compiled pipelines and layouts. To share
them across sessions for the same context and workload, pass a
``CompiledExecution`` to each session. Each session still has its own binding
sets and command state.
Sessions sharing a ``CompiledExecution`` must use matching graph tensor strides.

Each session also has its own tiling selection. Building shared pipelines fixes
graph tensor tiling and the tiling of its alias peers. Sessions created
afterward start with those choices; other tiling choices remain independent. A
session created earlier must select compatible graph tensor tiling before it
can reuse the pipelines.

The following excerpt from the standalone compute sample shows the common flow
with a runtime-owned context and buffers. The sample defines the GLSL workload
description and mapped-memory helpers earlier in the source file.

.. literalinclude:: ../sources/samples/2_run_glsl_compute.cpp
   :language: cpp
   :start-after: // [standalone-compute-execution-begin]
   :end-before: // [standalone-compute-execution-end]
   :dedent: 8

An application can instead retain ownership of its Vulkan® objects and wrap
them in a library context:

.. literalinclude:: ../sources/samples/4_record_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [wrapped-context-begin]
   :end-before: // [wrapped-context-end]
   :dedent: 8

Complete programs showing workload inspection, VGF and standalone compute
execution, and application-owned Vulkan® context wrapping are available in the
``samples`` directory.

Workload sources
----------------

``Workload::fromVGF(...)`` loads a VGF file or decodes a caller-owned memory
buffer. ``Workload::fromComputeShader(...)`` constructs a standalone compute
workload. ``Workload::fromDataGraph(...)`` constructs a standalone Vulkan®
data-graph workload.

The standalone compute samples describe their module, dispatch, and resources
directly:

.. literalinclude:: ../sources/samples/sample_utils.hpp
   :language: cpp
   :start-after: // [compute-description-begin]
   :end-before: // [compute-description-end]

Standalone data-graph workloads use the same resource model and add graph
pipeline metadata. Sample 5 leaves its module implementation unspecified so
the Session can bind a module or binary later:

.. literalinclude:: ../sources/samples/5_create_standalone_data_graph.cpp
   :language: cpp
   :start-after: // [data-graph-description-begin]
   :end-before: // [data-graph-description-end]
   :dedent: 8

Modules can contain SPIR-V™, or GLSL and HLSL source when the matching optional
backend is available. GLSL and HLSL modules are supported for compute
executables. Check an optional backend with ``supports(Feature::GlslModules)``
or ``supports(Feature::HlslModules)``.

.. literalinclude:: ../sources/samples/2_run_glsl_compute.cpp
   :language: cpp
   :start-after: // [glsl-support-check-begin]
   :end-before: // [glsl-support-check-end]
   :dedent: 8

A workload can also contain placeholder modules. Sample 3 binds a GLSL
implementation before configuring its session:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [placeholder-module-binding-begin]
   :end-before: // [placeholder-module-binding-end]
   :dedent: 8

The VGF execution sample encodes a small in-memory VGF with a placeholder
module:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [vgf-building-begin]
   :end-before: // [vgf-building-end]

The encoded bytes remain alive while ``Workload`` decodes their caller-owned
memory:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [in-memory-vgf-loading-begin]
   :end-before: // [in-memory-vgf-loading-end]
   :dedent: 8

Workload inspection
-------------------

A workload exposes non-owning views of its public resources. Sample 1 inspects
the identity, access, kind, and kind-specific requirements of each resource:

.. literalinclude:: ../sources/samples/1_inspect_workload_metadata.cpp
   :language: cpp
   :start-after: // [resource-inspection-begin]
   :end-before: // [resource-inspection-end]

Similarly, executables expose their module and descriptor interface:

.. literalinclude:: ../sources/samples/1_inspect_workload_metadata.cpp
   :language: cpp
   :start-after: // [executable-inspection-begin]
   :end-before: // [executable-inspection-end]

``Session::workload()`` and ``BindingSet::workload()`` provide read-only access
to the original workload, which must outlive them.

``ExecutableView::specializationInfo()`` returns a copy of the original explicit
specialization entries and bytes, excluding shader defaults.
``pushConstantSize()`` returns the largest original range end (``offset + size``),
or zero when no ranges are declared.

Executable module replacement
-----------------------------

Bind replacement code with
``Session::bindModule(workload.executable(index), implementation)`` before
pipeline compilation. This overrides embedded code and placeholder bindings for
that executable, allowing steps sharing a module to use different
implementations. The original workload remains unchanged.

Replacement code must use the workload's entry point and descriptor interface
and be compatible with its push-constant layout. An executable cannot have both
a module replacement and a data-graph binary.

Successful pipeline compilation fixes the code for all sessions sharing a
``CompiledExecution``. Later bindings are rejected, and other sessions must have
no pending bindings when configured.

Resource binding
----------------

The workload exposes its public resources in order through
``Workload::resources()`` or by index through ``Workload::resource(...)``.

Choose public tensor or image tiling with ``Session::setTensorTiling()`` or
``Session::setImageTiling()`` before ``configure()``. Tensors normally default to
Linear and images to Optimal; alias groups select a compatible common tiling.

``ImageRequirements::imageType`` explicitly selects a 1D, 2D, or 3D image. When
omitted, the type is inferred as ``vk::ImageType::e3D`` if ``extent.depth > 1``,
or ``vk::ImageType::e2D`` otherwise. ``ImageRequirementsView::imageType()``
reports the resolved type; allocation helpers use it for both image and
image-view creation. Every extent component must be positive.

``Context::createImage()`` and library-created intermediate images require
uncompressed, single-plane formats with one texel per block and a concrete
extent. Their allocation sizes come from Vulkan® memory requirements.

To create and bind resources:

#. Create each public resource from its requirements, directly through Vulkan®
   or with ``Context::createTensor()``, ``createBuffer()`` or ``createImage()``.
   Use ``session.tensorTiling(resource)`` or ``session.imageTiling(resource)``
   for creation and the requirements' ``usage(tiling)`` query.

#. Bind every required public resource with ``bindTensor()``, ``bindBuffer()``,
   or ``bindImage()`` before preparing an execution.

``BindingSet`` cannot inspect a caller-owned Vulkan® handle's creation tiling.

Sample 3 binds every public resource according to its kind:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [runtime-resource-binding-begin]
   :end-before: // [runtime-resource-binding-end]

Supply ``BoundMemoryInfo`` when ``requiresBoundMemoryInfo()`` is true. Image
bindings also need a view, subresource range and any requested sampler.

``ImageRequirementsView::requiredSubresourceRange()`` derives an unspecified
aspect mask from the image format. Sampled depth/stencil images default to the
depth aspect; an explicit range can select depth or stencil individually.
Workload construction rejects aspect masks that do not match the format and
sampled depth/stencil ranges that select both aspects.

External image/tensor aliases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an application-owned image with a library-created tensor peer, follow the
Vulkan® `memory aliasing rules
<https://docs.vulkan.org/spec/latest/chapters/resources.html#resources-memory-aliasing>`_.
This example uses a single-subresource, linear RGBA8 SNORM image and an R8 SINT
tensor shaped ``[height, width, 4]``:

#. Create the unbound image with ``eAlias`` and the image requirements'
   ``usage(vk::ImageTiling::eLinear)``. Use ``ePreinitialized`` as its initial
   layout for this example.

#. Query the image layout, then configure the peer's byte strides before
   querying its allocation requirements:

   .. code-block:: cpp

      Session session(context, workload);
      const auto peer = imageResource.aliasPeer(0);
      const auto layout = image.getSubresourceLayout({vk::ImageAspectFlagBits::eColor, 0, 0});
      session.setImageTiling(imageResource, vk::ImageTiling::eLinear);
      session.setTensorStrides(peer, {static_cast<int64_t>(layout.rowPitch), 4, 1});
      session.configure();
      const auto imageReq = image.getMemoryRequirements();
      const auto peerReq = session.aliasPeerMemoryRequirements(peer);
      const auto regionSize = std::max(imageReq.size, layout.offset + peerReq.memory.size);
      const auto memoryTypes = imageReq.memoryTypeBits & peerReq.memory.memoryTypeBits;

   For a 3D image, the tensor's additional spatial stride comes from ``layout.depthPitch``.

#. Allocate ``regionSize`` bytes using a type from ``memoryTypes``. Neither
   resource may require a dedicated allocation. Choose ``baseOffset`` so the
   image binding and ``baseOffset + layout.offset`` both meet their respective
   alignments; allocate ``baseOffset + regionSize`` bytes if it is nonzero.
   Bind the image at ``baseOffset`` and create its view.

#. Supply the shared region and place the tensor at the image's subresource
   offset. Bind the remaining public resources before preparing:

   .. code-block:: cpp

      auto bindings = session.createBindingSet();
      bindings.bindImage(imageResource,
                         {*image, {*memory, baseOffset, regionSize}, *imageView, {},
                          vk::ImageLayout::ePreinitialized,
                          {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
      bindings.setAliasPeerOffset(peer, layout.offset);
      // Bind other public resources here.
      auto execution = session.prepare(bindings);
      execution.run();

For multiple peers, query each ``aliasPeer(index)`` and allocate for them all.
Bound aliases must supply the same finite ``BoundMemoryInfo`` region; peer
offsets are relative to it and default to zero. ``Context::create*()`` allocates
for one resource only. The caller guarantees that the allocation's actual memory
type is supported by every alias.

Alias groups share one memory dependency scope, independent of peer placement
offsets. One executable may access multiple tensor/buffer peers when they require
the same image layout state. Simultaneous accesses to distinct image peers, or to
both an optimal tensor and an image peer in the same group, are rejected.

Push constants
--------------

Call ``BindingSet::bindPushConstants(executable, data)`` before preparing an
execution to supply push constants for an executable from the same workload.
Each call copies the bytes and replaces that executable's previous payload,
without changing other executables' payloads.

The payload size must equal the largest ``offset + size`` of the executable's
compiled push-constant ranges, including any leading gap. For example, a
four-byte range at offset 16 requires a 20-byte payload. Empty data is accepted
only when the executable requires zero bytes; it cannot clear a nonempty
payload.

For a configured session whose first executable expects a single ``uint32_t``
at offset zero, using a binding set with its required resources already bound:

.. code-block:: cpp

   std::uint32_t value = 42;
   const auto executable = workload.executable(0);

   bindings.bindPushConstants(
       executable,
       {reinterpret_cast<const std::byte *>(&value), sizeof(value)});
   auto execution = session.prepare(bindings);

The source bytes need not remain valid after ``bindPushConstants()`` returns.
``prepare()`` captures the bound payloads, so later binding changes do not
affect an already prepared execution. ``bindings.pushConstants(executable)``
returns a copy of the currently bound payload.

Running and recording
---------------------

``PreparedExecution::run()`` records, submits, and waits for completion using
library-managed command and fence state.

``PreparedExecution::record()`` records into a caller-owned command buffer. The
application is responsible for the command-buffer lifecycle, submission, and
synchronization.

.. literalinclude:: ../sources/samples/4_record_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [recorded-execution-begin]
   :end-before: // [recorded-execution-end]
   :dedent: 8

Pass ``RecordOptions`` to the ``record()`` overload to record caller-owned
commands immediately before and after each executable dispatch. The hook
receives the recording command buffer and an ``ExecutableView`` whose index,
name, and kind identify the executable, together with an
``ExecutableRecordPhase`` indicating whether it runs before or after the
dispatch. This supports timestamps and debug labels without transferring
ownership of their Vulkan® objects to the library.

Pipeline options
----------------

``SessionOptions`` accepts a pipeline cache and optional compute-pipeline
robustness and neural statistics settings. The application must enable the
corresponding Vulkan® extensions and device features before requesting them.
Neural statistics settings are applied during graph pipeline and graph session
creation.
The pipeline cache is not used for binary-backed pipeline creation.

Ownership and lifetimes
-----------------------

- The application retains ownership of Vulkan® objects and resources it supplies.
  Runtime allocations own the Vulkan® resource and backing memory they expose.
- All bound resources, whether caller-owned or runtime-owned, must remain valid
  until submitted work has completed.
- A runtime-owned ``Context`` must outlive its sessions, allocations, and any
  submitted work that uses them.
- The Vulkan® objects wrapped by ``Context`` must outlive the context, its
  sessions, allocations created through it, and submitted work that uses them.
- A ``Workload`` must outlive its non-owning views and every ``Session`` created
  from it. A ``Session`` must outlive its binding sets and prepared executions.
- ``CompiledExecution`` is tied to the ``Context`` and ``Workload`` supplied at
  creation. Sessions keep its compiled state alive if the ``CompiledExecution``
  object is destroyed; the context and workload must still outlive those
  sessions.
- VGF memory passed to ``Workload::fromVGF(data, size)`` must remain valid for
  the workload lifetime. The same applies to caller-owned constant payloads in
  a standalone data-graph description.
- The library borrows data-graph binary bytes when they are bound to a
  ``Session``. The pointer must meet the device's
  ``dataGraphPipelineBinaryDataAlignment`` requirement. With the
  default ``DataGraphBinaryStorageMode::ImplementationManaged`` mode, keep the
  bytes valid until ``Session::configure()`` returns. If pipeline creation fails
  and is retried, keep them alive or bind a replacement binary before retrying.
  The optional identifier is copied when bound.
- With ``DataGraphBinaryStorageMode::ApplicationBacked``, Vulkan® may use the
  supplied bytes directly. Keep them accessible and unchanged until every
  pipeline created from them is destroyed, including compiled state shared by Sessions.
- ``Session::prepare()`` snapshots a binding set. Later changes to that binding
  set do not change the prepared execution.
