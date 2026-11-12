Using the ML Workload Library for Vulkan®
==========================================

See :doc:`architecture` for execution responsibilities, internal phases, and
resource ownership. This guide provides the corresponding integration examples.

Execution model
---------------

Create a Vulkan® context through the library or wrap application-owned objects.
When wrapping, enable the extensions and features required by the workload.
Both approaches are illustrated below.

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

   See :ref:`wl-architecture-binaries` for binary/module compatibility and
   :ref:`wl-usage-lifetimes` for the required binary-storage lifetime.

4. Create a ``BindingSet`` and bind each public resource using caller-owned or
   runtime-owned allocations.
5. Prepare a binding snapshot and either run it or record it into an
   application command buffer.

To reuse compiled pipelines across sessions for the same context and workload,
pass a ``CompiledExecution`` to each session. Follow the tiling, stride, and
serialization requirements in :ref:`wl-architecture-compilation`.

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

``Workload::fromVGF(path)`` loads a VGF file, while
``Workload::fromVGF(ArrayView<const std::byte>)`` decodes a caller-owned memory
buffer.
``Workload::fromComputeShader(...)`` constructs a standalone compute
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

Set ``implementation.entryPoint`` when the replacement uses a different entry
point. Omitting it preserves the workload's entry point; an explicit empty name
is rejected. This applies to executable and placeholder module bindings, and
selects the entry point for source compilation as well as pipeline creation.
GLSL source retains its required ``main`` function, which is exported under
the selected name in SPIR-V™. HLSL source defines the selected function name.
Workload module views continue to describe the original imported module.
When constructing a standalone workload, a supplied
``ModuleImplementation::entryPoint`` also takes precedence over the enclosing
description's ``entryPoint``.

Replacement code must retain the descriptor interface and be compatible with
the effective push-constant layout. An executable cannot have both a module
replacement and a data-graph binary.

Supply replacements before the first successful shared compilation; see
:ref:`wl-architecture-compilation` for binding and reuse restrictions.

Executable specialization overrides
-----------------------------------

Call ``Session::setSpecializationInfo(executable, specialization)`` before pipeline
compilation to replace one compute or data-graph executable's entire map. For
example, to select specialization constant ID 0:

.. code-block:: cpp

   const uint32_t value = 42;
   const auto* bytes = reinterpret_cast<const std::byte*>(&value);
   SpecializationInfo specialization;
   specialization.mapEntries = {{0, 0, sizeof(value)}};
   specialization.data.assign(bytes, bytes + sizeof(value));
   session.setSpecializationInfo(workload.executable(0), std::move(specialization));
   session.configure();

An empty map selects shader defaults. Steps sharing a module can use different
maps. ``ExecutableView::specializationInfo()`` continues to report the original
workload map. ``Session::specializationInfo(executable)`` returns an independent
copy of the pending map, or the specialization overrides used to compile the
pipelines once pipelines exist; it does not inspect shader defaults.

Set overrides before the first successful shared compilation and follow
:ref:`wl-architecture-compilation` when reusing compiled state or retrying.
Data-graph binaries require an empty effective specialization map.

.. _wl-usage-interface-mapping:

Resource binding
----------------

The workload exposes its public resources in order through
``Workload::resources()`` or by index through ``Workload::resource(...)``.

For standalone compute workloads, public resource indices match the order of
``ComputeShaderDescription::resources``, with one distinct resource per entry.
Entries are not deduplicated, including when compatible entries are later bound
to the same external Vulkan® object. Descriptor set and binding numbers do not
determine public resource order.

Applications can map VGF model-interface binding IDs directly to public
workload resources, without inspecting the model resource table:

.. code-block:: cpp

    #include <mlworkloadlib/vgf_import_info.hpp>

    VgfImportInfo info;
    auto workload = Workload::fromVGF(path, info);
    for (const auto &input : info.inputs) {
        auto resource = workload.resource(input.publicResourceIndex);
        // Bind the application's input identified by input.binding.
    }
    // Resolve outputs through info.outputs in the same way.

Use the returned IDs to select application inputs and outputs; resolve ambiguous
IDs, conflicting bindings, and missing mappings explicitly. Do not infer interface
IDs from resource indices or descriptor locations. See
:ref:`wl-architecture-interfaces` for mapping identity, ordering, and lifetime
rules. Existing overloads omit the metadata, and standalone workloads are unaffected.

WL creates intermediate resources during preparation. When they alias external
memory, the caller still supplies allocation and placement metadata required by
the alias APIs; interface mapping does not replace that contract.

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

.. _wl-usage-external-aliases:

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

.. _wl-usage-image-layouts:

Image layouts and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distinct public resources may bind the same external image and subresource range
when their required and supplied entry layouts are compatible. Their accesses
share synchronization state; no memory alias group is needed for repeated
bindings of the same image. Conflicting required layouts, supplied entry layouts
or alias-group assignments are rejected.

Supply the image's actual entry layout in ``ImageBindingInfo::layout``, or
establish its workload-required layout before execution when omitting that field.
Choose the entry layout with reuse in mind:

- A concrete layout other than ``eUndefined`` or ``ePreinitialized`` is restored
  after execution and can be used for replay.
- ``eUndefined`` discards prior contents on every execution.
- ``ePreinitialized`` preserves initial contents but permits only one execution
  of the prepared commands, as in the external-alias example above.

Keep ``implicitBarrier`` enabled when a later layout transition or restoration
is required. Memory sharing must follow workload-declared alias groups; repeated
bindings of the same image do not require such a group. See
:ref:`wl-architecture-image-replay` for transition and replay behavior and
:ref:`wl-architecture-dependencies` for supported simultaneous alias accesses.

.. _wl-usage-push-constants:

Push constants
--------------

Before pipeline compilation, call ``Session::setPushConstantSize(executable, size)``
to replace a compute executable's push-constant layout. A nonzero size creates
one compute-stage range at offset zero; zero removes all ranges. Sizes must be
divisible by four and within the device's ``maxPushConstantsSize`` limit.
Without an override, the original workload ranges and their offsets are kept.
This allows replacement code to use push constants absent from the VGF:

.. code-block:: cpp

   auto executable = workload.executable(0);
   session.bindModule(executable, implementation);
   session.setPushConstantSize(executable, 40);
   session.configure();

``Session::pushConstantSize(executable)`` returns the effective largest range end;
``ExecutableView::pushConstantSize()`` retains the original workload value.
Layout overrides follow :ref:`wl-architecture-compilation`, including rejection
of pending overrides whose values match the compiled layout.

Call ``BindingSet::bindPushConstants(executable, data)`` before preparing an
execution to supply push constants for an executable from the same workload.
Each call copies the bytes and replaces that executable's previous payload,
without changing other executables' payloads.

The payload is indexed from byte offset zero. Its size must be divisible by
four and cannot exceed the largest compiled range end (``offset + size``).
Recording updates only the intersections of the supplied prefix with those
ranges, including their original offsets. It does not pad the payload or
change the pipeline layout. A 28-byte payload can therefore update a 36-byte
range. Empty data explicitly records no update; omitting a binding altogether
still fails preparation when an executable declares nonempty ranges.

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

The caller must establish every byte read by the shader, following Vulkan®
push-constant pipeline-layout compatibility rules. ``record()`` can preserve
compatible state established earlier in the caller's command buffer. ``run()``
starts its own command buffer and cannot inherit that state. Inspection returns
the supplied bytes, without padding.

Running and recording
---------------------

``PreparedExecution::run()`` records, submits, and waits for completion using
library-managed command and fence state.

Call ``PreparedExecution::record()`` with an already recording, queue-compatible
command buffer. The application ends and submits it, supplies synchronization
with surrounding work, and keeps the prepared execution and its resources alive
through completion. See :ref:`wl-architecture-submission` for synchronization
and reuse requirements.

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

``SessionOptions::pipelineCache`` supplies the default application pipeline cache.
Executables inherit this cache. Before shared pipeline compilation, call
``Session::setExecutableOptions(executable, options)``
with ``ExecutableOptions::pipelineCache`` to select a cache for one executable.
Passing ``{}`` or ``{nullptr}`` explicitly disables its application cache instead
of inheriting the session default. Each call replaces the previous options.
The selected cache is forwarded for binary-backed pipeline creation, where
Vulkan® ignores it.

Cache pointers are borrowed: keep each referenced RAII cache object at a stable
address and its Vulkan® cache valid until configuration finishes using it.
See :ref:`wl-architecture-compilation` for shared compilation, pending-input
compatibility, and retry rules.

Set ``SessionOptions::failOnCompileRequired`` to prevent compilation for all
pipeline creations. The device must support ``pipelineCreationCacheControl`` and
have it enabled. A compile-required result throws
``vk::SystemError`` with ``ePipelineCompileRequired``, even without this policy.
Preventing compilation does not guarantee an application-cache hit.

Set ``SessionOptions::collectPipelineCreationFeedback`` to collect original
creation reports. ``Session::pipelineCreationReport(executable)`` returns a copy,
or ``std::nullopt`` when collection is disabled or compilation has not published
all pipelines successfully. Compute reports have one stage; graph reports have
none, including binary-backed graphs. Check the ``eValid`` flag independently
for the pipeline and each stage before interpreting feedback. An application-cache
hit requires both ``eValid`` and ``eApplicationPipelineCacheHit``.

Failed compilation discards all reports and pipelines. Reports are shared with
the compiled state and remain available after the creating session is destroyed.
Access does not require the querying session to configure. Reuse and rejected
repeated configuration preserve the original reports. Binary-backed creation
accepts the same policy and feedback options.

``SessionOptions`` also accepts optional compute-pipeline robustness and neural
statistics settings. The application must enable the corresponding Vulkan®
extensions and device features before requesting them. Neural statistics settings
are applied during graph pipeline and graph session creation.

.. _wl-usage-lifetimes:

Ownership and lifetimes
-----------------------

Keep the workload, context, session, prepared execution, and bound resources
alive for their dependent use, including submitted GPU work. Supplied Vulkan®
objects remain application-owned. See :ref:`wl-architecture-lifetimes` for the
complete ownership and destruction order.

Caller-owned VGF bytes and standalone graph constant payloads must remain valid
for the workload lifetime. Bound binary bytes must meet the device's
``dataGraphPipelineBinaryDataAlignment`` requirement:

- With default ``ImplementationManaged`` storage, retain the bytes through
  successful ``configure()``; keep them alive or bind a replacement before retrying
  failed compilation.
- With ``ApplicationBacked`` storage, retain accessible, unchanged bytes until
  every derived pipeline is destroyed, including pipelines shared by sessions.

The binary identifier is copied when bound. See :ref:`wl-architecture-binaries`
for the binary compilation path and :ref:`wl-architecture-preparation` for
binding-snapshot behavior.
